import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import logging
import models.resnet as model
BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)


def initialize_weights(*models):
    """
    Initialize Model Weights
    :param modules:
    :return:
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class Confidence(nn.Module):
    def __init__(self, num_class=19, num_output=1):
        super(Confidence, self).__init__()

        def down(in_channels, out_channels):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU()
            )

        def up(in_channels, out_channels):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                torch.nn.ReLU()
            )

        self.down1 = down(num_class, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.up3 = up(256, 128)
        self.up2 = up(128, 64)
        self.up1 = nn.ConvTranspose2d(64, num_output, kernel_size=4, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        x = self.conv(x)

        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)

        return self.sigmoid(x)


class SideOutput(nn.Module):
    def __init__(self, num_output, num_class=19, kernel_sz=None, stride=None, padding=None):
        super(SideOutput, self).__init__()
        self.conv1 = nn.Conv2d(num_output, num_class, 3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_class, 1, 1, padding=0, bias=False)
        # if kernel_sz is not None:
        #     self.upsample = nn.ConvTranspose2d(num_class, num_class, kernel_sz, stride, padding=padding, bias=False)
        # else:
        #     self.upsample = None

    def forward(self, x):
        side_output = self.conv2(self.relu(self.conv1(x)))
        # if self.upsample:
        #     side_output = self.upsample(side_output)
        return side_output


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        rate1, rate2, rate3 = tuple(atrous_rates)

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.rate1 = ASPPConv(in_channels, out_channels, rate1)
        self.rate2 = ASPPConv(in_channels, out_channels, rate2)
        self.rate3 = ASPPConv(in_channels, out_channels, rate3)
        self.tail = ASPPPooling(in_channels-256, out_channels)
        self.sigmoid = nn.Sigmoid()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x, edges):
        x_size = x.size()
        edge0, edge1, edge2, edge4 = tuple(edges)
        edge4_features = self.edge_conv(F.interpolate(self.sigmoid(edge4), x_size[2:], mode='bilinear', align_corners=True))
        edge2_features = self.edge_conv(F.interpolate(self.sigmoid(edge2), x_size[2:], mode='bilinear', align_corners=True))
        edge1_features = self.edge_conv(F.interpolate(self.sigmoid(edge1), x_size[2:], mode='bilinear', align_corners=True))
        edge0_features = self.edge_conv(F.interpolate(self.sigmoid(edge0), x_size[2:], mode='bilinear', align_corners=True))

        out_head = self.head(torch.cat([edge4_features, x], dim=1))
        out_1 = self.rate1(torch.cat([edge2_features, x], dim=1))
        out_2 = self.rate2(torch.cat([edge1_features, x], dim=1))
        out_3 = self.rate3(torch.cat([edge0_features, x], dim=1))
        out_tail = self.tail(x)
        return self.project(torch.cat([out_head, out_1, out_2, out_3, out_tail], dim=1))


class Net(nn.Module):
    def __init__(self, config, **kwargs):
        super(Net, self).__init__()
        layers_ = config.MODEL.LAYERS
        pretrained = config.MODEL.PRETRAINED
        classes = config.MODEL.classes
        atrous_rates = config.MODEL.atrous_rates

        if layers_ == 50:
            resnet = model.resnet50(pretrained=pretrained)
        elif layers_ == 101:
            resnet = model.resnet101(pretrained=pretrained)
        else:
            resnet = model.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
                                    resnet.conv3, resnet.bn3, resnet.relu)
        self.max_pool = resnet.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        # del resnet

        self.side_output0 = SideOutput(128, num_class=classes)
        self.side_output1 = SideOutput(256, num_class=classes)
        self.side_output2 = SideOutput(512, num_class=classes)
        # self.side_output3 = SideOutput(1024, 1, kernel_sz=15, stride=8, padding=7)
        self.side_output4 = SideOutput(2048, num_class=classes)

        self.sigmoid = nn.Sigmoid()
        self.fuse = nn.Conv2d(4, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.confidence = Confidence(classes, 1)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.aspp = ASPP(in_channels=2048 + 256, atrous_rates=atrous_rates)

        self.bot_fine = nn.Conv2d(256, 48, kernel_size=1, bias=False)

        self.cls = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, classes, kernel_size=1)
        )

        initialize_weights(self.confidence)
        initialize_weights(self.cls)
        # if self.training:
        #     self.aux = nn.Sequential(
        #         nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
        #         nn.BatchNorm2d(256),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(256, classes, kernel_size=1)
        #     )

    def forward(self, x):
        x_size = x.size()
        assert (x_size[2]) % 8 == 0 and (x_size[3]) % 8 == 0
        # x_arr = x.cpu().numpy().transpose((0,2,3,1)).astype(np.uint8)
        # canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        # for i in range(x_size[0]):
        #     canny[i] = cv2.Canny(x_arr[i], 10, 100)
        # canny = torch.from_numpy(canny).cuda().float()

        x = self.layer0(x)
        side_output0 = self.side_output0(x)
        side_output0_wo_canny = F.interpolate(side_output0, x_size[2:], mode='bilinear', align_corners=True)
        # side_output0 = self.cw(torch.cat([side_output0_wo_canny, canny], dim=1))

        x = self.max_pool(x)

        x_layer1 = self.layer1(x)
        side_output1 = self.side_output1(x_layer1)
        side_output1_wo_canny = F.interpolate(side_output1, x_size[2:], mode='bilinear', align_corners=True)
        # side_output1 = self.cw(torch.cat([side_output1_wo_canny, canny], dim=1))

        x = self.layer2(x_layer1)
        side_output2 = self.side_output2(x)
        side_output2_wo_canny = F.interpolate(side_output2, x_size[2:], mode='bilinear', align_corners=True)
        # side_output2 = self.cw(torch.cat([side_output2_wo_canny, canny], dim=1))

        x_tmp = self.layer3(x)

        x = self.layer4(x_tmp)
        side_output4 = self.side_output4(x)
        side_output4_wo_canny = F.interpolate(side_output4, x_size[2:], mode='bilinear', align_corners=True)
        # side_output4 = self.cw(torch.cat([side_output4_wo_canny, canny], dim=1))

        preds_boundary = self.sigmoid(self.fuse(torch.cat((side_output4_wo_canny, side_output2_wo_canny, side_output1_wo_canny, side_output0_wo_canny), dim=1)))

        x = self.aspp(x, [side_output0, side_output1, side_output2, side_output4])  # 此处是原图的1/8

        dec0_fine = self.bot_fine(x_layer1)
        x = F.interpolate(x, x_layer1.size()[2:], mode='bilinear', align_corners=True)

        x = self.cls(torch.cat([dec0_fine, x], dim=1))
        x = F.interpolate(x, x_size[2:], mode='bilinear')
        # 在原图分辨率上做的
        confidence_map = self.confidence(x)

        return preds_boundary, x, confidence_map


def get_seg_model(cfg, **kwargs):
    model = Net(cfg, **kwargs)
    # model.init_weights(cfg.MODEL.PRETRAINED)

    return model


def main():
    # 把输入的分辨率得限制在8的倍数+1
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3'
    net = Net().cuda()
    input = torch.rand(4, 3, 1024, 2048).cuda()
    net.eval()
    print(net)
    preds_boundary, x, confidence_map = net(input)
    print(preds_boundary.size())
    print(x.size())
    print(confidence_map.size())


if __name__ == '__main__':
    main()