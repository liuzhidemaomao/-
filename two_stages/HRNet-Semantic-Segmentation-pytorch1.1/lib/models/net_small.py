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
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        # print('res', res.shape)
        return self.project(res)


class Net_small(nn.Module):
    def __init__(self, config, device, **kwargs):
        super(Net_small, self).__init__()
        layers_ = config.MODEL.LAYERS
        pretrained = config.MODEL.PRETRAINED
        classes = config.MODEL.classes
        atrous_rates = config.MODEL.atrous_rates
        self.device = device

        if layers_ == 50:
            resnet = model.resnet50(pretrained=pretrained)
        elif layers_ == 34:
            resnet = model.resnet34(pretrained=pretrained)
        else:
            resnet = model.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1_my, resnet.bn1, resnet.relu)
        self.max_pool = resnet.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        # del resnet
        self.relu = nn.ReLU()
        self.confidence = Confidence(classes, 1)

        self.aspp = ASPP(in_channels=512, atrous_rates=atrous_rates)

        self.cls = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, classes, kernel_size=1)
        )

        initialize_weights(self.confidence)
        initialize_weights(self.cls)
        initialize_weights(self.aspp)

    def forward(self, x, error_map=None):
        x_size = x.size()
        assert (x_size[2]) % 8 == 0 and (x_size[3]) % 8 == 0

        if error_map is None:
            error_map = torch.zeros([x_size[0], 1, x_size[2], x_size[3]])
            error_map = error_map.to(device=self.device)

        x = torch.cat([x, error_map], 1)
        x = self.layer0(x)
        x = self.max_pool(x)

        x_layer1 = self.layer1(x)

        x = self.layer2(x_layer1)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.aspp(x)  # 此处是原图的1/8

        x = self.cls(x)
        x = F.interpolate(x, x_size[2:], mode='bilinear')
        # 在原图分辨率上做的
        confidence_map = self.confidence(x)

        return x, confidence_map


def get_seg_model(cfg, device, **kwargs):
    model = Net_small(cfg, device, **kwargs)
    # model.init_weights(cfg.MODEL.PRETRAINED)

    return model


def main():
    # 把输入的分辨率得限制在8的倍数+1
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3'
    net = Net_small().cuda()
    input = torch.rand(4, 3, 1024, 2048).cuda()
    net.eval()
    print(net)
    preds_boundary, x, confidence_map = net(input)
    print(preds_boundary.size())
    print(x.size())
    print(confidence_map.size())


if __name__ == '__main__':
    main()