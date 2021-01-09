import torch
from torch import nn
import torch.nn.functional as F

import models.resnet as models
from itertools import chain


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


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


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
        x_down1 = self.down1(x)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)

        x = self.conv(x_down3)

        x_up3 = self.up3(x)
        x_up2 = self.up2(x_up3)
        x_up1 = self.up1(x_up2)

        return self.sigmoid(x_up1)


class PSPNet(nn.Module):
    def __init__(self, config, device, **kwargs):
        super(PSPNet, self).__init__()
        layers = config.MODEL.LAYERS
        bins = config.MODEL.bins
        dropout = config.MODEL.dropout
        classes = config.MODEL.classes
        zoom_factor = config.MODEL.zoom_factor
        use_ppm = config.MODEL.use_ppm
        pretrained = config.MODEL.PRETRAINED
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.device = device

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.conv1_my = resnet.conv1_my
        self.layer0 = nn.Sequential(resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu)
        self.max_pool = resnet.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        del resnet

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.confidence = Confidence(classes, 1)

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

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )
        initialize_weights(self.conv1_my)
        initialize_weights(self.confidence)
        initialize_weights(self.ppm)
        initialize_weights(self.cls)
        initialize_weights(self.aux)

    def forward(self, x, boundary_gt=None, error_map=None):
        x_size = x.size()
        h, w = x_size[2], x_size[3]
        assert (x_size[2]) % 8 == 0 and (x_size[3]) % 8 == 0
        if error_map is None:
            if boundary_gt is None:
                error_map = torch.ones([x_size[0], 1, x_size[2], x_size[3]])
                error_map = error_map.to(device=self.device)
            else:
                error_map = boundary_gt
                # print(error_map.shape)
            # error_map = torch.ones([x_size[0], 1, x_size[2], x_size[3]])
            # error_map = error_map.to(device=self.device)
        x = torch.cat([x, error_map], 1)
        x = self.conv1_my(x)
        x_layer0 = self.layer0(x)

        x = self.max_pool(x_layer0)
        x_layer1 = self.layer1(x)

        x = self.layer2(x_layer1)

        x_tmp = self.layer3(x)

        x_feature = self.layer4(x_tmp)

        if self.use_ppm:
            x = self.ppm(x_feature)
        else:
            x = x_feature
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        confidence_map = self.confidence(x)

        aux = self.aux(x_tmp)
        if self.zoom_factor != 1:
            aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)

        return x, aux, confidence_map

    def get_backbone_params(self):
        return chain(self.layer0.parameters(), self.max_pool.parameters(), self.layer1.parameters(),
                     self.layer2.parameters(), self.layer3.parameters(), self.layer4.parameters())

    def get_decoder_params(self):
        return chain(self.conv1_my.parameters(), self.confidence.parameters(), self.ppm.parameters(), self.cls.parameters(), self.aux.parameters())


def get_seg_model(cfg, device, **kwargs):
    model = PSPNet(cfg, device, **kwargs)
    # model.init_weights(cfg.MODEL.PRETRAINED)
    return model


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    input = torch.rand(4, 3, 473, 473).cuda()
    model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=8, use_ppm=True, pretrained=True).cuda()
    model.eval()
    print(model)
    output, boundary = model(input)
    print('PSPNet', output.size(), boundary.size())
