import torch.nn as nn
from torchvision.models import AlexNet
from torchvision.models import resnet18, resnet50, alexnet
import torch.nn.functional as F

__all__ = ['AlexNet', 'Resnet']

from framework.registry import Backbones


def init_classifier(fc):
    nn.init.xavier_uniform_(fc.weight, .1)
    nn.init.constant_(fc.bias, 0.)
    return fc


@Backbones.register('resnet50')
@Backbones.register('resnet18')
class Resnet(nn.Module):
    def __init__(self, num_classes, pretrained=False, args=None):
        super(Resnet, self).__init__()
        if '50' in args.backbone:
            print('Using resnet-50')
            resnet = resnet50(pretrained=pretrained)
            self.in_ch = 2048
        else:
            resnet = resnet18(pretrained=pretrained)
            self.in_ch = 512
        self.conv1 = resnet.conv1
        self.relu = resnet.relu
        self.bn1 = resnet.bn1
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(self.in_ch, num_classes, bias=False)
        if args.in_ch != 3:
            self.init_conv1(args.in_ch, pretrained)

    def init_conv1(self, in_ch, pretrained):
        model_inplanes = 64
        conv1 = nn.Conv2d(in_ch, model_inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        old_weights = self.conv1.weight.data
        if pretrained:
            for i in range(in_ch):
                self.conv1.weight.data[:, i, :, :] = old_weights[:, i % 3, :, :]
        self.conv1 = conv1

    def forward(self, x):
        net = self
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)

        l1 = net.layer1(x)
        l2 = net.layer2(l1)
        l3 = net.layer3(l2)
        l4 = net.layer4(l3)
        logits = self.fc(l4.mean((2, 3)))
        return x, l1, l2, l3, l4, logits

    def get_lr(self, fc_weight):
        lrs = [
            ([self.conv1, self.layer1, self.layer2, self.layer3, self.layer4], 1.0),
            (self.fc, fc_weight)
        ]
        return lrs
