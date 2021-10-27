"""https://github.com/htwang14/augmix/blob/master/third_party/ResNeXt_DenseNet/models/resnext.py"""
import math
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from models.cifar10.resnet_DuBIN import DuBIN
from models.cifar10.resnet_DuBN import DualBatchNorm2d

class ResNeXtBottleneck_DuBN(nn.Module):
    """ResNeXt Bottleneck Block type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)."""
    expansion = 4

    def __init__(self, inplanes, planes, cardinality, base_width, stride=1, downsample=None, ibn=None):
        super(ResNeXtBottleneck_DuBN, self).__init__()

        dim = int(math.floor(planes * (base_width / 64.0)))

        self.conv_reduce = nn.Conv2d(inplanes, dim * cardinality, kernel_size=1, stride=1, padding=0, bias=False)
        if ibn == 'a':
            self.bn_reduce = DuBIN(dim * cardinality)
        else:
            self.bn_reduce = DualBatchNorm2d(dim * cardinality)

        self.conv_conv = nn.Conv2d(dim * cardinality, dim * cardinality,
            kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = DualBatchNorm2d(dim * cardinality)

        self.conv_expand = nn.Conv2d(dim * cardinality, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = DualBatchNorm2d(planes * 4)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        bottleneck = self.conv_reduce(x)
        bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)

        bottleneck = self.conv_conv(bottleneck)
        bottleneck = F.relu(self.bn(bottleneck), inplace=True)

        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand(bottleneck)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + bottleneck, inplace=True)


class CifarResNeXt_DuBIN(nn.Module):
    """ResNext optimized for the Cifar dataset, as specified in https://arxiv.org/pdf/1611.05431.pdf."""

    def __init__(self, block, depth, cardinality, base_width, num_classes, init_stride=1, ibn_cfg=('a', 'a', None)):
        super(CifarResNeXt_DuBIN, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101'
        layer_blocks = (depth - 2) // 9

        self.cardinality = cardinality
        self.base_width = base_width
        self.num_classes = num_classes

        self.conv_1_3x3 = nn.Conv2d(3, 64, kernel_size=3, stride=init_stride, padding=1, bias=False)
        self.bn_1 = DualBatchNorm2d(64)

        self.inplanes = 64
        self.stage_1 = self._make_layer(block, 64, layer_blocks, 1, ibn=ibn_cfg[0])
        self.stage_2 = self._make_layer(block, 128, layer_blocks, 2, ibn=ibn_cfg[1])
        self.stage_3 = self._make_layer(block, 256, layer_blocks, 2, ibn=ibn_cfg[2])
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, ibn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                DualBatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, self.base_width, stride, downsample, ibn=ibn))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality, self.base_width, ibn=ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def ResNeXt29_DuBIN(num_classes=10, cardinality=4, base_width=32, init_stride=1):
    model = CifarResNeXt_DuBIN(ResNeXtBottleneck_DuBN, 29, cardinality, base_width, num_classes, init_stride=init_stride)
    return model
