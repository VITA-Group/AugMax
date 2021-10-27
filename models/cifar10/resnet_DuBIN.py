''' PyTorch implementation of ResNet taken from 
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
and used by 
https://github.com/TAMU-VITA/ATMC/blob/master/cifar/resnet/resnet.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cifar10.resnet_DuBN import DualBatchNorm2d

class DuBIN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`

    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
    def __init__(self, planes, ratio=0.5):
        super(DuBIN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = DualBatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

class BasicBlock_DuBIN(nn.Module):

    def __init__(self, in_planes, mid_planes, out_planes, stride=1, ibn=None):
        super(BasicBlock_DuBIN, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if ibn == 'a':
            self.bn1 = DuBIN(mid_planes)
        else:
            self.bn1 = DualBatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = DualBatchNorm2d(out_planes)

        self.IN = nn.InstanceNorm2d(out_planes, affine=True) if ibn == 'b' else None

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                DualBatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if self.IN is not None:
            out = self.IN(out)
        out = F.relu(out)
        # print(out.size())
        return out


class ResNet_DuBIN(nn.Module):
    def __init__(self, block, num_blocks, ibn_cfg=('a', 'a', 'a', None), num_classes=10, init_stride=1):
        '''
        For cifar (32*32) images, init_stride=1, num_classes=10/100;
        For Tiny ImageNet (64*64) images, init_stride=2, num_classes=200;
        See https://github.com/snu-mllab/PuzzleMix/blob/b7a795c1917a075a185aa7ea078bb1453636c2c7/models/preresnet.py#L65. 
        '''
        super(ResNet_DuBIN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=init_stride, padding=1, bias=False)
        if ibn_cfg[0] == 'b':
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
        else:
            self.bn1 = DualBatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, ibn=ibn_cfg[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, ibn=ibn_cfg[2])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, ibn=ibn_cfg[3])
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, ibn=None):
        layers = []
        layers.append(block(self.in_planes, planes, planes, stride, None if ibn == 'b' else ibn))
        self.in_planes = planes

        for i in range(1,num_blocks):
            layers.append(block(self.in_planes, planes, planes, 1, None if (ibn == 'b' and i < num_blocks-1) else ibn))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18_DuBIN(num_classes=10, init_stride=1, ibn_cfg=('a', 'a', 'a', None)):
    return ResNet_DuBIN(BasicBlock_DuBIN, [2, 2, 2, 2], ibn_cfg=ibn_cfg, num_classes=num_classes, init_stride=init_stride)

def ResNet18_DuBIN_b(num_classes=10, init_stride=1, ibn_cfg=('b', 'b', None, None)):
    return ResNet_DuBIN(BasicBlock_DuBIN, [2, 2, 2, 2], ibn_cfg=ibn_cfg, num_classes=num_classes, init_stride=init_stride)
    
def ResNet34_DuBIN(num_classes=10, init_stride=1, ibn_cfg=('a', 'a', 'a', None)):
    return ResNet_DuBIN(BasicBlock_DuBIN, [3,4,6,3], ibn_cfg=ibn_cfg, num_classes=num_classes, init_stride=init_stride)

def ResNet34_DuBIN_b(num_classes=10, init_stride=1, ibn_cfg=('b', 'b', None, None)):
    return ResNet_DuBIN(BasicBlock_DuBIN, [3,4,6,3], ibn_cfg=ibn_cfg, num_classes=num_classes, init_stride=init_stride)



if __name__ == '__main__':
    from thop import profile
    # net = ResNet34_DuBIN() # GFLOPS: 1.1615, model size: 21.2821MB
    net = ResNet34_DuBIN_b() # GFLOPS: 1.1615, model size: 21.2821MB
    x = torch.randn(1,3,32,32)
    # net = ResNet34_DuBIN(num_classes=200, init_stride=2) # GFLOPS: 1.1615, model size: 21.3796MB
    # x = torch.randn(1,3,64,64)
    flops, params = profile(net, inputs=(x, ))
    y = net(x)
    print(y.size())
    print('GFLOPS: %.4f, model size: %.4fMB' % (flops/1e9, params/1e6))
