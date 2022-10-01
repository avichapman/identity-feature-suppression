"""
The Wide Resnet 101 model used by:
[1] P. Chen, J. Ye, G. Chen, J. Zhao, and P.-A. Heng, “Beyond Class-Conditional Assumption: A Primary Attempt to Combat
Instance-Dependent Label Noise,” Proc. AAAI Conf. Artif. Intell., 2021, [Online].
Available: http://arxiv.org/abs/2012.05458.

Code is adapted from:
https://github.com/chenpf1025/IDN/blob/master/networks/wideresnet.py
"""

from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as functional


def conv3x3(in_planes: int, out_planes: int, stride: int = 1):
    # noinspection PyTypeChecker
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class WideBasic(nn.Module):
    def __init__(self, in_planes: int, planes: int, dropout_rate: float, stride: int = 1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        # noinspection PyTypeChecker
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        # noinspection PyTypeChecker
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            # noinspection PyTypeChecker
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(functional.relu(self.bn1(x))))
        out = self.conv2(functional.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class IdnWideResNet(nn.Module):
    """
    This is the version of the wide Resnet101 used in "Beyond Class-Conditional Assumption: A Primary Attempt to Combat
    Instance-Dependent Label Noise"
    """

    def __init__(self):
        super().__init__()
        self.in_planes = 16

        depth = 28
        widen_factor = 10
        dropout_rate = 0.0

        assert ((depth-4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)//6
        k = widen_factor

        state_counts = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, state_counts[0])
        self.layer1 = self._wide_layer(WideBasic, state_counts[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, state_counts[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, state_counts[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(state_counts[3], momentum=0.9)

        self.out_channels = state_counts[3]

    def _wide_layer(self, block: Type, planes: int, num_blocks: int, dropout_rate: float, stride: int) -> nn.Module:
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = functional.relu(self.bn1(out))
        out = functional.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return out
