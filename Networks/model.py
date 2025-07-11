import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

from .operations import *
from .sam import AlignModule
from .serialization import copy_state_dict
from thop import profile
from torchsummary import summary
from .quantum import MultiScaleQuantumBlock
from .Dwdconv import DWConvTranspose


class Cell(nn.Module):
    """Basic form of a cell.
    Consisted of two branches, with 2 and 6 light conv 3x3, respectively.
    There are two interaction modules in the middle and tail of the cell."""

    def __init__(self, in_channels, out_channels, genotypes):
        super(Cell, self).__init__()
        mid_channels = out_channels // 4
        self.conv1a = Conv1x1(in_channels, mid_channels)
        self.conv1b = Conv1x1(in_channels, mid_channels)

        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        # The first interaction module.
        self._op2 = OPS[genotypes[0]](mid_channels, mid_channels)

        self.conv3a = LightConv3x3(mid_channels, mid_channels)
        self.conv3b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        # The second interaction module.
        self._op3 = OPS[genotypes[1]](mid_channels, mid_channels)

        # Fusing operation.
        self.conv4a = Conv1x1Linear(mid_channels, out_channels)
        self.conv4b = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def forward(self, x):
        identity = x
        x1a = self.conv1a(x)
        x1b = self.conv1b(x)

        x2a = self.conv2a(x1a)
        x2b = self.conv2b(x1b)
        x2a, x2b = self._op2((x2a, x2b))

        x3a = self.conv3a(x2a)
        x3b = self.conv3b(x2b)
        x3a, x3b = self._op3((x3a, x3b))

        x4 = self.conv4a(x3a) + self.conv4b(x3b)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x4 + identity
        return F.relu(out)


class MSINet(nn.Module):
    """The basic structure of the proposed MSINet."""

    def __init__(self, args, num_classes, channels, genotypes):
        super(MSINet, self).__init__()
        self.num_classes = num_classes
        self.channels = channels

        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.cells = nn.ModuleList()
        # Consisted of 6 cells in total.
        for i in range(3):
            in_channels = self.channels[i]
            out_channels = self.channels[i + 1]
            print(genotypes[i * 4 : i * 4 + 4])
            self.cells += [
                Cell(in_channels, out_channels, genotypes[i * 4 : i * 4 + 2]),
                Cell(out_channels, out_channels, genotypes[i * 4 + 2 : i * 4 + 4])
            ]
            if i != 2:
                # Downsample
                self.cells += [
                    nn.Sequential(
                        Conv1x1(out_channels, out_channels),
                        nn.AvgPool2d(2, stride=2)
                    )
                ]

        self.sam_mode = 'none'
        self.align_module = AlignModule(16, 8, channels[-1])

        self._init_params()

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        for cell_idx, cell in enumerate(self.cells):
            x = cell(x)
            if cell_idx == 5:
                f_x = x

        return x, f_x

    def forward(self, x, train_transfer=False, test_transfer=False):
        x, f_x = self.featuremaps(x)

        return x
    
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def msinet_x1_0( num_classes=1000):
    genotypes, pretrained_weight = genotype_factory['msmt']
    model = MSINet(
        None,
        num_classes,
        channels=[64, 256, 384, 512],
        genotypes=genotypes
    )

    copy_state_dict(
        torch.load(
            osp.join('/shares/wenzhe/Experiments/pretrained/MSINet', pretrained_weight)
        )['state_dict'], model
        )
    return model

class DQML(nn.Module):
    def __init__(self,):
        super(DQML, self).__init__()

        self.vgg = msinet_x1_0()
        self.quantum = MultiScaleQuantumBlock(512, 512)
        self.de_pred = nn.Sequential(
                                    DWConvTranspose(512,256, kernel_size=4,stride=2,padding=1, dilation=1,output_padding=0),
                                    nn.ReLU(),
                                    DWConvTranspose(256,128, kernel_size=4,stride=2,padding=1, dilation=1,output_padding=0),
                                    nn.ReLU(),
                                    DWConvTranspose(128,40, kernel_size=4,stride=2,padding=1, dilation=1,output_padding=0),
                                    nn.ReLU(),
                                    DWConvTranspose(40,1, kernel_size=4,stride=2,padding=1, dilation=1,output_padding=0),
                                    nn.ReLU(),
                                    )

    def forward(self, x):
        x1 = self.vgg(x)  
        x = self.quantum(x1)
        x = x1 * x + x1
        x = self.de_pred(x)  
        return x




