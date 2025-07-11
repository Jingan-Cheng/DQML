import torch
import torch.nn as nn
from .SPP import SPRModule

class DWConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dilation=1, output_padding=0):
        super(DWConvTranspose, self).__init__()
        self.depthwise = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, 
                                             stride=stride, padding=padding, dilation=dilation, 
                                             output_padding=output_padding, groups=in_channels)
    
        self.pointwise = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1)
        # self.spr = SPRModule(in_channels)
    
    def forward(self, x):
        # weight = self.spr(x)
        # x = x * weight
        
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
