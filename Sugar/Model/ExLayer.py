import torch
import torch.nn as nn
import CommonLayer
def xup_(i_c,params_list):
    o_c = params_list[0]
    models = nn.Sequential(
        nn.UpsamplingNearest2d(2,2),
        XConv(i_c, o_c)
    )
    return models

class XConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(XConv, self).__init__()
        self.conv2 = nn.Conv2d(in_channels,out_channels,2,1,1,2)
        self.conv1 = nn.Conv2d(in_channels,out_channels,1,1,0)

    def forward(self, x):
        x2 = self.conv2(x)
        x1 = self.conv1(x)
        return x2+x1


def psup_(i_c,params_list):
    o_c = params_list[0]
    if len(params_list)>1:
        k_s = params_list[1]
    else:
        k_s = 3
    models = nn.Sequential(
        nn.Conv2d(i_c,o_c*4,k_s,1,k_s//2),
        torch.nn.PixelShuffle(2)
    )
    return models