import torch.nn as nn
import torch
import CommonLayer
def sdown_(i_c,params_list):
    o_c = params_list[0]
    l = ShiftConv(i_c,o_c,2)
    return l
def sup_(i_c,params_list):
    o_c = params_list[0]
    models = nn.Sequential(
        nn.UpsamplingNearest2d(2,2),
        ShiftConv(i_c, o_c)
    )
    return models

def scbr_(ch,params):
    sc = sconv_(ch,params)
    b = CommonLayer.bn_(params[0],params)
    r = CommonLayer.relu_(params[0], params)
    model = nn.Sequential(sc,b,r)
    return model

def sconv_(i_c,params_list):
    o_c = params_list[0]
    l = ShiftConv(i_c,o_c)
    return l

def sres_(ch,params):
    model = SResBlock(ch,params)
    return model

class SResBlock(nn.Container):
    def __init__(self,i_c,params):
        super(SResBlock,self).__init__()
        self.scbr = scbr_(i_c,params)
        self.sc = sconv_(params[0],params)
        self.b = CommonLayer.bn_(params[0],params)

    def forward(self,x):
        src = x
        x = self.scbr(x)
        x = self.sc(x)
        x = self.b(x)
        output = src+x
        return output

class shift3x3(nn.Module):
    def __init__(self, channels):
        super(shift3x3, self).__init__()
        self.group_num=9
        self.channels = channels
        self.A = channels // 9
        self.B = channels%9
        if self.A==0:
            self.group_num=channels

        self.zero_pad = nn.ZeroPad2d(1)

        self.n_list = [self.A for i in range(9)]
        for i in range(channels % 9):
            self.n_list[i] += 1

        self.d_x = [1, 1, 0, 1, 2, 0, 0, 2, 2]
        self.d_y = [1, 0, 1, 2, 1, 0, 2, 2, 0]
        self.d_x_end = [-1, -1, -2, -1, None, -2, -2, None, None]
        self.d_y_end = [-1, -2, -1, None, -1, -2, None, None, -2]

    def forward(self, x):
        ch_idx = 0
        x = self.zero_pad(x)
        p_list = []
        for i in range(self.group_num):
            n = self.n_list[i]
            d_x = self.d_x[i]
            d_y = self.d_y[i]
            d_x_end = self.d_x_end[i]
            d_y_end = self.d_y_end[i]
            p_list.append(x[:, ch_idx:ch_idx + n, d_y:d_y_end, d_x:d_x_end])
            ch_idx += n
        return torch.cat(p_list, 1)

class ShiftConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ShiftConv, self).__init__()
        self.shift = shift3x3(in_channels)
        self.conv  = torch.nn.Conv2d(in_channels, out_channels, 1, stride)

    def forward(self, x):
        x=self.conv(self.shift(x))
        return x

