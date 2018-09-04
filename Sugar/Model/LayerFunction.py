import torch
import torch.nn as nn
from .DIYLayer.Efficient_densenet import _DenseBlock
from .DIYLayer.IntConv import IntConv
#from Model.encoding.nn import Encoding
import CommonLayer

def encoding_(i_c,params_list):
    o_c = params_list[0]
    if len(params_list)>1:
        k_s = params_list[1]
    else:
        k_s = 3
    models = nn.Sequential(
        Encoding(12,12),
        nn.Conv2d(i_c, o_c, k_s, 1, k_s // 2)
    )
    return models



def dense_(ch,params):
    o_c = params[0]
    growth = params[1]
    layer_num = params[2]

    return Efficient_DenseBlock(ch,o_c,growth,layer_num)


def bdown_(i_c,params_list):
    o_c = params_list[0]
    is_open = params_list[1]
    l = IntConv(i_c,o_c,2,is_open)
    return l

def bup_(i_c,params_list):
    models = nn.Sequential(
        nn.UpsamplingNearest2d(2,2),
        bconv_(i_c, params_list)
    )
    return models

def bcbr_(ch,params):
    bc = bconv_(ch,params)
    b = CommonLayer.bn_(params[0],params)
    r = CommonLayer.relu_(params[0], params)
    model = nn.Sequential(bc,b,r)
    return model

def bconv_(i_c,params_list):
    o_c = params_list[0]
    is_open = params_list[2]
    l = IntConv(i_c,o_c,1,is_open)
    return l



class ResBlock(nn.Container):
    def __init__(self,i_c,params):
        super(ResBlock,self).__init__()
        self.cbr = CommonLayer.cbr_(i_c,params)
        self.c = CommonLayer.conv_(params[0],params)
        self.b = CommonLayer.bn_(params[0],params)
    def forward(self,x):
        src = x
        x = self.cbr(x)
        x = self.c(x)
        x = self.b(x)
        output = src+x
        return output




class MobileConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(MobileConv, self).__init__()
        self.conv_0=torch.nn.Conv2d(in_channels, in_channels, 3, stride,groups=in_channels,padding=1)
        self.conv_1=torch.nn.Conv2d(in_channels, out_channels, 1, 1)
    def forward(self, x):
        x = self.conv_1(self.conv_0(x))
        return x


def res_(ch,params):
    model = ResBlock(ch,params)
    return model





class DenseLayer(nn.Container):
    def __init__(self,in_c,growth,bn_size):
        super(DenseLayer, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, growth * bn_size, 1, 1, 0),
            nn.BatchNorm2d(growth * bn_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth * bn_size, growth, 3, 1, 1)
        )
    def forward(self,x):
        return torch.cat([x,self.model(x)],1)

class DenseBlock(nn.Container):
    def __init__(self,in_channels,out_channels,growth_rate,layer_num,bn_size=4):
        super(DenseBlock, self).__init__()
        self.l_list = nn.ModuleList()
        self.num = layer_num
        ii = in_channels
        self.main = nn.Sequential()
        for i in range(self.num):
            layer = DenseLayer(ii + i * growth_rate,
                               growth_rate,
                               bn_size)
            self.main.add_module('denselayer%d' % (i + 1), layer)

        self.main.add_module('conv',
                        nn.Conv2d(ii + self.num * growth_rate,
                                  out_channels,3,1,1
                                  )
                        )
    def forward(self,x):
        return self.main(x)


class Efficient_DenseBlock(nn.Container):
    def __init__(self,in_channels,out_channels,growth_rate,layer_num,bn_size=4):
        super(Efficient_DenseBlock, self).__init__()
        self.num = layer_num
        ii = in_channels
        self.main = _DenseBlock(self.num, ii, bn_size, growth_rate, 0)

        self.conv = nn.Conv2d(ii+ layer_num*growth_rate,
                                  out_channels,3,1,1
                                  )

    def forward(self,x):
        x = self.main(x)
        x = self.conv(x)
        return x

import numpy as np
from torch.autograd import Variable
if __name__ =='__main__':
    tensor = torch.FloatTensor(np.random.sample((2,64,128,128)))

    mmn = DenseBlock(64,64,12,16)

    v_t = Variable(tensor).cuda()

    mmn.cuda()
    print mmn
    for i in range(1000):

        print (mmn(v_t)[1]).size()
