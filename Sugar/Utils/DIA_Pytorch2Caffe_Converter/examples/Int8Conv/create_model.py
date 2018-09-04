#!usr/bin/env python
#-*-coding:utf-8-*-

import torch.nn as nn
import torch

def loss(a,b):
    l = (torch.abs(a - b).mean() / torch.abs(b).mean()).data[0]
    return l
from collections import OrderedDict

def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)
class resblock(nn.Container):
    def __init__(self, inplanes, planes, stride=1):
        super(resblock, self).__init__()
        m = OrderedDict()
        m['conv1'] = conv3x3(inplanes, planes, stride)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv3x3(planes, planes)
        m['bn2'] = nn.BatchNorm2d(planes)
        self.group1 = nn.Sequential(m)
        self.relu= nn.Sequential(nn.ReLU(inplace=True))

    def forward(self, x):
        residual = x
        out = self.group1(x) + residual
        out = self.relu(out)
        return out

def create():
    '''
    创建一个简单的网络，5次下采样、5resblock、5次上采样
    '''
    ori_list = [nn.Conv2d(3, 16, 3, 1, 1)]
    c=16
    layer = 5
    for i in range(layer):
        ori_list.append(nn.Conv2d(c, c*2, 4, 2, 1))
        ori_list.append(nn.BatchNorm2d(c*2))
        ori_list.append(nn.ReLU())
        c*=2

    for i in range(layer):
        ori_list.append(resblock(c, c))

    for i in range(layer):
        ori_list.append(nn.UpsamplingNearest2d(2,2))
        ori_list.append(nn.Conv2d(c, c//2, 3, 1, 1))
        bn = nn.BatchNorm2d(c//2)
        bn.weight.data.fill_(0.5)
        ori_list.append(bn)
        ori_list.append(nn.ReLU())
        c/=2
    ori_list.append(nn.Conv2d(16, 1, 3, 1, 1))
    model = nn.Sequential(*ori_list)
    return model
from collections import Iterable
def p(m):
    if isinstance(m, Iterable) ==False:
        s  =  False
        for n,v in vars(m).items():
            if isinstance(v,nn.Sequential):
                s = True
        if s:
            for n, v in vars(m).items():
                p(v)
        else:
            print 'layer',m
        return
    for layer in m:
        p(layer)


if __name__ =='__main__':
    net = create()
    print hasattr(net, '__iter__')
    for layer in net:
        print 'layer',layer
    p(net)