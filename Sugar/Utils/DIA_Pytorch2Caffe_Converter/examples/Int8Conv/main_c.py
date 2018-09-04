#!usr/bin/env python
#-*-coding:utf-8-*-

import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
from create_model import create
def loss(a,b):
    l = (torch.abs(a - b).mean() / torch.abs(b).mean()).data[0]
    return l


def showmask(output):
    sig = torch.nn.Sigmoid()
    sig.cuda()
    output = sig(output)
    output = (output[0].cpu().data.numpy().transpose(1,2,0))*255
    return cv2.merge([output]*3)

import cv2

def create_tensor(path):
    data = cv2.imread(path)
    h,w,c = data.shape
    w,h = int(w / 1.5), int(h / 1.5)
    h=h//32*32
    w=w//32*32
    src = cv2.resize(data,(w,h) )


    data = np.array([src.transpose(2,0,1).astype(np.float32)])
    print data.shape
    tensor = torch.from_numpy(data)/127.5-1
    print tensor.size()
    v_t = Variable(tensor)
    return src,v_t

def get_model_and_input():
    model = create()
    model.load_state_dict(torch.load('/root/group-dia/wssy/CheckPointsTemp/train_seg_int/latest_net_0.pth'))
    s,images = create_tensor('/root/group-dia/wssy/1.png')
    return model, images

if __name__ =='__main__':
    path = '/root/group-dia/wssy/1.png'
    src,v_t =create_tensor(path)

    #数据准备
    bit_ = 8
    #载入已训练的网络
    model = create()
    model.load_state_dict(torch.load('/root/group-dia/wssy/CheckPointsTemp/train_seg_int/latest_net_0.pth'))
    model.eval()
    #清除BN层，由conv的weight与bias折叠
    model = ClearBN(model)
    print model
    model.cuda()
    #获取原始结果
    ori = model(v_t)
    #创建管理者abm
    abm = ABManager()
    abm.set_model(model)
    Qx, Sx, Zx = op.Q(v_t)
    #传入校正数据，此处仅1个数据
    abm.add(v_t)
    #量化网络
    abm.Q(Sx.data[0], Zx.data[0])
    #非量化模块整理
    for m in abm.layer_list:
        m.others2net()

    #返回量化模型
    q_model = abm.get_model()
    print q_model
    #获得量化结果
    output = q_model(Qx)
    #转为float与原始数据对比
    output = abm.So*(output-abm.Zo)


    #比较与展示
    sig = torch.nn.Sigmoid()
    sig.cuda()

    diff = ((sig(ori)-sig(output))**2).mean().data[0]
    ori_v = (sig(ori)**2).mean().data[0]
    out_v = (sig(output)**2).mean().data[0]

    print diff,ori_v,out_v,diff/ori_v

    output = showmask(output).astype(np.uint8)
    ori = showmask(ori).astype(np.uint8)

    cv2.imshow('',np.concatenate((src,output,ori),1))

    cv2.waitKey()







