#!usr/bin/env python
#-*-coding:utf-8-*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        #卷积层
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, bias=True)  #输入是 1  通道  输出是10通道   内核是5*5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()  #随机选择输入的信道  将其设为 0
        #全连接层
        self.fc1 = nn.Linear(320, 50)  #输入向量的大小为320 输出向量的大小为 50
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        #x = self.conv1(x)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  #激活函数
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
# return F.log_softmax(x)


def get_model_and_input():
    pth_name = "lenet.pth"
    pth_file = os.path.split(os.path.abspath(__file__))[0] +'/'+ pth_name
    print ("pth file :",pth_file)
    model = LeNet()

    if os.path.isfile(pth_file):
        model.load_state_dict(torch.load(pth_file,map_location=lambda storage,loc: storage))
    else:
        print "Warning :load pth_file failed !!!"

    batch_size = 1
    channels = 1
    height = 28
    width = 28
    images = Variable(torch.rand(batch_size,channels,height,width))
    return model, images


