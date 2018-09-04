import torch.nn as nn
import torch
from LayerFunction import ResBlock
import numpy as np
from Model.CreateModel import CreateModel

class BuildHairNet(nn.Container):
    def __init__(self,w,channels,num):
        super(BuildHairNet, self).__init__()
        self.block_num = num
        f_layer,fc_1,fc_2 = self.create_conv_block(w, channels, num)
        self.A_1 = f_layer
        self.FC_A = fc_1
        self.FC_B = fc_2
        self.blur = nn.AvgPool2d(7,1,3)
    def create_block(self,in_c,out_c,num):
        model = nn.Sequential(
            nn.Conv2d(in_c*num,out_c*num,4,2,1,groups=num),
            nn.BatchNorm2d(out_c*num),
            nn.PReLU()
        )
        return model

    def create_conv_block(self,w,channels,block_num):
        w = w//block_num
        it = int(np.log(w)/np.log(2))-3

        area = w*w
        num = block_num**2
        cc = int(2**it * 8)

        model = nn.Sequential(
            nn.Conv2d(channels*num,cc*num,1, 1, 0,groups=num),
            nn.ReLU()
        )
        for i in range(it):
            half_cc= cc//2
            model.add_module(str(i)+'down',
                             self.create_block(cc,half_cc,num)
                             )
            area = area//4
            cc = half_cc

        model_fc = nn.Sequential(
            nn.Linear(cc  * area, 512),
            nn.Linear(512, 1)
        )
        model_fc_2 = nn.Sequential(
            nn.Linear(cc  * area, 512),
            nn.Linear(512, 1)
        )
        return model,model_fc,model_fc_2

    def forward(self, src,mask):

        return x_A,x_B

from torch.autograd import Variable

if __name__ =='__main__':
    tensor = torch.FloatTensor(np.random.sample((2,64,128,128)))

    mmn = MultiClassification(128,64)

    v_t = Variable(tensor).cuda()

    mmn.cuda()

    print (mmn(v_t)[1]).size()


