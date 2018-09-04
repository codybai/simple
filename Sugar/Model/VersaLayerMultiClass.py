import torch.nn as nn
import torch
from LayerFunction import ResBlock
import numpy as np
from Model.CreateModel import CreateModel

class MultiLayer(nn.Container):
    def __init__(self,w,channels,num):
        super(MultiLayer, self).__init__()
        self.block_num = num
        f_layer,fc_1,fc_2 = self.create_conv_block(w, channels, num)
        self.A_1 = f_layer
        self.FC_A = fc_1
        self.FC_B = fc_2

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

    def forward(self, input):
        b,c,h,w = input.size()
        per_w = w//self.block_num
        items = input.split(per_w,3)
        itemstemp = torch.cat(items,1)
        features = torch.cat(itemstemp.split(per_w,2),1)
        x = features
        for layer in self.A_1:
            x  = layer(x)

        x = x.view(b,self.block_num**2,-1)
        x_A = self.FC_A(x).view(b,1,self.block_num,self.block_num)
        x_B = self.FC_B(x).view(b, 1, self.block_num, self.block_num)
        return x_A,x_B
class MultiClassification(nn.Container):
    def __init__(self,w,channels):
        super(MultiClassification, self).__init__()
        self.num = 5

        self.sigmoid = nn.Sigmoid()
        self.A_list=nn.ModuleList()
        self.Up_list = nn.ModuleList()
        for i in range(5):
            block_num = 2*(2**i)
            if i >0:
                ci = 1
            else:
                ci = 0
            self.A_list.append(MultiLayer(w, channels+ci, block_num))
            self.Up_list.append(nn.UpsamplingNearest2d(w//block_num,w//block_num))

    def forward(self, input,tar = None):
        b,c,h,w = input.size()

        features_list = []
        maps_list = []
        input_ = input

        for i in range(self.num):

            output = self.A_list[i](input_)
            maps = self.Up_list[i](output)
            maps_list.append(maps)
            if tar is not None and i<1:
                t_map = tar[i]
            else:
                t_map = maps
            input_ = torch.cat([input,t_map.detach()],1)

            features_list.append(output)

        maps_cat = torch.cat(maps_list,1)
        return features_list,self.sigmoid(maps_cat)






from torch.autograd import Variable

if __name__ =='__main__':
    tensor = torch.FloatTensor(np.random.sample((2,64,128,128)))

    mmn = MultiClassification(128,64)

    v_t = Variable(tensor).cuda()

    mmn.cuda()

    print (mmn(v_t)[1]).size()


