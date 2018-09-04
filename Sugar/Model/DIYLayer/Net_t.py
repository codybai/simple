import torch.nn as nn
import torch
from Model.LayerFunction import ResBlock
class ReversalNet(nn.Container):
    def __init__(self, in_c, num,f_n=4,out_c =3):
        super(ReversalNet, self).__init__()
        self.channel_list = []
        self.A_module,ch = self.CreateEn(in_c,num,f_n)
        self.B_module = self.CreateDe(ch,num,out_c)
    def down_(self,in_c,out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.PReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.PReLU(),
        )
    def up_(self,in_c,out_c):
        return nn.Sequential(
            nn.UpsamplingNearest2d(2,2),
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.PReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.PReLU(),
        )
    def CreateEn(self,in_c,num,f_n):
        model = nn.Sequential(
            nn.Conv2d(in_c,f_n,3,1,1),
            nn.BatchNorm2d(f_n),
            nn.PReLU()
        )
        ch = f_n
        for i in range(num):
            next_ch = ch*2
            model.add_module(str(i+3),self.down_(ch,next_ch))
            ch = next_ch
        for i in range(num*4):
            model.add_module('res'+str(i),ResBlock(ch,[ch,3]))
        return model,ch
    def CreateDe(self,in_c,num,out_c=3):
        model = nn.Sequential()
        ch = in_c
        for i in range(num):
            next_ch = ch//2
            model.add_module(str(i),self.up_(ch,next_ch))
            self.channel_list.append(next_ch)
            ch = next_ch
        model.add_module(
                str(num),
                nn.Sequential(nn.Conv2d(ch, out_c, 3, 1, 1),
                              nn.Tanh())
        )
        self.channel_list.append(out_c)
        self.channel_list.reverse()
        return model
    def forward(self, x):
        output = []
        x = self.A_module(x)
        n = len(self.channel_list)
        for i,layer in enumerate(self.B_module):
            x = layer(x)
            output.append(x)
        output.reverse()
        return output


class GlobalNet(nn.Container):
    def __init__(self,params):
        super(GlobalNet,self).__init__()

        in_c = params['in_c']
        A_n = params['mini_net_width']
        M_n = params['width']
        mask_label = params['mini_out_c']
        num = params['mini_net_down']
        main_out_c = params['out_c']
        down_s_num = 4
        self.A = ReversalNet(in_c,num,A_n,mask_label)
        self.Main,ch = self.CreateA(in_c,M_n,self.A.channel_list,down_s_num)
        self.b_num = 0
        self.B = self.CreateB(ch,down_s_num,main_out_c)
        self.blur = nn.AvgPool2d(65,1,32)
        self.blur_small = nn.AvgPool2d(33,1,16)

    def down_(self,in_c,out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.PReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.PReLU(),
        )

    def up_(self,in_c,out_c):
        return nn.Sequential(
            nn.UpsamplingNearest2d(2,2),
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.PReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.PReLU(),
        )

    def CreateA(self,in_c,f_c,channel_list,num):
        model = nn.Sequential(

        )

        ch = in_c
        for i in range(num):
            cat_ch = ch + channel_list[i]
            next_ch = f_c
            if i ==0:
                model.add_module(
                    str(i),
                    nn.Sequential(
                        nn.Conv2d(cat_ch, next_ch, 3, 1, 1),
                        nn.BatchNorm2d(next_ch),
                        nn.PReLU()
                    )
                )
            else:
                model.add_module(str(i), self.down_(cat_ch, next_ch))
            ch = next_ch
        return model,ch

    def CreateB(self, in_c, num,main_out_c):
        model = nn.Sequential()
        ch = in_c

        for i in range(10):

            model.add_module('res'+str(i),ResBlock(ch,[ch,3]))
            self.b_num+=1



        for i in range(num-1):
            next_ch = ch//2
            model.add_module(str(i),self.up_(ch,next_ch))
            ch = next_ch
            self.b_num+=1
        model.add_module(
                str(num),
                nn.Sequential(nn.Conv2d(ch, main_out_c, 3, 1, 1),
                              nn.Tanh()
                              )
        )
        self.b_num+=1
        return model

    def forward(self,x,mask=None,alpha = 1):
        if mask is not None:
            d_x = x-self.blur(x)
            m_x = d_x+self.blur_small(mask)
        else:
            m_x = x
        global_list = self.A(m_x.detach())

        for i,layer in enumerate(self.Main):
            x = torch.cat((x, global_list[i]*alpha), 1)
            x = layer(x)
        for i,layer in enumerate(self.B):
            x = layer(x)

        output = x
        return output,global_list[0],m_x




