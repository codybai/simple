import torch.nn as nn
import torch
from Model.LayerFunction import ResBlock
from Model.CreateModel import CreateModel
class AE_NetForDIA(nn.Container):
    def __init__(self,str_create=None,loss = torch.nn.MSELoss()):
        super(AE_NetForDIA,self).__init__()
        cm = CreateModel()
        if str_create is None:
            str_create = 'I(3)-CBR(16,3)-' \
                  'Down(32)-CBR(32,3)-' \
                  'Down(64)-CBR(64,3)-' \
                  'Down(128)-CBR(128,3)-' \
                  'Down(256)-CBR(512,3)-' \
                  'Res(512,3)-Res(512,3)-' \
                  'Res(512,3)-Res(512,3)-' \
                  'Res(512,3)-Res(512,3)-' \
                  'Up(256,3)-CBR(256,3)-' \
                  'Up(128,3)-CBR(128,3)-' \
                  'Up(64,3)-CBR(64,3)-' \
                  'Up(32,3)-CBR(32,3)-' \
                  'CBR(32,3)-' \
                  'CBR(16,3)-' \
                  'Conv(3,3)-Tanh()'
        self.model = cm.Create(str_create)
        self.loss = loss
        self.num = 28



    def forward(self,x,tar=None):
        if tar is None:
            return self.model(x)

        else:
            a = x
            b = tar
            e = 0
            for i,layer in enumerate(self.model):
                if i < self.num//2 :
                    a = layer(a)
                    b = layer(b)
                    if i %3:
                        e+=self.loss(a,b.detach())

            return e




class ImageMaskNet(nn.Container):
    def __init__(self,models_a,models_b,models_c1,models_c2):
        super(ImageMaskNet,self).__init__()
        self.model_A = models_a
        self.model_B = models_b
        self.model_C1 = models_c1
        self.model_C2 = models_c2
        self.blur = torch.nn.AvgPool2d(7,1,3)
    def forward(self, x,src,tar_mask=None):
        a = self.model_A(x)
        src_ = src- self.blur(src)
        if tar_mask is None:
            temp = self.blur(a)
        else:
            temp = tar_mask
        a_src = torch.cat([temp,src_],1)
        b = self.model_B(a_src)
        c1 = self.model_C1(b)
        c2 = self.model_C2(b.detach())
        return a,c1,c2