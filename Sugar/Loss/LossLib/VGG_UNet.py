import torch.nn as nn

from Model.CreateModel import CreateModel
from Model.DIYLayer.VersaLayer import MinMaxNet


class VGG_UNet(nn.Module):
    def __init__(self, model_str,cat_str):
        super(VGG_UNet, self).__init__()

        cm = CreateModel()
        self.model = cm.Create(model_str)

        if cat_str is not None:
            self.cat_index = [int(x) for x in cat_str.split(',')]


        else:
            self.cat_index = []

    def forward(self, feature_list,alpha = 0.5):
        x = feature_list[0]
        index = 1
        for i,layer in enumerate(self.model):
            x = layer(x)
            if i in self.cat_index:
                x=alpha*feature_list[index]+x*(1-alpha)
                index+=1
        return x

class VGG_mtmask6666(nn.Module):
    def __init__(self, model_str,cat_str):
        super(VGG_mtmask6666, self).__init__()

        cm = CreateModel()
        self.model = cm.Create(model_str)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        if cat_str is not None:
            self.cat_index = [int(x) for x in cat_str.split(',')]


        else:
            self.cat_index = []

    def forward(self, x):

        output = self.model(x)
        mask = self.sigmoid(output)

        return mask

class VGG_mtmask_minmax(nn.Module):
    def __init__(self, model_str,cat_str):
        super(VGG_mtmask_minmax, self).__init__()

        cm = CreateModel()
        self.model = cm.Create(model_str)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        if cat_str is not None:
            self.cat_index = [int(x) for x in cat_str.split(',')]
        else:
            self.cat_index = []

        self.minmax = MinMaxNet(None,2)
        self.conv = nn.Conv2d(2,1,3,1,1)

    def forward(self, x):

        output = self.model(x)
        output = self.minmax(output)
        output = output[:,0:1]-output[:,1:]
        output = self.sigmoid(output)
        return output