import torch.nn as nn
def conv_(i_c,params_list):
    o_c = params_list[0]
    if len(params_list)>1:
        k_s = params_list[1]
    else:
        k_s = 3
    l = nn.Conv2d(i_c,o_c,k_s,1,k_s//2)
    return l

def down_(i_c,params_list):
    o_c = params_list[0]
    l = nn.Conv2d(i_c,o_c,4,2,1)
    return l



def ddown_(i_c,params_list):
    o_c = params_list[0]
    l = nn.Conv2d(i_c,o_c,4,2,4,3)
    return l

def up_(i_c,params_list):
    o_c = params_list[0]
    if len(params_list)>1:
        k_s = params_list[1]
    else:
        k_s = 3
    models = nn.Sequential(
        nn.UpsamplingNearest2d(2,2),
        nn.Conv2d(i_c,o_c,k_s,1,k_s//2)
    )
    return models

def bn_(ch,params):
    l = nn.BatchNorm2d(ch)
    return l

def relu_(ch,params):
    l = nn.PReLU()
    return l

def tanh_(ch,params):
    l = nn.Tanh()
    return l

def sigmoid_(ch,params):
    l = nn.Sigmoid()
    return l

def cbr_(ch,params):
    c = conv_(ch,params)
    b = bn_(params[0],params)
    r = relu_(params[0], params)
    model = nn.Sequential(
        c,
        b,
        r
    )
    return model

class view(nn.Module):
    def __init__(self,size):
        super(view,self).__init__()
        self.size = size
    def forward(self,x):
        return x.view(self.size)

def linear_(ch,params):
    l = nn.Sequential(
        view((-1,params[0])),
        nn.Linear(params[0], params[1])
    )
    return l
