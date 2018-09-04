import torch.nn as nn
import LayerFunction
import CommonLayer
import ResLayer
import ShiftLayer
import ExLayer
from DIYLayer import ShiftBlock
class CreateModel():
    def __init__(self):
        self.layers = {}
        self.layers['Conv'] = CommonLayer.conv_
        self.layers['BN'] = CommonLayer.bn_
        self.layers['ReLU'] = CommonLayer.relu_

        self.layers['Down'] = CommonLayer.down_
        self.layers['DDown'] = CommonLayer.ddown_
        self.layers['Up'] = CommonLayer.up_
        self.layers['Tanh'] = CommonLayer.tanh_
        self.layers['Sigmoid'] = CommonLayer.sigmoid_
        self.layers['CBR'] = CommonLayer.cbr_
        self.layers['Linear'] =CommonLayer.linear_

        self.layers['Res'] = LayerFunction.res_
        self.layers['ORes'] = ResLayer.ores_
        self.layers['DRes'] = ResLayer.dres_

        self.layers['Dense'] = LayerFunction.dense_

        #self.layers['Enc'] = LayerFunction.encoding_
        self.layers['SConv'] = ShiftLayer.sconv_
        self.layers['SUp'] = ShiftLayer.sup_
        self.layers['SCBR'] = ShiftLayer.scbr_
        self.layers['SRes'] = ShiftLayer.sres_
        self.layers['SDown'] = ShiftLayer.sdown_
        self.layers['SBlock'] = ShiftBlock.sb_
        self.layers['BConv'] = LayerFunction.bconv_
        self.layers['BUp'] = LayerFunction.bup_
        self.layers['BCBR'] = LayerFunction.bcbr_
        #self.layers['BRes'] = LayerFunction.bres_
        self.layers['BDown'] = LayerFunction.bdown_

        self.layers['XUp']  = ExLayer.xup_
        self.layers['PSUp'] = ExLayer.psup_
    def _create(self,list):
        model = nn.Sequential()
        id = 0
        for layer in list:
            name = layer[0]
            params = layer[1]
            if name =='I':
                ch = params[0]
            else:
                layer = self.layers[name]
                model.add_module(str(id)+name,layer(ch,params))
                if name in 'XUp,PSUp,' \
                           'SConv,SCBR,SUp,SRes,' \
                           'Conv,CBR,Down,DDown,Up,Res,Dense,Enc,' \
                           'BConv,BCBR,BUp,BDown':
                    ch = params[0]
            id+=1
        return model

    def str_to_table(self,str_):
        table = []
        layer_list = str_.split('-')
        for layer in layer_list:
            n_p = layer.split('(')
            name = n_p[0]
            params_list = n_p[1][:-1].split(',')
            p_l = []
            for pp in params_list:
                if len(pp)>=1:
                    p_l.append(int(pp))
            table.append([name,p_l])
        return table

    def Create(self,table):
        if isinstance(table,list):
            pass
        elif isinstance(table,str):
            table = self.str_to_table(table)
        else:
            assert 'type must be list or str.'
        return self._create(table)
from torch.autograd import Variable
import torch


if __name__ =='__main__':
    cm = CreateModel()
    tensor = Variable(torch.rand((2,3,64,64))).cuda()
    model_str  = 'I(3)-CBR(32,3)-' \
                 'Down(64)-' \
                 'CBR(128,3)-' \
                 'Down(256)-' \
                 'CBR(256,3)-' \
                 'ORes()-' \
                 'ORes()-' \
                 'ORes()-' \
                 'ORes()-' \
                 'ORes()-' \
                 'Up(128,3)-' \
                 'CBR(128,3)-' \
                 'Up(64,3)-' \
                 'CBR(64,3)-' \
                 'CBR(32,3)-' \
                 'CBR(3,3)'
    model = cm.Create(model_str)
    model.cuda()

    #print model.weight.type().size()

    torch.onnx.export(model,
                      tensor,
                      "testnet.proto", verbose=True)
    with torch.autograd.profiler.profile() as prof:
        model(tensor)

    print prof