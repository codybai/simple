import torch.nn as nn
import LossLib.tv_loss as tv_loss
from LossLib.Fix_VGG_loss import Fix_VGGLoss as VGGLoss
from LossLib.VGG_loss import GramVGGLoss
from .LossLib.StructTensor_loss import StructTensorLoss
from .LossLib.Binarization_loss import BinarizationLoss
from .LossLib.GANLoss import GANLoss
from .LossFunctionProcess import LossFunctionProcess
from .LossLib.CRFLoss import CRFLoss
class CreateLoss():
    def __init__(self,opt =None):
        self.losses = {}
        self.losses['L1'] = nn.L1Loss()
        self.losses['L2'] = nn.MSELoss()
        self.losses['tv'] = tv_loss.tv_()
        self.losses['CE'] = nn.CrossEntropyLoss()
        self.losses['BCE'] = nn.BCELoss()
        self.losses['ST'] = StructTensorLoss(35)
        self.losses['BT'] = BinarizationLoss()
        self.losses['GAN'] = GANLoss()
        self.losses['CRF'] = CRFLoss()
        self.gpu_id_list = opt.gpu_id_list

        if opt is not None:
            self.add_loss('VGG', VGGLoss(opt.vggLayerNames,
                                         gpu_id_list = opt.gpu_id_list))

            self.add_loss('GVGG', GramVGGLoss(opt.vggLayerNames,
                                         gpu_id_list=opt.gpu_id_list))

            #self.add_loss('VGGC', VGGLoss(opt.vggLayerNames,loss =torch.nn.CosineSimilarity(),
            #                             gpu_id_list=opt.gpu_id_list))

    def cuda(self):
        for name in self.losses:
            self.losses[name].cuda(self.gpu_id_list[0])

    def add_loss(self,name,loss):
        self.losses[name] = loss

    def _create(self,list):
        lf = LossFunctionProcess()
        for loss in list:
            weight = loss[0]
            func_name = loss[1]
            func = self.losses[func_name]
            params_list = loss[2]
            lf.add_loss(weight,func,params_list)
        return lf

    def str_to_table(self,str_):
        table = []
        loss_list = str_.split('-')
        for loss in loss_list:
            w_f = loss.split('*')
            weight = w_f[0]
            func = w_f[1]
            f_p = func.split('(')
            func_name = f_p[0]
            func_params = f_p[1][:-1]
            params_list = func_params.split(',')
            table.append([float(weight),func_name,params_list])
        return table

    def Create(self,table):
        if isinstance(table,list):
            pass
        elif isinstance(table,str):
            table = self.str_to_table(table)
        else:
            assert 'type must be list or str.'
        return self._create(table)

if __name__ =='__main__':
    cl = CreateLoss()
    cl.add_loss('VGG',VGGLoss())
    cl.cuda()
    loss_str  = '3*VGG(S_1,T_1)-' \
                 '1*L1(S_1,T_1)-' \
                 '0.3*tv(S_1)-' \
                 '5*L2(S_2,T_2)'
    loss = cl.Create(loss_str)
