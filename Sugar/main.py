from Model.ModelManager import ModelManager
from Loop.LoopProcess import LoopProcess
from Option.OptionBase import OptionBase
import os
import numpy as np
import torch
import cv2
from torch.autograd import Variable
import Utils.ProcessFunctional as PF

from Loss.LossLib.VGG_loss import VGGModel
from Loss.LossLib.VGG_UNet import VGG_UNet

def function(input,models,loss,optimiz,ex):
    #-----------------------------------Set
    src = input[0]
    style = input[1]
    model_org = models[0]
    vgg = ex[0]
    gpu_id_list = ex[1]

    #--------------------------------------EnCode
    vgg_temp  = torch.nn.parallel.data_parallel(vgg, src, gpu_id_list)
    temp = [x.detach() for x in vgg_temp]
    # --------------------------------------DeCode
    output_o= torch.nn.parallel.data_parallel(model_org, temp, gpu_id_list)
    #----------------------------------------TransferMask
   # src_list = [src]
    tar_list = [src,style]
    output_group_list = [
        [output_o],                # org_fix
    ]
    Log_dict={}
    Log_dict['Loss'] = []
    for i in range(len(output_group_list)):
        l = loss.Run(
           # src_list,
            output_group_list[i],
            tar_list
        )
        l.backward()
        Log_dict['Loss'].append(1/np.log(l.data[0]))

    for opti_ in optimiz:
        opti_.step()

    Log_dict['Image'] = [[src.data]]+[
        [output_group_list[i][0].data] for i in range(len(output_group_list))
    ]

    return Log_dict

def pre(img):
    h, w, c = img.shape
    start_x = 0.3+np.random.random()*0.4
    start_y = 0.3+np.random.random()*0.4
    b_x = int((w-256)*start_x)
    b_y = int((h-256)*start_y)
    e_x = min(w,b_x+256)
    e_y = min(h,b_y+256)
    return img[b_y:e_y,b_x:e_x]

def group_resize(input):
    h,w = input[0].shape[:2]
    scale = 256./min(h,w)
    h_ = int(h*scale+0.5)
    w_ = int(w*scale+0.5)
    input += [cv2.imread('/root/Desktop/Sugar/Dataset/train/style/style.png')]
    output_list = [cv2.resize(x,(w_,h_)) for x in input]
    return output_list

if __name__ == '__main__':
    __file_name__ = __file__.split('/')[-1]
    os.environ['TORCH_MODEL_ZOO'] = __file__.replace(__file_name__,
                                                     'TORCH_MODEL_ZOO')
    opt = OptionBase({
        'Root':         '/root/Desktop/Sugar/Dataset/train',
        'key_name':     '.JPEG',
        'input_dir':    None,

        'TestRoot': '/root/group-dia/Image_net_2012/test',
        'Testkey_name': '.JPEG',
        'Testinput_dir': None,
        'Test':         False,

        'batch_size':       1,
        'continue_train': True,
        'vggLayerNames':'conv3_1,'
                        'conv4_1,conv4_2,'
                        'conv4_4,'
                        'conv5_1,conv5_2',

        'loss':         '1000000*L2(O_1,T_1)-'
                        '0.1*GVGG(O_1,T_2)-'
                        '1000000*L1(O_1,T_1)',

        'gpu_id_list': [0],
        'max_batch': 1e8,
        'epoch':    1000,
        'show_iter': 10,
        'multi_rank':   1,
        'multi_size':   2,
        'master_addr':  '10.244.26.12',
        'checkpoints':  '/root/checkpoints/vgg_decode_4_1',
    })

    mm = ModelManager(opt)

    model_str_a = 'I(512)-CBR(512,3)-' \
                  'Res(512,3)-Res(512,3)-' \
                  'Up(256,3)-CBR(256,3)-' \
                  'Up(128,3)-CBR(128,3)-' \
                  'Up(64,3)-CBR(64,3)-' \
                  'CBR(32,3)-' \
                  'CBR(16,3)-' \
                  'Conv(8,3)-Conv(8,3)-Conv(8,3)-Conv(x_value,3)-Tanh()'

    layer_name = 'relu4_1'

    mm.model_list=[
        VGG_UNet(model_str_a.replace('x_value', '3'), None)
    ]

    lp = LoopProcess(
        opt,
        mm,
        pre,
        group_resize
    )
    vgg = VGGModel(layer_name)
    lp.Run(function,ex=[vgg,opt.gpu_id_list])
