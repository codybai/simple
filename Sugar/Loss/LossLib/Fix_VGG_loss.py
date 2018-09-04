import torch.nn as nn
import torch
import os
class Fix_VGG19Modules(nn.Module):
    def __init__(self, layersName):
        super(Fix_VGG19Modules, self).__init__()
        self.features = self.make_layers(
            [64, 64, 'M',
             128, 128, 'M',
             256, 256, 256, 256, 'M',
             512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M']
        )
        self.layername = layersName.split(',')
        self.C_name_list = [
            'conv1_1', 'conv1_2',
            'conv2_1', 'conv2_2',
            'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
            'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
            'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4'
        ]
        self.R_name_list = [
            'relu1_1', 'relu1_2',
            'relu2_1', 'relu2_2',
            'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4',
            'relu4_1', 'relu4_2', 'relu4_3', 'relu4_4',
            'relu5_1', 'relu5_2', 'relu5_3', 'relu5_4'
        ]



    def Split(self):
        self.layer_modules =nn.ModuleList()
        Cindex = 0
        Rindex = 0
        lastlayer = self.layername[-1]

        temp_modules = nn.Sequential()

        for i,s in enumerate(self.features):
            temp_modules.add_module(str(i),s)
            lname = s.__class__.__name__
            if lname.find('Conv') != -1:
                if self.C_name_list[Cindex] in self.layername:
                    self.layer_modules.append(temp_modules)
                    temp_modules=nn.Sequential()
                    if self.C_name_list[Cindex] == lastlayer:
                        break
                Cindex += 1
            if lname.find('ReLU') != -1:
                if self.R_name_list[Rindex] in self.layername:
                    self.layer_modules.append(temp_modules)
                    temp_modules=nn.Sequential()
                    if self.R_name_list[Rindex] == lastlayer:
                        break
                Rindex += 1
        del self.features
    def preprocess(self,x):
        x = x.add(1).mul(127.5)
        r = x[:, 0:1, :, :]-123.680
        g = x[:, 1:2, :, :]-116.779
        b = x[:, 2:3, :, :]-103.939
        x=torch.cat((b, g, r),1)
        return x

    def make_layers(self,cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.preprocess(x)
        err_list = []
        for s in self.layer_modules:
            x = s(x)
            err_list.append(x)
        return err_list

class Fix_VGGLoss(nn.Module):
    def __init__(self,layersName = "conv1_1,conv2_1",
                 loss = nn.MSELoss(),gpu_id_list = [0]):
        super(Fix_VGGLoss, self).__init__()
        self.layerName = layersName
        self.vgg = Fix_VGG19Modules(layersName)
        self.loss = loss
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
        cached_file = os.path.join(model_dir, 'vgg_cpu.pth')
        self.load_state_dict(torch.load(cached_file))
        self.vgg.Split()
        self.gpu_id_list = gpu_id_list
        self.eval()
    def forward(self, input, tar):
        inputFeatures = self.vgg(input)
        tarFeatures = self.vgg(tar)

        err_vgg = None
        length = inputFeatures.__len__()
        for i in range(length):
            e = self.loss(inputFeatures[i], tarFeatures[i].detach())
            if err_vgg is None:
                err_vgg = e
            else:
                err_vgg += e
        return err_vgg

class Fix_VGGModel(nn.Module):
    def __init__(self,layersName = "conv1_1,conv2_1",loss = nn.MSELoss(),
                 gpu_id_list = None):
        super(Fix_VGGModel, self).__init__()
        self.layerName = layersName
        self.vgg = Fix_VGG19Modules(layersName)
        self.loss = loss
        #if gpu_id_list is not None and len(gpu_id_list)>1:
        #    self.t_paralle=PF.T_parallel(gpu_id_list)
        #else:
        #    self.t_paralle = None
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
        cached_file = os.path.join(model_dir, 'vgg_cpu.pth')
        self.load_state_dict(torch.load(cached_file))
        self.vgg.Split()

    def forward(self, input):
        #if self.t_paralle is not None:
        #    return self.t_paralle(self.vgg,input)
        #else:
        return self.vgg(input)