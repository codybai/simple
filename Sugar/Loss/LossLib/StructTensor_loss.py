import torch.nn as nn
import torch
import os
class StructTensor(nn.Module):
    def __init__(self,raid=9):
        super(StructTensor, self).__init__()
        self.pad = nn.ZeroPad2d(1)
        self.avg_x = nn.AvgPool2d((3, 1), 1, (1, 0))
        self.avg_y = nn.AvgPool2d((1, 3), 1, (0, 1))
        self.blur = nn.AvgPool2d(raid,1,raid//2)
        self.relu = nn.ReLU()
    def sobel_xy(self,x):
        b,c,h,w = x.size()
        x = x.mean(1).view(b,1,h,w)

        lx = x[:,:,1:-1,0:-2]
        rx = x[:,:,1:-1,2:]

        tx = x[:,:,0:-2,1:-1]
        bx = x[:,:,2:  ,1:-1]

        rlx = rx - lx
        btx = bx - tx
        output_x = rlx + self.avg_x(rlx)*3
        output_y = btx + self.avg_y(btx)*3

        return output_x,output_y

    def forward(self, x):
        x_ = self.pad(x)

        sx,sy = self.sobel_xy(x_)

        xx = self.blur(sx*sx)
        yy = self.blur(sy*sy)
        xy = self.blur(sx*sy)

        b = -(xx + yy)
        c = xx * yy - xy * xy

        deta = b * b - 4 * c
        deta = self.relu(deta)
        deta = torch.sqrt(deta)
        l1 = (-b + deta) / 2

        x_d = xy
        y_d = xx - l1

        l_xy = (x_d ** 2 + y_d ** 2+ 1e-8) ** 0.5 + 1e-8

        direx = (x_d / l_xy )
        direy = (y_d / l_xy )

        output = torch.cat([direx,direy,direx],1)

        return output

class StructTensorLoss(nn.Module):
    def __init__(self,raid = 9,
                 loss = nn.MSELoss()):
        super(StructTensorLoss, self).__init__()
        self.st = StructTensor(raid)
        self.loss = loss

    def forward(self, input, tar):
        inputFeatures = self.st(input)
        tarFeatures = self.st(tar)
        e = self.loss(inputFeatures, tarFeatures.detach())

        return e


class GramMatrix(nn.Module):
    def __init__(self):
        super(GramMatrix, self).__init__()
        self.down = nn.MaxPool2d(4,4)
        self.pad = nn.ZeroPad2d(32)

    def forward(self, input):
        a, b, c, d = input.size()
        if c<32 or d<32:
            input = self.pad(input)
        input = self.down(input)
        a, b, c, d = input.size()  # a=batch size(=1)
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        return G.div(a * b * c * d)

class GramVGGLoss(nn.Module):
    def __init__(self,layersName = "conv1_1,conv2_1",
                 loss = nn.MSELoss(),gpu_id_list = [0]):
        super(GramVGGLoss, self).__init__()
        self.layerName = layersName
        self.vgg = VGG19Modules(layersName)
        self.loss = loss
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
        cached_file = os.path.join(model_dir, 'vgg_cpu.pth')
        self.load_state_dict(torch.load(cached_file))
        self.gpu_id_list = gpu_id_list
        self.gram = GramMatrix()

    def forward(self, input, tar):
        inputFeatures = self.vgg(input)
        tarFeatures = self.vgg(tar)

        err_vgg = 0
        length = inputFeatures.__len__()
        for i in range(length):
            e = self.loss(self.gram(inputFeatures[i]), self.gram(tarFeatures[i]).detach())
            err_vgg += e
        return err_vgg

class VGGModel(nn.Module):
    def __init__(self,layersName = "conv1_1,conv2_1",loss = nn.MSELoss(),
                 gpu_id_list = None):
        super(VGGModel, self).__init__()
        self.layerName = layersName
        self.vgg = VGG19Modules(layersName)
        self.loss = loss
        #if gpu_id_list is not None and len(gpu_id_list)>1:
        #    self.t_paralle=PF.T_parallel(gpu_id_list)
        #else:
        #    self.t_paralle = None
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
        cached_file = os.path.join(model_dir, 'vgg_cpu.pth')
        self.load_state_dict(torch.load(cached_file))


    def forward(self, input):
        #if self.t_paralle is not None:
        #    return self.t_paralle(self.vgg,input)
        #else:
        return self.vgg(input)
