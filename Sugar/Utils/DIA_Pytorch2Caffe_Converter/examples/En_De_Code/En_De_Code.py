import torch
import torch.nn as nn
import torch.nn.parallel
import os
from torch.autograd import Variable

def get_model_and_input(pth_name):
    pth_file = os.path.split(os.path.abspath(__file__))[0] + '/' + pth_name
    print("pth file :", pth_file)
    model = En_De_Code(n_ud=3, input_chn=3, min_chn=32, max_chn=256, n_res=5, n_en_conv=1, n_de_conv=1, gpu_id=[0])
    if os.path.isfile(pth_file):
        model.load_state_dict(torch.load(pth_file, map_location=lambda storage, loc: storage))
    else:
        print "Warning :Load pth_file failed !!!"

    batch_size = 1
    channels = 3
    height = 512
    width = 512
    images = Variable(torch.rand(batch_size, channels, height, width))
    return model, images


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1)

class BasicBlock(nn.Module):
    def __init__(self, channels):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(channels, channels, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(channels)

        self.conv2 = conv3x3(channels, channels,1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):#   y=f(x)+x
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #out = self.prelu(out)

        out += residual
        return out

class EnCoder_auto(nn.Container):
    def __init__(self, n_ud, input_chn, min_chn, max_chn, n_res, n_en_conv):
        super(EnCoder_auto, self).__init__()
        main = nn.Sequential(self.create_block(input_chn, min_chn, 9, 1, 4))
        in_chn = min_chn
        for i in range(n_ud):
            out_chn = min(in_chn*2, max_chn)
            main.add_module('conv_'+str(i), self.create_block(in_chn, out_chn, 3, 2, 1, n_en_conv))
            in_chn = out_chn
        for i in range(n_res):
            main.add_module('res_'+str(i), BasicBlock(out_chn))

        # main = nn.Sequential(
        #     # input is nc x isize x isize
        #     self.create_bloack(  3,  32, 9, 1, 4),
        #     self.create_bloack( 32,  64, 3, 2, 1),
        #     self.create_bloack( 64, 128, 3, 2, 1),
        #     self.create_bloack(128, 256, 3, 2, 1),
        #     self.create_bloack(256, 512, 3, 2, 1),
        #     self.create_bloack(512, 768, 3, 2, 1),
        #     self.create_bloack(768, 768, 3, 2, 1),
        #     self.create_bloack(768, 768, 3, 2, 1),
        #     BasicBlock(768),
        #     BasicBlock(768),
        #     BasicBlock(768),
        #     BasicBlock(768),
        #     BasicBlock(768),
        # )
        self.main = main

    def create_block(self, in_c, out_c, k_size, stride, pad, n_en_conv = 1):
        block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k_size, stride, pad),
            nn.BatchNorm2d(out_c),
            nn.PReLU(out_c),
        )
        if n_en_conv > 1:
            for i in range(n_en_conv-1):
                block.add_module('convmore' + str(i), nn.Sequential(
                nn.Conv2d(out_c, out_c, k_size, 1, pad),
                nn.BatchNorm2d(out_c),
                nn.PReLU(out_c),
            ))
        return block

    def forward(self, input):

        # x = input
        # for s in self.main:
        #     x = s(x)
        # return x
        return self.main(input)

class DeCoder_auto(nn.Container):
    def __init__(self, n_ud, input_chn, min_chn, max_chn, n_res, n_de_conv):
        super(DeCoder_auto, self).__init__()
        main = nn.Sequential()
        in_chn = min(min_chn * pow(2, n_ud), max_chn)
        for i in range(n_ud):
           out_chn = min(min_chn * pow(2, n_ud-i-1), max_chn)
           main.add_module('deconv_'+str(i), self.create_block(in_chn, out_chn, 3, 2, 1, n_de_conv))
           in_chn = out_chn
        main.add_module('last_conv', nn.Conv2d(min_chn, input_chn, 3, 1, 1))
        main.add_module('tanh', nn.Tanh())
        # main = nn.Sequential(
        #     self.create_block(768, 768, 3, 2,1),
        #     self.create_block(768, 768, 3, 2,1),
        #     self.create_block(768, 512, 3, 2,1),
        #     self.create_block(512, 256, 3, 2,1),
        #     self.create_block(256, 128, 3, 2,1),
        #     self.create_block(128, 64, 3, 2,1),
        #     self.create_block(64, 32, 3, 2,1),
        #     nn.Conv2d(32, 3, 3, 1, 1),
        #     nn.Tanh(),
        # )
        self.main = main

    def create_block(self, in_c, out_c, k_size, stride, padding, n_de_conv = 1):
        if stride ==2:
            block = nn.Sequential(
                nn.UpsamplingNearest2d(2, 2),
                nn.Conv2d(in_c, out_c, k_size, 1, padding),
                nn.BatchNorm2d(out_c),
                nn.PReLU(out_c),
            )
            if n_de_conv>1:
                for i in range(n_de_conv-1):
                    block.add_module('deconvmore' + str(i), nn.Sequential(
                    nn.Conv2d(out_c, out_c, k_size, 1, padding),
                    nn.BatchNorm2d(out_c),
                    nn.PReLU(out_c),
                    ))
            return block
        else:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, k_size, 1, padding),
                nn.BatchNorm2d(out_c),
                nn.PReLU(out_c),
            )

    def forward(self, input):
        # x = input
        # for s in self.main:
        #     x = s(x)
        # return x
        return self.main(input)

class En_De_Code(nn.Container):
    def __init__(self, n_ud, input_chn, min_chn, max_chn, n_res, n_en_conv, n_de_conv, gpu_id):
        super(En_De_Code, self).__init__()
        main = nn.Sequential(
            # input is nc x isize x isize
            EnCoder_auto(n_ud, input_chn, min_chn, max_chn, n_res, n_en_conv),
            DeCoder_auto(n_ud, input_chn, min_chn, max_chn, n_res, n_de_conv),
        )
        if gpu_id:
            self.gpu_id = gpu_id[0]
            self.main = main.cuda(self.gpu_id)
        else:
            self.main = main


    def forward(self, input):
        if  self.gpu_id and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.main, input, self.gpu_id)
        else:
            return self.main(input)



import torch
import torch.nn as nn
from torch.nn import functional

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, use_ELU=True):
    if use_ELU:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                       padding=(kernel_size-1)//2 + dilation -1, dilation=dilation, bias=True),
                             nn.ELU(alpha=1.0, inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                       padding=(kernel_size-1)//2 + dilation - 1, dilation=dilation, bias=True))


def get_disp(in_planes, out_planes=2, kernel_size=3):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False),
                         nn.Tanh())

def upsample_nn(x, ratio):
    s = x.size()
    h = s[2]
    w = s[3]
    return functional.upsample_nearest(x, (h * ratio, w * ratio))


def encoder_conv_block(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(conv(in_planes, out_planes, kernel_size=kernel_size, stride=1, dilation=1, use_ELU=True),
                         conv(out_planes, out_planes, kernel_size=kernel_size, stride=1, dilation=1, use_ELU=True),
                         conv(out_planes, 2 * out_planes, kernel_size=kernel_size, stride=2, dilation=1, use_ELU=True),
                         )


class decoder_conv_block(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(decoder_conv_block, self).__init__()
        self.conv0 = conv(in_planes, out_planes, kernel_size=1, stride=1, dilation=1, use_ELU=False)
        self.conv1 = conv(in_planes, out_planes, kernel_size=kernel_size, stride=1, dilation=1, use_ELU=True)
        self.conv2 = conv(out_planes, out_planes, kernel_size=kernel_size, stride=1, dilation=2, use_ELU=True)
        self.conv3 = conv(out_planes, out_planes, kernel_size=kernel_size, stride=1, dilation=4, use_ELU=True)
        self.conv4 = conv(out_planes, out_planes, kernel_size=kernel_size, stride=1, dilation=8, use_ELU=False)
        self.ELU = nn.ELU(alpha=1.0, inplace=True)

    def forward(self, x):
        shortcut = self.conv0(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        out = shortcut + conv4
        conv5 = self.ELU(out)
        return conv5

class ResConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1):
        super(ResConv, self).__init__()
        self.conv0 = conv(in_planes, 4 * out_planes, kernel_size=1, stride=stride, use_ELU=False)
        self.conv1 = conv(in_planes, out_planes, kernel_size=1, stride=1)
        self.conv2 = conv(out_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.conv3 = conv(out_planes, 4 * out_planes, kernel_size=1, stride=1, use_ELU=False)
        self.ELU = nn.ELU(alpha=1.0, inplace=True)
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

    def forward(self, x):
        shortcut = []
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.in_planes != 4 * self.out_planes or self.stride == 2:
            shortcut = self.conv0(x)
        else:
            shortcut = x
        out += shortcut
        out = self.ELU(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, num_resconvs, in_planes, out_planes, kernel_size=3, stride=1, dilation=1):
        super(ResBlock, self).__init__()
        model = [ResConv(in_planes, out_planes, kernel_size=kernel_size, stride=1, dilation=dilation)]
        if stride == 1:
            for i in range(num_resconvs - 1):
                model += [ResConv(4 * out_planes, out_planes, kernel_size=kernel_size, stride=1, dilation=dilation)]
        else:
            for i in range(num_resconvs - 2):
                model += [ResConv(4 * out_planes, out_planes, kernel_size=kernel_size, stride=1, dilation=dilation)]
            model += [ResConv(4 * out_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

