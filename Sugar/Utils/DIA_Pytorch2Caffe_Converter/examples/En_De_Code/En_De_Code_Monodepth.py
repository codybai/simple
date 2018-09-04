import torch
import torch.nn as nn
import torch.nn.parallel
import os
from torch.autograd import Variable

def get_model_and_input(pth_name):
    pth_file = os.path.split(os.path.abspath(__file__))[0] + '/' + pth_name
    print("pth file :", pth_file)
    model = Network()
    # model = En_De_Code(n_ud=3, input_chn=3, min_chn=32, max_chn=256, n_res=5, n_en_conv=1, n_de_conv=1, gpu_id=[0])
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
    # return functional.upsample(x, (h * ratio, w * ratio))
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

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1_1 = encoder_conv_block(3, 16)  # in:3  out:32
        self.conv1_2 = encoder_conv_block(32, 32)  # in:32  out:64
        self.conv1_3 = nn.Sequential(ResBlock(num_resconvs=3, in_planes=64, out_planes=64, kernel_size=3, stride=2))      # in:64    out:256
        self.conv1_4 = nn.Sequential(ResBlock(num_resconvs=4, in_planes=256, out_planes=128, kernel_size=3, stride=2))    # in:256   out:512
        self.conv1_5 = nn.Sequential(conv(in_planes=512, out_planes=512, kernel_size=3, stride=1),
                                     conv(in_planes=512, out_planes=512, kernel_size=3, stride=1))                        # in:512   out:512
        self.conv1_6 = nn.Sequential(ResBlock(num_resconvs=2, in_planes=512, out_planes=256, kernel_size=3, dilation=2))  # in:512   out:1024
        self.conv1_7 = nn.Sequential(ResBlock(num_resconvs=2, in_planes=1024, out_planes=512, kernel_size=3, dilation=4)) # in:1024  out:2048
        self.conv1_8 = nn.Sequential(ResBlock(num_resconvs=2, in_planes=2048, out_planes=512, kernel_size=3, dilation=2)) # in:2048  out:2048
        self.conv1_9 = nn.Sequential(conv(in_planes=2048, out_planes=1024, kernel_size=3, stride=1),
                                     conv(in_planes=1024, out_planes=1024, kernel_size=3, stride=1))                        # in:1024  out:512
        self.upconv1_3 = conv(1024, 512, 3, 1)
        self.iconv1_3 = nn.Sequential(decoder_conv_block(768, 512, 3))
        self.disp1_3 = get_disp(512, 2, 3)
        self.upconv1_2 = conv(512, 256, 3, 1)
        self.iconv1_2 = nn.Sequential(decoder_conv_block(322, 256, 3))
        self.disp1_2 = get_disp(256, 2, 3)
        self.upconv1_1 = conv(256, 128, 3, 1)
        self.iconv1_1 = nn.Sequential(decoder_conv_block(162, 128, 3))
        self.disp1_1 = get_disp(128, 2, 3)
        self.upconv1_0 = conv(128, 64, 3, 1)
        self.iconv1_0 = nn.Sequential(decoder_conv_block(66, 64, 3))
        self.disp1_0 = get_disp(64, 2, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, 0.02 / n)
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        # encoder
        conv1_1 = self.conv1_1(x)        # 1/2
        conv1_2 = self.conv1_2(conv1_1)  # 1/4
        conv1_3 = self.conv1_3(conv1_2)  # 1/8
        conv1_4 = self.conv1_4(conv1_3)  # 1/16
        conv1_5 = self.conv1_5(conv1_4)
        conv1_6 = self.conv1_6(conv1_5)
        conv1_7 = self.conv1_7(conv1_6)
        conv1_8 = self.conv1_8(conv1_7)
        conv1_9 = self.conv1_9(conv1_8)

        # decoder
        #disp1_3
        upsample1_3 = upsample_nn(conv1_9, 2)
        upconv1_3 = self.upconv1_3(upsample1_3)
        concat1_3 = torch.cat((upconv1_3, conv1_3), 1)
        iconv1_3 = self.iconv1_3(concat1_3)
        disp1_3 = self.disp1_3(iconv1_3)

        #disp1_2
        udisp1_3 = upsample_nn(disp1_3, 2)
        upsample1_2 = upsample_nn(iconv1_3, 2)
        upconv1_2 = self.upconv1_2(upsample1_2)
        concat1_2 = torch.cat((upconv1_2, conv1_2, udisp1_3), 1)
        iconv1_2 = self.iconv1_2(concat1_2)
        disp1_2 = self.disp1_2(iconv1_2)

        #disp1_1
        udisp1_2 = upsample_nn(disp1_2, 2)
        upsample1_1 = upsample_nn(iconv1_2, 2)
        upconv1_1 = self.upconv1_1(upsample1_1)
        concat1_1 = torch.cat((upconv1_1, conv1_1, udisp1_2), 1)
        iconv1_1 = self.iconv1_1(concat1_1)
        disp1_1 = self.disp1_1(iconv1_1)

        # disp1_0
        udisp1_1 = upsample_nn(disp1_1, 2)
        upsample1_0 = upsample_nn(iconv1_1, 2)
        upconv1_0 = self.upconv1_0(upsample1_0)
        concat1_0 = torch.cat((upconv1_0, udisp1_1), 1)
        iconv1_0 = self.iconv1_0(concat1_0)
        disp1_0 = self.disp1_0(iconv1_0)

        return 0.03 * disp1_0, 0.03 * disp1_1, 0.03 * disp1_2, 0.03 * disp1_3

