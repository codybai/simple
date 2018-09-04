import torch
import torch.nn as nn
import torch.nn.parallel
import os
from torch.autograd import Variable

def get_model_and_input(pth_name):
    pth_file = os.path.split(os.path.abspath(__file__))[0] + '/' + pth_name
    print("pth file :", pth_file)
    model = En_De_Code_auto(n_ud=3, input_chn=3, min_chn=16, max_chn=128, n_res=5, n_en_conv=1, n_de_conv=1, gpu_id=[0])
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

def conv3x3(in_planes, out_planes, stride=1, group = 1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, dilation=1, groups=group)

class BasicBlock(nn.Module):
    def __init__(self, channels):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(channels, channels, 1, 2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(channels)

        self.conv2 = conv3x3(channels, channels,1, 2)
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
            main.add_module('conv_'+str(i), self.create_block(in_chn, out_chn, 3, 2, 1, 1, 2, n_en_conv))
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

    def create_block(self, in_c, out_c, k_size, stride, pad, dilation=1, groups=1, n_en_conv = 1):
        block = nn.Sequential(
            # nn.Conv2d(in_c, out_c, k_size, stride, pad, dilation, groups),
            nn.Conv2d(in_c, in_c, k_size, stride, pad, dilation, groups),
            nn.Conv2d(in_c, out_c, 1, 1, 0, 1, 1),
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
           main.add_module('deconv_'+str(i), self.create_block(in_chn, out_chn, 3, 2, 1, 1, 2, n_de_conv))
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

    def create_block(self, in_c, out_c, k_size, stride, padding, dilation=1, groups=1, n_de_conv = 1):
        if stride ==2:
            block = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(in_c, in_c, k_size, 1, padding, dilation, groups),
                nn.Conv2d(in_c, out_c, 1, 1, 0, 1, 1),
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

class En_De_Code_auto(nn.Container):
    def __init__(self, n_ud, input_chn, min_chn, max_chn, n_res, n_en_conv, n_de_conv, gpu_id):
        super(En_De_Code_auto, self).__init__()
        self.gpu_id = gpu_id[0]
        main = nn.Sequential(
            # input is nc x isize x isize
            EnCoder_auto(n_ud, input_chn, min_chn, max_chn, n_res, n_en_conv),
            DeCoder_auto(n_ud, input_chn, min_chn, max_chn, n_res, n_de_conv),
        )
        if gpu_id:
            self.main = main.cuda(self.gpu_id)
        else:
            self.main = main


    def forward(self, input):
        if  self.gpu_id and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.main, input, self.gpu_id)
        else:
            return self.main(input)

class ConcatTable(nn.Container):
    def __init__(self, netA,netB):
        super(ConcatTable, self).__init__()
        self.A = netA
        self.B = netB

    def forward(self, input):
        return [self.A(input), self.B(input)]

class FlattenTable(nn.Container):
    def __init__(self):
        super(FlattenTable, self).__init__()

    def forward(self, input):
        out = []
        while(isinstance(input,list)):
            out.append(input[1])
            input = input[0]
        out.append(input)
        return out

class Beauty_layers(nn.Container):
    def __init__(self, size,nz, nc, ngf, ngpu, n_extra_layers=0):
        super(Beauty_layers, self).__init__()
        self.ngpu = ngpu

        n_extra_layers = min(n_extra_layers,self.s(size))
        self.main = self.defineD_beauty_layers(size,nz,nc,ngf,n_extra_layers)
    def s(self,x):
        if x <=8:
            return 0

        else:
            return 1+self.s(x//2)

    def sig(self, net,ndf, nf_mult_prev, nf_mult):
        if net is None:
            net = nn.Sequential()
        net.add_module('s_1',
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1))
        net.add_module('s_2',
            nn.BatchNorm2d(ndf * nf_mult))
        net.add_module('s_3',
            nn.LeakyReLU(0.2, True))
        net.add_module('s_4',
            nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1))
        net.add_module('s_5',nn.Sigmoid())
        return net

    def defineD_beauty_layers(self, size, input_nc, output_nc, ndf, n_layers):

        if n_layers == 0:
            return None
            #defineD_pixelGAN(input_nc, output_nc, ndf)
        else:
            netD = nn.Sequential(
                nn.Conv2d(input_nc, ndf, 4, 2, 1),
                nn.LeakyReLU(0.2, True)
            )

            nf_mult_prev = min(2 ** (n_layers - 2), 8)

            nf_mult = min(2 ** (n_layers - 1), 8)
            cat1 = nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1),
                #nn.Conv2d(ndf * nf_mult, ndf * nf_mult, 4, 2, 1),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            )


            #nf_mult_prev = min(2 ** (n_layers - 1), 8)
            #nf_mult = min(2 ** n_layers, 8)
            self.sig(cat1, ndf, nf_mult, nf_mult)


            cat2 = nn.Sequential()
            self.sig(cat2, ndf, nf_mult_prev, nf_mult)


            concat = ConcatTable(cat1, cat2)

            concat_pre = concat
            for i in range(n_layers-2):

                n = n_layers - 2 - i

                nf_mult_prev = min(2 ** (n - 1), 8)
                nf_mult = min(2 ** n, 8)

                branch1 = nn.Sequential(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1),
                    nn.BatchNorm2d(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    concat_pre
                )

                branch2 = nn.Sequential()
                self.sig(branch2, ndf, nf_mult_prev, nf_mult)

                concat_temp = ConcatTable(branch1,branch2)
                concat_pre = concat_temp

            netD.add_module('c_1',concat_pre)
            netD.add_module('f_1',FlattenTable())
        return netD
    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor):
            if isinstance(self.ngpu, int) and self.ngpu > 1:
                gpu_ids = range(self.ngpu)
            elif isinstance(self.ngpu, list):
                gpu_ids = self.ngpu
        return nn.parallel.data_parallel(self.main, input, gpu_ids)
