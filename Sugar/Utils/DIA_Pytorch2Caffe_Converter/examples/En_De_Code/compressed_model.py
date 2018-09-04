import torch
import torch.nn as nn
import torch.nn.parallel
import os
from torch.autograd import Variable

def get_model_and_input(pth_name,structure_model_name):
    pth_file = os.path.split(os.path.abspath(__file__))[0] + '/' + pth_name
    print("pth file :", pth_file)
    # define use whitch model
    model = define_G(input_nc=3, output_nc=3, ngf=64, which_model_netG=structure_model_name, gpu_ids=[0])

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

def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'AE_ud3':
        netG = En_De_Code_auto(n_ud=3, input_chn=input_nc, min_chn=32, max_chn=256, n_res=5, n_en_conv=1, n_de_conv=1, gpu_id=gpu_ids)
    elif which_model_netG == 'AE_ud3_half':
        netG = En_De_Code_auto(n_ud=3, input_chn=input_nc, min_chn=16, max_chn=128, n_res=5, n_en_conv=1, n_de_conv=1, gpu_id=gpu_ids)
    elif which_model_netG == 'AE_ud3_halfx2':
        netG = En_De_Code_auto(n_ud=3, input_chn=input_nc, min_chn=8, max_chn=64, n_res=5, n_en_conv=1, n_de_conv=1, gpu_id=gpu_ids)
    elif which_model_netG == 'AE_ud3_halfx1.5_group_1x1':
        netG = En_De_Code_auto(n_ud=3, input_chn=input_nc, min_chn=12, max_chn=96, n_res=5, n_en_conv=1, n_de_conv=1,
                               gpu_id=gpu_ids, group=True, sep=True)
    elif which_model_netG == 'AE_ud7_en1de3':
        netG = En_De_Code_auto(n_ud=7, input_chn=input_nc, min_chn=32, max_chn=512, n_res=5, n_en_conv=1, n_de_conv=3, gpu_id=gpu_ids)
    elif which_model_netG == 'AE_ud7_en1de3_morefat':
        netG = En_De_Code_auto(n_ud=7, input_chn=input_nc, min_chn=32, max_chn=768, n_res=5, n_en_conv=1, n_de_conv=3, gpu_id=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        print('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(device_id=gpu_ids[0])
    netG.apply(weights_init)
    return netG

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or  classname.find('InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    else:
        print('normalization layer [%s] is not found' % norm)
    return norm_layer

def conv3x3(in_planes, out_planes, stride=1, group = 1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, dilation=1, groups=group)

class BasicBlock(nn.Module):
    def __init__(self, channels, group):
        super(BasicBlock, self).__init__()
        if group:
            self.conv1 = conv3x3(channels, channels, 1, 4)
        else:
            self.conv1 = conv3x3(channels, channels, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(channels)
        if group:
            self.conv2 = conv3x3(channels, channels, 1, 4)
        else:
            self.conv2 = conv3x3(channels, channels, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):#   y=f(x)+x
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        return out

class EnCoder_auto(nn.Container):
    def __init__(self, n_ud, input_chn, min_chn, max_chn, n_res, n_en_conv, group, sep):
        super(EnCoder_auto, self).__init__()
        main = nn.Sequential(self.create_block(input_chn, min_chn, 9, 1, 4))
        in_chn = min_chn
        for i in range(n_ud):
            out_chn = min(in_chn*2, max_chn)
            if group:
                main.add_module('conv_' + str(i), self.create_block(in_chn, out_chn, 3, 2, 1, 1, 2, n_en_conv, sep))
            else:
                main.add_module('conv_' + str(i), self.create_block(in_chn, out_chn, 3, 2, 1, 1, 1, n_en_conv, sep))
            in_chn = out_chn
        for i in range(n_res):
            main.add_module('res_'+str(i), BasicBlock(out_chn, group))

        self.main = main

    def create_block(self, in_c, out_c, k_size, stride, pad, dilation=1, groups=1, n_en_conv = 1, sep = False):
        if sep:
            block = nn.Sequential(
                nn.Conv2d(in_c, in_c, k_size, stride, pad, dilation, groups),
                nn.Conv2d(in_c, out_c, 1, 1, 0, 1, 1),
                nn.BatchNorm2d(out_c),
                nn.PReLU(out_c),
            )
        else:
            block = nn.Sequential(
                nn.Conv2d(in_c, out_c, k_size, stride, pad, dilation, groups),
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
        return self.main(input)

class DeCoder_auto(nn.Container):
    def __init__(self, n_ud, input_chn, min_chn, max_chn, n_res, n_de_conv, group, sep):
        super(DeCoder_auto, self).__init__()
        main = nn.Sequential()
        in_chn = min(min_chn * pow(2, n_ud), max_chn)
        for i in range(n_ud):
           out_chn = min(min_chn * pow(2, n_ud-i-1), max_chn)
           if group:
               main.add_module('deconv_' + str(i), self.create_block(in_chn, out_chn, 3, 2, 1, 1, 2, n_de_conv, sep))
           else:
               main.add_module('deconv_' + str(i), self.create_block(in_chn, out_chn, 3, 2, 1, 1, 1, n_de_conv, sep))
           in_chn = out_chn
        main.add_module('last_conv', nn.Conv2d(min_chn, input_chn, 3, 1, 1))
        main.add_module('tanh', nn.Tanh())
        self.main = main

    def create_block(self, in_c, out_c, k_size, stride, padding, dilation=1, groups=1, n_de_conv = 1, sep = False):
        if stride == 2:
            if sep:
                block = nn.Sequential(
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.Conv2d(in_c, in_c, k_size, 1, padding, dilation, groups),
                    nn.Conv2d(in_c, out_c, 1, 1, 0, 1, 1),
                    nn.BatchNorm2d(out_c),
                    nn.PReLU(out_c),
                )
            else:
                block = nn.Sequential(
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.Conv2d(in_c, out_c, k_size, 1, padding, dilation, groups),
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
        return self.main(input)

class En_De_Code_auto(nn.Container):
    def __init__(self, n_ud, input_chn, min_chn, max_chn, n_res, n_en_conv, n_de_conv, gpu_id, group=False, sep=False):
        super(En_De_Code_auto, self).__init__()
        self.gpu_id = gpu_id[0]
        main = nn.Sequential(
            # input is nc x isize x isize
            EnCoder_auto(n_ud, input_chn, min_chn, max_chn, n_res, n_en_conv, group, sep),
            DeCoder_auto(n_ud, input_chn, min_chn, max_chn, n_res, n_de_conv, group, sep),
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

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if  self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)
