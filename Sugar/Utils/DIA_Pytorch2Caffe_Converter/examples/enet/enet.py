import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os


class Bottleneck3x3(nn.Module):
    def __init__(self, inplanes, planes, pad=1, dilation=1):
        super(Bottleneck3x3, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, planes, kernel_size=3, padding=pad, dilation=dilation),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, inplanes, kernel_size=1),
            nn.BatchNorm2d(inplanes),
        )
        self.prelu = nn.PReLU(inplanes)

    def forward(self, x):
        residual = x
        out = self.convs(x)
        out += residual
        out = self.prelu(out)

        return out


class Bottleneck5x5(nn.Module):
    def __init__(self, inplanes, planes):
        super(Bottleneck5x5, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, planes, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(planes, planes, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, inplanes, kernel_size=1),
            nn.BatchNorm2d(inplanes),
        )
        self.prelu = nn.PReLU(inplanes)

    def forward(self, x):
        residual = x
        out = self.convs(x)
        out += residual
        out = self.prelu(out)

        return out


class BottleneckDown2(nn.Module):
    def __init__(self, inplanes, planes, outplanes):
        super(BottleneckDown2, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=2, stride=2),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, outplanes, kernel_size=1),
            nn.BatchNorm2d(outplanes),
        )
        self.convm = nn.Conv2d(inplanes, outplanes, kernel_size=1)
        self.prelu = nn.PReLU(outplanes)

    def forward(self, x):
        residual = x
        out = self.convs(x)
        residual, indices = F.max_pool2d(residual, kernel_size=2, stride=2, return_indices=True)
        residual = self.convm(residual)

        out += residual
        out = self.prelu(out)

        #return residual, indices
        return out, indices


class BottleneckDim(nn.Module):  # down dim
    def __init__(self, inplanes, planes, outplanes):
        super(BottleneckDim, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, outplanes, kernel_size=1),
            nn.BatchNorm2d(outplanes),
        )
        self.conv2 = nn.Conv2d(inplanes, outplanes, kernel_size=1)
        self.prelu = nn.PReLU(outplanes)

    def forward(self, x):
        residual = x
        out = self.convs(x)
        residual = self.conv2(residual)
        out += residual
        out = self.prelu(out)

        return out


class BottleneckUp(nn.Module):  # upsample
    def __init__(self, inplanes, planes, outplanes):
        super(BottleneckUp, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(planes, planes, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            nn.Conv2d(planes, outplanes, kernel_size=1),
            nn.BatchNorm2d(outplanes),
        )
        self.conv2 = nn.Conv2d(inplanes, outplanes, kernel_size=1)
        self.uppool = nn.MaxUnpool2d(2, stride=2)

    def forward(self, x, mp_indices):
        residual = x
        out = self.convs(x)
        residual = self.conv2(residual)
        residual = self.uppool(residual, mp_indices)
        out += residual
        out = F.relu(out, inplace=True)

#return residual
        return out


# fabby ver
class EnetPhoto(nn.Module):
    def __init__(self):
        super(EnetPhoto, self).__init__()

        # init section
        self.init_conv = nn.Conv2d(3, 29, kernel_size=7, stride=4, padding=3)  # photo
        self.init_bn = nn.BatchNorm2d(32)
        self.init_prelu = nn.PReLU()

        # section Init
        self.init_downDim = 16
        self.init_Dim = 64
        self.bottleInit1 = BottleneckDown2(32, self.init_downDim, self.init_Dim)
        self.bottleInitx = nn.Sequential(
            Bottleneck3x3(self.init_Dim, self.init_downDim),
            Bottleneck3x3(self.init_Dim, self.init_downDim),
        )

        # section 1
        self.bottle1_downDim = 16
        self.bottle1_Dim = 128
        self.bottle1_1 = BottleneckDown2(self.init_Dim, self.bottle1_downDim, self.bottle1_Dim)
        self.bottle1_x = nn.Sequential(
            Bottleneck3x3(self.bottle1_Dim, self.bottle1_downDim),
            Bottleneck3x3(self.bottle1_Dim, self.bottle1_downDim),
            Bottleneck3x3(self.bottle1_Dim, self.bottle1_downDim),
            Bottleneck3x3(self.bottle1_Dim, self.bottle1_downDim),
        )

        # section 2
        self.bottle2_downDim = 8
        self.bottle2_Dim = 128
        self.bottle2_1 = BottleneckDown2(self.bottle1_Dim, self.bottle1_downDim, self.bottle2_Dim)
        self.bottle2_x = nn.Sequential(
            Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim),
            Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim, pad=2, dilation=2),  # dilated 2
            Bottleneck5x5(self.bottle2_Dim, self.bottle2_downDim),  # asymmetric 5
            Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim, pad=4, dilation=4),  # dilated 4
            Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim),
            Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim, pad=8, dilation=8),  # dilated 8

            Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim),
            Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim, pad=2, dilation=2),  # dilated 2
            Bottleneck5x5(self.bottle2_Dim, self.bottle2_downDim),  # asymmetric 5
            Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim, pad=4, dilation=4),  # dilated 4
            Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim),
            Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim, pad=8, dilation=8),  # dilated 8
        )

        # section 3
        self.bottle3_1 = BottleneckDim(self.bottle2_Dim * 2, 4, self.bottle2_Dim)
        self.bottle3_1up = BottleneckUp(self.bottle2_Dim, 8, self.bottle1_Dim)
        self.bottle3_2 = BottleneckDim(self.bottle1_Dim * 2, 8, self.bottle1_Dim)
        self.bottle3_2up = BottleneckUp(self.bottle1_Dim, 8, self.init_Dim)
        self.bottle3_3 = BottleneckDim(self.init_Dim * 2, 4, self.init_Dim)
        self.bottle3_3up = BottleneckUp(self.init_Dim, 4, 32)

        self.bottle3_4 = BottleneckDim(64, 4, 32)
        self.bottle3_5 = nn.ConvTranspose2d(32, 2, kernel_size=6, padding=1, stride=4)
        self.weights_init()

    def weights_init(self):
        for idx, m in enumerate(self.modules()):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        # init section
        init_out = self.init_conv(x)
        init_mp = F.max_pool2d(x, kernel_size=4, stride=4)
        input = torch.cat((init_out, init_mp), 1)
        input = self.init_bn(input)
        input_down = self.init_prelu(input)

        # section Init
        init_down, init_indices = self.bottleInit1(input_down)
        input = self.bottleInitx(init_down)

        # section 1
        bottl1_down, bottle1_indices = self.bottle1_1(input)
        input = self.bottle1_x(bottl1_down)

        # section 2
        bottle2_down, bottle2_indices = self.bottle2_1(input)
        input = self.bottle2_x(bottle2_down)
        input = torch.cat((input, bottle2_down), 1)

        # section3
        input = self.bottle3_1(input)
        input = self.bottle3_1up(input, bottle2_indices)
        input = torch.cat((input, bottl1_down), 1)

        input = self.bottle3_2(input)
        input = self.bottle3_2up(input, bottle1_indices)
        input = torch.cat((input, init_down), 1)

        input = self.bottle3_3(input)
        input = self.bottle3_3up(input, init_indices)
        input = torch.cat((input, input_down), 1)

        input = self.bottle3_4(input)
        out = self.bottle3_5(input)

        return out
#return input
#y = [init_down, init_indices]
#return y


def get_model_and_input():
    pth_name = "enet.pth"
    pth_file = os.path.split(os.path.abspath(__file__))[0] +'/'+ pth_name
    print("pth file :", pth_file)
    model = EnetPhoto()

    if os.path.isfile(pth_file):
        model.load_state_dict(torch.load(pth_file,map_location=lambda storage,loc: storage))
    else:
        print "Warning: load pth_file failed !!!"

    batch_size = 1
    channels = 3
    height = 256
    width = 192
    images = Variable(torch.ones(batch_size,channels,height,width))
    return model, images

