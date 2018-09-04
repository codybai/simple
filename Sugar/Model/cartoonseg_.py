import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
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

            nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),

            nn.Conv2d(planes, outplanes, kernel_size=1),
            nn.BatchNorm2d(outplanes),
        )
        self.convm = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1),
            nn.BatchNorm2d(outplanes)
        )

        self.prelu = nn.PReLU(outplanes)

    def forward(self, x):
        residual = x
        out = self.convs(x)
        residual, indices = F.max_pool2d(residual, kernel_size=2, stride=2, return_indices=True)
        residual = self.convm(residual)
        out += residual
        out = self.prelu(out)

        return out, indices


class BottleneckDim_Res(nn.Module):  # down dim
    def __init__(self, inplanes, planes, outplanes, usePrelu):
        super(BottleneckDim_Res, self).__init__()
        self.usePrelu = usePrelu
        if self.usePrelu:
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
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),

                nn.Conv2d(planes, planes, kernel_size=3, padding=1),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),

                nn.Conv2d(planes, outplanes, kernel_size=1),
                nn.BatchNorm2d(outplanes),
            )
        self.resconv = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1),
            nn.BatchNorm2d(outplanes)
        )

        self.prelu = nn.PReLU(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.convs(x)
        residual = self.resconv(residual)
        out += residual

        if self.usePrelu:
            out = self.prelu(out)
        else:
            out = self.relu(out)

        return out


class BottleneckDim(nn.Module):  # down dim
    def __init__(self, inplanes, planes, outplanes, usePrelu):
        super(BottleneckDim, self).__init__()
        self.usePrelu = usePrelu
        if self.usePrelu:
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
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),

                nn.Conv2d(planes, planes, kernel_size=3, padding=1),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),

                nn.Conv2d(planes, outplanes, kernel_size=1),
                nn.BatchNorm2d(outplanes),
            )

        self.prelu = nn.PReLU(outplanes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.convs(x)
        out += residual
        if self.usePrelu:
            out = self.prelu(out)
        else:
            out = self.relu(out)

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

        return out


class BottleneckUp_Res(nn.Module):  # upsample
    def __init__(self, inplanes, planes, outplanes):
        super(BottleneckUp_Res, self).__init__()
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

        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1),
            nn.BatchNorm2d(outplanes)
        )

        self.uppool = nn.MaxUnpool2d(2, stride=2)

    def forward(self, x, mp_indices):
        residual = x
        out = self.convs(x)
        residual = self.conv2(residual)
        residual = self.uppool(residual, mp_indices)
        out += residual
        out = F.relu(out, inplace=True)

        return out

class BottleneckUp_Res_spacial(nn.Module):  # upsample
    def __init__(self, inplanes, planes, outplanes):
        super(BottleneckUp_Res_spacial, self).__init__()
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

        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 16, kernel_size=1), # just for bottle 5_1
            nn.BatchNorm2d(16)
        )

        self.uppool = nn.MaxUnpool2d(2, stride=2)
        self.rejection = nn.Conv2d(16,outplanes,kernel_size = 1)

    def forward(self, x, mp_indices):
        residual = x
        out = self.convs(x)
        residual = self.conv2(residual)
        residual = self.uppool(residual, mp_indices)
        residual = self.rejection(residual)
        out += residual
        out = F.relu(out, inplace=True)

        return out


# fabby ver
class EnetPhoto_V6(nn.Module):
    def __init__(self):
        super(EnetPhoto_V6, self).__init__()

        # init section
        self.init_conv = nn.Conv2d(3, 13, kernel_size=7, stride=4, padding=3)  # video
        self.init_bn = nn.BatchNorm2d(16)
        self.init_prelu = nn.PReLU(16)  # nessary

        # section 1
        self.bottle1_downDim = 16
        self.bottle1_Dim = 64
        self.bottle1_1 = BottleneckDown2(16, self.bottle1_downDim, self.bottle1_Dim)  # bottle 1_1
        self.bottle1_x = nn.Sequential(
            Bottleneck3x3(self.bottle1_Dim, self.bottle1_downDim),  # bottle 1_2
            Bottleneck3x3(self.bottle1_Dim, self.bottle1_downDim),  # bottle 1_3
            Bottleneck3x3(self.bottle1_Dim, self.bottle1_downDim),  # bootle 1_5
            Bottleneck3x3(self.bottle1_Dim, self.bottle1_downDim),  # bootle 1_5
        )

        # section 2
        self.bottle2_downDim = 32
        self.bottle2_Dim = 128
        self.bottle2_1 = BottleneckDown2(self.bottle1_Dim, self.bottle1_downDim, self.bottle2_Dim)
        self.bottle2_x = nn.Sequential(
            Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim),  # bottle 2_2
            Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim, pad=2, dilation=2),  # dilated 2  # bottle 2_3
            Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim, pad=4, dilation=4),  # dilated 8  # bottle 2_7
            Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim, pad=2, dilation=2),
            Bottleneck5x5(self.bottle2_Dim, self.bottle2_downDim),  # asymmetric 5  # bottle 2_8
        )

        # section 3
        self.bottle3_1 = BottleneckDim_Res(self.bottle2_Dim * 2, self.bottle2_downDim, self.bottle2_Dim, usePrelu=True)
        self.bottle3_x = nn.Sequential(
            Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim, pad=4, dilation=4),  # dilated 8  # bottle 2_7
            Bottleneck3x3(self.bottle2_Dim, self.bottle2_downDim, pad=2, dilation=2),
        )
        self.b3_x_up = nn.ConvTranspose2d(128, 16, kernel_size=8, padding=2, stride=4)

        # section 4
        self.bottle4_1up = BottleneckUp_Res(self.bottle2_Dim, self.bottle1_downDim, self.bottle1_Dim)
        self.bottle4_2 = BottleneckDim_Res(self.bottle1_Dim * 2, 32,  128, usePrelu=False)
        self.bottle4_3 = BottleneckDim(128, 32, 128, usePrelu=False)
        self.b4_x_up = nn.ConvTranspose2d(128, 16, kernel_size=4, padding=1, stride=2)

        # section 5
        self.bottle5_1up = BottleneckUp_Res_spacial(128, 16, 64)
        self.bottle5_2 = BottleneckDim(64, 32, 64, usePrelu=False)

        # section 6
        self.bottle6_1 = nn.ConvTranspose2d(96, 32, kernel_size=8, padding=2, stride=4)
        self.bottle6_2 = nn.Conv2d(32, 5, kernel_size=3, padding=1)

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
        init_down = torch.cat((init_out, init_mp), 1)
        # pdb.set_trace()
        init_down = self.init_bn(init_down)
        init_down = self.init_prelu(init_down)

        # section 1
        bottle1_down, bottle1_indices = self.bottle1_1(init_down)  # bottle 1_1
        bottle1_5 = self.bottle1_x(bottle1_down)

        # section 2
        bottle2_down, bottle2_indices = self.bottle2_1(bottle1_5)  # bottle 2_1
        bottle2_8 = self.bottle2_x(bottle2_down)
        # concat_2
        concat_2 = torch.cat((bottle2_8, bottle2_down), 1)

        # section3
        bottle3_1 = self.bottle3_1(concat_2)
        bottle3_8 = self.bottle3_x(bottle3_1)
        b3_x_up = self.b3_x_up(bottle3_8)

        # section4
        bottle4_1 = self.bottle4_1up(bottle3_8, bottle2_indices)
        # concat_1
        concat_1 = torch.cat((bottle1_down, bottle4_1), 1)

        bottle4_2 = self.bottle4_2(concat_1)
        bottle4_3 = self.bottle4_3(bottle4_2)
        b4_x_up = self.b4_x_up(bottle4_3)

        # section5
        bottle5_1 = self.bottle5_1up(bottle4_3, bottle1_indices)
        bottle5_2 = self.bottle5_2(bottle5_1)

        concat_3 = torch.cat((bottle5_2,b4_x_up,b3_x_up),1)

        # section6
        bottle6_1 = self.bottle6_1(concat_3)
        out = self.bottle6_2(bottle6_1)

        return out

def get_model_and_input():
    pth_name = "cartoonseg_1.1.0.pkl"
    pth_file = os.path.split(os.path.abspath(__file__))[0] +'/'+ pth_name
    print("pth file :", pth_file)
    model = EnetPhoto_V6()

    if os.path.isfile(pth_file):
        model.load_state_dict(torch.load(pth_file,map_location=lambda storage,loc: storage)['EnetPhoto'])
    else:
        print "Warning: load pth_file failed !!!"

    batch_size = 1
    channels = 3
    height = 480 
    width = 352
    images = Variable(torch.ones(batch_size,channels,height,width))
    return model, images


if __name__ == "__main__":
    images = Variable(torch.randn(1, 3, 256, 192))
    d = EnetPhoto_V6()
    print d
    # print "do forward..."
    outputs = d(images)
    print (outputs.size())  # (10, 100)
