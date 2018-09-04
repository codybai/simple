import torch.nn as nn
import torch
import numpy as np
from Model.CreateModel import CreateModel
class VL(nn.Container):
    def __init__(self, dict_num,input_dims,dims):
        super(VL, self).__init__()

        self.conv = nn.Conv2d(input_dims,dims,1,1,0)
        self.channel_list = []
        self.codeword_list = torch.nn.ParameterList()
        self.num = dict_num
        self.dim = dims
        self.down = nn.AvgPool2d(2,2)
        for i in range(dict_num):
            c = torch.nn.Parameter(torch.from_numpy(
                np.random.sample((dims,1,1))).float()
            )
            self.codeword_list.append(c)
    def forward(self, input,need = False):
        #input = self.conv(input)
        if need ==False:
            #input = self.down(input)
            return input
        b,c,h,w = input.size()
        t_list = []
        dt_list = []
        dts = 0
        for i in range(self.num):
            c = self.codeword_list[i]
            t = input-c
            dt = (t**2).sum(1)**0.5
            s = dt.view(b,1,h,w)

            t_list.append(t)
            dt_list.append(s)
            dts+=dt
        s_list = []
        for i in range(self.num):
            w = dt_list[i]/dts
            s = (t_list[i]*w).sum(2).sum(2)
            s_list.append(s.view(b,self.dim,1))
        output = torch.cat(s_list,2)
        return output


class MinMaxNet(nn.Container):
    def __init__(self, encode_module,c_num):
        super(MinMaxNet, self).__init__()

        self.model = encode_module
        self.c_num = c_num
    def forward(self, input):
        #input = self.conv(input)
        if self.model is None:
            temp = input
        else:
            temp = self.model(input)
        b,c,h,w = temp.size()
        per_c = c//self.c_num
        index_c = 0
        output = []
        for i in range(self.c_num):
            group_ = temp[:,index_c:index_c+per_c]
            index_c+=per_c
            min_map = group_.min(1)[0].view(b,-1,h,w)
            max_map = group_.max(1)[0].view(b,-1,h,w)
            final_map = min_map+max_map
            output.append(final_map)

        final_output = torch.cat(output,1)
        return final_output


class UpdaterX(nn.Container):
    def __init__(self):
        super(UpdaterX, self).__init__()
        cm = CreateModel()
        self.main = cm.Create(
            'I(8)-CBR(16,3)-' \
            'Down(32)-CBR(32,3)-' \
            'Down(64)-CBR(64,3)-' \
            'Down(128)-CBR(128,3)-' \
            'Res(128,3)-Res(128,3)'
        )

        self.correct = cm.Create(
            'I(128)-CBR(512,3)-' \
            'Res(512,3)-Res(512,3)'
        )

        self.X = cm.Create(
            'I(128)-CBR(512,3)-'
            'Res(512,3)-Res(512,3)-Tanh()'
        )

    def forward(self, input):
        temp = self.main(input)
        d_code = self.correct(temp)
        x  = self.X(temp)
        return d_code,x


def create_down(in_c,out_c,stride=2):
        dilation = stride if stride >2 else 1
        if dilation ==1:
            model = nn.Sequential(
                nn.Conv2d(in_c,out_c,4,2,1),
                nn.BatchNorm2d(out_c),
                nn.PReLU()
            )
        else:
            log_ = int(np.log(dilation)/np.log(2))
            k = log_*2-1
            model = nn.Sequential(
                nn.Conv2d(in_c, in_c, k, 1, k//2,groups=in_c),
                nn.Conv2d(in_c,out_c,k,stride=dilation,padding=(log_-1)*(dilation//log_),dilation=dilation//log_),
                nn.BatchNorm2d(out_c),
                nn.PReLU()
            )
        return model

def create_up(in_c,out_c,stride=2,dilation=1):
        model = nn.Sequential(
            nn.UpsamplingNearest2d(stride,stride),
            nn.Conv2d(in_c,out_c,3,1,(dilation-1)+1),
            nn.BatchNorm2d(out_c),
            nn.PReLU()
        )
        return model


class MultiModel_Encoder(nn.Container):
    def __init__(self,in_c):
        super(MultiModel_Encoder, self).__init__()
        self.A_part = nn.Sequential(
            create_down(in_c, 8),
            create_down( 8, 8),
            create_down( 8, 16),
            create_down(16, 16),
            create_down(16, 32),
            create_down(32, 32),
            create_down(32, 64),
            create_down(64, 64)
        )

        self.B_part = nn.Sequential(
            create_down(in_c,   8, 4),
            create_down(8,  16, 4),
            create_down(16, 32, 4)
        )

        self.C_part = nn.Sequential(
            create_down(in_c,  16, 16)
        )

        self.D_part = nn.Sequential(
            create_down(in_c,  64, 256),
            create_up(64, 8,64)
        )


    def forward(self, input):
        A_code = self.A_part(input)
        B_code = self.B_part(input)
        C_code = self.C_part(input)
        D_code = self.D_part(input)

        return [A_code,B_code,C_code,D_code]


class MultiModel_Decoder(nn.Container):
    def __init__(self,class_num):
        super(MultiModel_Decoder, self).__init__()
        self.A_decode = nn.Sequential(
            create_up(64, 64),
            create_up(64, 32),
        )


        self.AB_decode = nn.Sequential(
            create_up(32, 16),
            create_up(16, 16)
        )

        self.ABC_decode = nn.Sequential(
            create_up(16, 8),
            create_up(8, 8),
        )

        self.ABCD_decode = nn.Sequential(
            create_up(8, 8),
            create_up(8, 8),
            nn.Conv2d(8, class_num, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, input):
        A_code = input[0]
        B_code = input[1]
        C_code = input[2]
        D_code = input[3]

        A = self.A_decode(A_code)
        AB = self.AB_decode(A+B_code)
        ABC = self.ABC_decode(AB + C_code)
        ABCD = self.ABCD_decode(ABC+D_code)

        return ABCD


class MultiModel(nn.Container):
    def __init__(self,class_num):
        super(MultiModel, self).__init__()

        self.A_part = nn.Sequential(
            self.create_down( 3, 8),
            self.create_down( 8, 8),
            self.create_down( 8, 16),
            self.create_down(16, 16),
            self.create_down(16, 32),
            self.create_down(32, 32),
            self.create_down(32, 64),
            self.create_down(64, 64)
        )

        self.B_part = nn.Sequential(
            self.create_down(3,   8, 4),
            self.create_down(8,  16, 4),
            self.create_down(16, 32, 4)
        )

        self.C_part = nn.Sequential(
            self.create_down(3,  16, 16)
        )

        self.D_part = nn.Sequential(
            self.create_down(3,  64, 256),
            self.create_up(64, 8,64)
        )

        self.A_decode = nn.Sequential(
            self.create_up(64, 64),
            self.create_up(64, 32),
        )


        self.AB_decode = nn.Sequential(
            self.create_up(32, 16),
            self.create_up(16, 16)
        )

        self.ABC_decode = nn.Sequential(
            self.create_up(16, 8),
            self.create_up(8, 8),
        )

        self.ABCD_decode = nn.Sequential(
            self.create_up(8, 8),
            self.create_up(8, 8),
            nn.Conv2d(8, class_num, 3, 1, 1),
            nn.Tanh()
        )


    def forward(self, input):
        A_code = self.A_part(input)
        B_code = self.B_part(input)
        C_code = self.C_part(input)
        D_code = self.D_part(input)

        A = self.A_decode(A_code)
        AB = self.AB_decode(A+B_code)
        ABC = self.ABC_decode(AB + C_code)
        ABCD = self.ABCD_decode(ABC+D_code)

        return ABCD


class MultiLayer(nn.Container):
    def __init__(self,w,channels,num,o_c):
        super(MultiLayer, self).__init__()
        self.block_num = num
        self.o_c = o_c
        f_layer,fc = self.create_conv_block(w, channels, num)
        self.A_1 = f_layer
        self.FC = fc
    def create_block(self,in_c,out_c,num):
        model = nn.Sequential(
            nn.Conv2d(in_c*num,out_c*num,4,2,1,groups=num),
            #nn.BatchNorm2d(out_c*num),
            nn.PReLU()
        )
        return model

    def create_conv_block(self,w,channels,block_num):
        w = w//block_num
        it = int(np.log(w)/np.log(2))-3

        area = w*w
        num = block_num**2
        cc = int(2**it * 8)

        model = nn.Sequential(
            nn.Conv2d(channels*num,cc*num,1, 1, 0,groups=num),
            nn.ReLU()
        )
        for i in range(it):
            half_cc= cc//2
            model.add_module(str(i)+'down',
                             self.create_block(cc,half_cc,num)
                             )
            area = area//4
            cc = half_cc

        model_fc = nn.Sequential(
            nn.Linear(cc  * area, 512),
            nn.Linear(512, self.o_c)
        )
        return model,model_fc

    def forward(self, input):
        b,c,h,w = input.size()
        per_w = w//self.block_num
        items = input.split(per_w,3)
        itemstemp = torch.cat(items,1)
        features = torch.cat(itemstemp.split(per_w,2),1)
        x = self.A_1(features)

        x = x.view(b,self.block_num**2,-1)
        return self.FC(x).view(b,self.o_c,self.block_num,self.block_num)

class MultiSample(nn.Container):
    def __init__(self,w,channels,num,o_c):
        super(MultiSample, self).__init__()
        self.block_num = num
        self.o_c = o_c
        f_layer,fc = self.create_conv_block(w, channels, num)
        self.A_1 = f_layer
        self.FC = fc
    def create_block(self,in_c,out_c):
        model = nn.Sequential(
            nn.Conv2d(in_c,out_c,4,2,1),
            nn.BatchNorm2d(out_c),
            nn.PReLU()
        )
        return model

    def create_conv_block(self,w,channels,block_num):
        w = w//block_num
        it = int(np.log(w)/np.log(2))-3

        area = w*w
        num = block_num**2
        cc = int(2**it * 8)

        model = nn.Sequential(
            nn.Conv2d(channels,cc,1, 1, 0),
            nn.BatchNorm2d(cc),
            nn.ReLU()
        )
        for i in range(it):
            half_cc= cc*2
            model.add_module(str(i)+'down',
                             self.create_block(cc,half_cc)
                             )
            area = area//4
            cc = half_cc

        model.add_module('final',
                         nn.Conv2d(cc, cc, 1, 1, 0),
                         )
        model_fc = nn.Sequential(
            nn.Linear(cc, 512),
            nn.Linear(512, self.o_c)
        )
        return model,model_fc

    def forward(self, input):
        b,c,h,w = input.size()
        per_w = w//self.block_num
        output = []
        for i in range(self.block_num):
            row = input[:,:,per_w*i:per_w*i+per_w]
            final_row = []
            for j in range(self.block_num):
                item = row[:,:,:,per_w*j:per_w*j+per_w]
                feature = self.A_1(item)
                bb,cc,hh,ww = feature.size()
                feature = feature.view(bb,cc,-1)
                max_feature = torch.max(feature, 2)[0]
                min_feature = torch.min(feature, 2)[0]
                feature = max_feature+min_feature
                last = self.FC(feature).view(b,self.o_c,1,1)
                final_row.append(last)
            output.append(torch.cat(final_row,3))
        return torch.cat(output,2)


class MultiClassification(nn.Container):
    def __init__(self,w,channels,o_c=1,list_len =5):
        super(MultiClassification, self).__init__()
        self.num = list_len

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.A_list = nn.ModuleList()
        self.Up_list = nn.ModuleList()
        for i in range(self.num):
            block_num = 2*(2**i)
            if i >0:
                ci = i
            else:
                ci = 0
            self.A_list.append(MultiLayer(w, channels+ci*o_c, block_num,o_c))
            self.Up_list.append(nn.UpsamplingNearest2d(w//block_num,w//block_num))

    def forward(self, input,tar = None):
        b,c,h,w = input.size()
        #features_list = []
        maps_list = []

        for i in range(self.num):
            output = self.A_list[i](input)
            if i > 0:
                maps = self.sigmoid(self.Up_list[i](output))
            else:
                maps = self.sigmoid(self.Up_list[i](output))
            maps_list.append(maps)
            if tar is not None and i<5:
                t_map = tar[i]
            else:
                t_map = maps
            input= torch.cat([input,t_map.detach()],1)

            #features_list.append(output)

        maps_cat = torch.cat(maps_list,1)
        return maps_cat


from torch.autograd import Variable

if __name__ =='__main__':
    tensor = torch.FloatTensor(np.random.sample((2,64,128,128)))

    mmn = MultiClassification(128,64,3,4)

    v_t = Variable(tensor).cuda()

    mmn.cuda()

    print (mmn(v_t)[1]).size()


