import torch.nn as nn
import torch



def sb_(i_c,params_list):

    o_c = params_list[0]
    if len(params_list) > 1:
        k_s = params_list[1]
    else:
        k_s = 3
    o_temp = o_c//8
    o_temp = max(4,o_temp)
    models = nn.Sequential(
        ShiftBlock(i_c,o_c)

    )
    return models
class ShiftBlock(nn.Container):
    def __init__(self,i_c,o_c):
        super(ShiftBlock, self).__init__()
        self.pad = nn.ZeroPad2d(1280)
        self.conv1 = nn.Conv2d(i_c,8,3,1,1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(i_c+256*8,o_c,1,1,0)
    def forward(self, input):
        #input = self.conv(input)
        bb,cc,hh,ww = input.size()
        small_input = self.conv1(input)
        padinput = self.pad(small_input)

        cropinput = padinput[:,:,1280-hh//2:1280-hh//2+hh*2,1280-ww//2:1280-ww//2+ww*2]

        y_step = hh//16
        x_step = ww//16
        patch_list = []

        for i in range(16):
            for j in range(16):
                y_ = i*y_step
                x_ = i*x_step
                patch = cropinput[:,:,y_:y_+hh,x_:x_+ww]
                patch_list.append(patch)
        output = torch.cat(patch_list,1)
        output = torch.cat([input,self.relu(output)],1)
        output = self.conv2(output)
        return output
