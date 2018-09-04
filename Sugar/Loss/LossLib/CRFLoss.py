import torch
import torch.nn as nn
import numpy as np
class CRFLoss(nn.Module):
    def __init__(self,k_size = 7):
        super(CRFLoss, self).__init__()
        self.k_size = k_size
        self.pad = nn.ZeroPad2d(k_size//2)
        half_size = self.k_size // 2
        x_map,y_map = np.meshgrid(range(k_size),range(k_size))
        self.dist = (x_map-half_size)**2+(y_map-half_size)**2
        self.dist_w = np.exp(-self.dist/6.)

    def forward(self, x,src):
        bb,cc,hh,ww = x.size()
        pad_x = self.pad(x)
        pad_src = self.pad(src)
        sum = 0
        for i in range(self.k_size):
            for j in range(self.k_size):
                local_x = pad_x[:,:,i:i+hh,j:j+ww]
                local_src = pad_src[:, :, i:i + hh, j:j + ww]
                color_diff = (((local_src-src)**2+1e-8).sum(1)).view(-1,1,hh,ww)
                x_label = torch.clamp((x-0.5)*100+0.5,0,1)
                local_label= torch.clamp((local_x-0.5)*100+0.5,0,1)
                x_diff = torch.clamp((torch.abs(local_x-x))-0.01,0,1)
                miu = torch.abs(x_label-local_label)

                weight = torch.exp(-(color_diff+self.dist[i,j])/4)+self.dist_w[i,j]
                sum+=(x_diff*miu+(x_diff*(1-miu)).detach())*weight
        # return (d_r.abs()+d_b.abs()).mean()
        return sum.mean()

if __name__ == '__main__':
    import numpy as np
    from torch.autograd import  Variable
    tensor_x = torch.FloatTensor(np.random.sample((2, 1,128,128)))
    tensor_src = torch.FloatTensor(np.random.sample((2, 3, 128, 128)))
    mmn = CRFLoss()

    tensor_x = Variable(tensor_x).cuda()
    tensor_src = Variable(tensor_src).cuda()
    mmn.cuda()
    print mmn(tensor_x,tensor_src).size()