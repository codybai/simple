import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
class IntConv(nn.Container):
    def __init__(self, in_c, out_c,stride = 1,is_open = 1):
        super(IntConv, self).__init__()
        self.stride = stride
        self.is_open = is_open
        k_s = stride+2
        kernel = np.random.sample((out_c, in_c, k_s, k_s)).astype(np.float32)
        bias = np.zeros((out_c)).astype(np.float32)

        self.IntMax = 2**4
        self.bias = nn.Parameter(
            torch.from_numpy(bias)
        )
        self.kernel = nn.Parameter(
            torch.from_numpy(kernel)
        )
        self.kernel.data.normal_(0.0, 0.02)

    def ToInt(self,value):
        if self.is_open>0:
            v_1024 = self.IntMax * value
            int_v = v_1024.int().float()
            return int_v
        else:
            return self.IntMax *value


    def forward(self, x):
        k = self.ToInt(self.kernel)
        x = self.ToInt(x)
        b = self.ToInt(self.bias)

        #print 'k:', k.min(),k.max()
        #print 'x:', x.min(),x.max()
        #print 'b:', b.min(),b.max()
        output = F.conv2d(x/self.IntMax,
                          k/self.IntMax,
                          b/self.IntMax,
                          (self.stride,self.stride),(1,1),1)
        return output


if __name__ =='__main__':
    tensor = torch.FloatTensor(np.random.sample((3,512,32,32)))
    c = nn.Conv2d(512,3,3,1,1)
    print c.bias

    mmn = IntConv(512,3)
    v_t = Variable(tensor).cuda()
    mmn.cuda()

    optimiz = torch.optim.Adam(mmn.parameters(),
                               lr=1e-3, betas=(0.5, 0.999))

    for i in range(1000):
        out =  (mmn(v_t)).mean()
        print out.data[0]
        out.backward()
        optimiz.step()
