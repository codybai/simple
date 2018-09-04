import torch
import torch.nn as nn

class tv_(nn.Module):
    def __init__(self):
        super(tv_, self).__init__()

    def forward(self, x):
        x_right  = x[:, :, 1:, 0:-1]
        x_bottom = x[:, :, 0:-1, 1:]
        x = x[:, :, 0:-1, 0:-1]
        d_r = x_right-x
        d_b = x_bottom-x
        # return (d_r.abs()+d_b.abs()).mean()
        return (d_r**2 + d_b**2).mean()

class TV(nn.Module):
    def __init__(self, ngpu):
        self.ngpu = ngpu
        super(TV, self).__init__()
        self.main=tv_()

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor):
            if isinstance(self.ngpu, int) and self.ngpu > 1:
                gpu_ids = range(self.ngpu)
            elif isinstance(self.ngpu, list):
                gpu_ids = self.ngpu
        return nn.parallel.data_parallel(self.main, input, gpu_ids)