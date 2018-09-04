import torch.nn as nn
import torch
import os
import numpy as np
class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.CE = nn.BCELoss()
        self.true = None
        self.false = None

    def forward(self, x,tar,netD):
        netD.zero_grad()
        f_out = netD(x.detach())
        r_out = netD(tar)

        if self.true is None or self.true.numel()!= f_out.numel():
            type_tensor = type(f_out)
            data_type = type(f_out.data)
            fake_tensor = data_type(f_out.size()).fill_(0)
            self.false = type_tensor(fake_tensor, requires_grad=False)
            real_tensor = data_type(f_out.size()).fill_(1)
            self.true = type_tensor(real_tensor, requires_grad=False)
        fake_ = self.CE(f_out, self.false)
        real_ = self.CE(r_out, self.true)

        (fake_+real_).backward()
        netD.isTest = False
        netD.step()
        netD.zero_grad()
        fake_ = self.CE(netD(x), self.true)
        netD.isTest = True
        return fake_

