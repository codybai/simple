import torch.nn as nn
import torch
import os

class BinarizationLoss(nn.Module):
    def __init__(self,half = 0.5):
        super(BinarizationLoss, self).__init__()
        self.half = half

    def forward(self, input):
        dist = (input-self.half)**2

        return 1/(0.1+dist.mean())

