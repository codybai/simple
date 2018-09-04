import torch.nn as nn
import torch
from LayerFunction import ResBlock
import numpy as np
from Model.CreateModel import CreateModel

class EdgeLayer(nn.Container):
    def __init__(self):
        super(EdgeLayer, self).__init__()
        self.model =  nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh(),
            nn.ReLU(),
        )
    def forward(self, input):
        mask = self.model(input)
        mask_3 = torch.cat([mask]*3,1)
        return mask_3,input*mask_3





from torch.autograd import Variable

if __name__ =='__main__':
    tensor = torch.FloatTensor(np.random.sample((2,64,128,128)))

    mmn = MultiClassification(128,64)

    v_t = Variable(tensor).cuda()

    mmn.cuda()

    print (mmn(v_t)[1]).size()


