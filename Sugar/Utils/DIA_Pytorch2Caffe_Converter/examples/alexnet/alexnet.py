import os
import torch
import argparse
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return nn.functional.log_softmax(x)


    def alexnet(pretrained=False, **kwargs):
        r"""AlexNet model architecture from the
        `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

        Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
        model = AlexNet(**kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        return model


def get_model_and_input():
    pth_name = "alexnet.pth"
    pth_file = os.path.split(os.path.abspath(__file__))[0] + '/' + pth_name
    print("pth file :", pth_file)
    model = AlexNet()

    if os.path.isfile(pth_file):
        model.load_state_dict(torch.load(pth_file,map_location=lambda storage,loc: storage))
    else:
        print "Warning: load pth_file failed !!!"

    batch_size = 1
    channels = 3
    height = 224
    width = 224
    images = Variable(torch.rand(batch_size,channels,height,width))
    return model, images



