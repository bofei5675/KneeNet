from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self,pretrained,dropout,use_cuda):
        super(ResNet,self).__init__()

        self.net = resnet18(pretrained=pretrained)

        self.net.avgpool = nn.AvgPool2d(28)

        self.net.fc = nn.Sequential(nn.Dropout(dropout),nn.Linear(512,8))
        if use_cuda:
            self.net.cuda()

    def forward(self, inp):

        return self.net(inp)


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss,self).__init__()


    def forward(self, inp, target):
        diff = inp - target
        diff_sq = diff ** 2
        return diff_sq.mean()


