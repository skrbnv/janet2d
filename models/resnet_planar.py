import torch.nn as nn
import torch
from torchvision.models import resnet18


class WeightedAlignment(nn.Module):
    def __init__(self, planes, midshape) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.empty(planes, planes))
        nn.init.kaiming_normal_(self.weights)
        self.bn = nn.BatchNorm2d(planes)
        self.scale = torch.tensor(midshape[-1] * midshape[-2])**.5 * planes

    def forward(self, x):
        identity = x.clone()
        # x shape is B,C,H,W
        x = x.flatten(-2)
        # x shape is B,C,H*W
        x = x @ x.transpose(-1, -2)
        # x shape is B,C,C
        x *= self.weights
        # x shape is B,C,C
        x = torch.sum(x, dim=-1, keepdim=True)
        # x shape is B,C,1
        x = x.unsqueeze(-1)
        # x shape is B,C,1,1
        x = x / self.scale  # norm
        x = identity + x  # auto broadcast to identity
        x = self.bn(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel=3,
                 stride=1,
                 padding=1,
                 midshape=(4, 4)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel,
                      stride=stride,
                      padding=padding), nn.BatchNorm2d(out_channels),
            nn.GELU(approximate='tanh'))
        self.alignment = WeightedAlignment(out_channels, midshape)

    def forward(self, x):
        x = self.conv(x)
        x = self.alignment(x)
        return x


def rn18():
    model = resnet18()
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
    model.fc = nn.Linear(512, 10)
    #
    model.layer1[0].conv1 = ConvBlock(64, 64, 3, 1, 1, False, (32, 32))
    model.layer1[0].conv2 = ConvBlock(64, 64, 3, 1, 1, False, (32, 32))
    model.layer1[1].conv1 = ConvBlock(64, 64, 3, 1, 1, False, (32, 32))
    model.layer1[1].conv2 = ConvBlock(64, 64, 3, 1, 1, False, (32, 32))
    #
    model.layer2[0].conv1 = ConvBlock(64, 128, 3, 2, 1, False, (16, 16))
    model.layer2[0].conv2 = ConvBlock(128, 128, 3, 1, 1, False, (16, 16))
    model.layer2[0].downsample[0] = ConvBlock(64, 128, 1, 2, 0, False, (8, 8))
    model.layer2[1].conv1 = ConvBlock(128, 128, 3, 1, 1, False, (16, 16))
    model.layer2[1].conv2 = ConvBlock(128, 128, 3, 1, 1, False, (16, 16))
    #
    model.layer3[0].conv1 = ConvBlock(128, 256, 3, 2, 1, False, (8, 8))
    model.layer3[0].conv2 = ConvBlock(256, 256, 3, 1, 1, False, (8, 8))
    model.layer3[0].downsample[0] = ConvBlock(128, 256, 1, 2, 0, False, (4, 4))
    model.layer3[1].conv1 = ConvBlock(256, 256, 3, 1, 1, False, (8, 8))
    model.layer3[1].conv2 = ConvBlock(256, 256, 3, 1, 1, False, (8, 8))
    #
    model.layer4[0].conv1 = ConvBlock(256, 512, 3, 2, 1, False, (4, 4))
    model.layer4[0].conv2 = ConvBlock(512, 512, 3, 1, 1, False, (4, 4))
    model.layer4[0].downsample[0] = ConvBlock(256, 512, 1, 2, 0, False, (2, 2))
    model.layer4[1].conv1 = ConvBlock(512, 512, 3, 1, 1, False, (4, 4))
    model.layer4[1].conv2 = ConvBlock(512, 512, 3, 1, 1, False, (4, 4))
    #
    return model