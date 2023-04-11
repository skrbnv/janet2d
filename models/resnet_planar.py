import torch.nn as nn
from torchvision.models import resnet18
from libs.semiatt import PlanarConv2d

def rn18_modified():
    model = resnet18()
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
    model.fc = nn.Linear(512, 10)
        #
    model.layer1[0].conv1 = PlanarConv2d(64, 64, 3, 1, 1)
    model.layer1[0].conv2 = PlanarConv2d(64, 64, 3, 1, 1)
    model.layer1[1].conv1 = PlanarConv2d(64, 64, 3, 1, 1)
    model.layer1[1].conv2 = PlanarConv2d(64, 64, 3, 1, 1)
    #
    model.layer2[0].conv1 = PlanarConv2d(64, 128, 3, 2, 1)
    model.layer2[0].conv2 = PlanarConv2d(128, 128, 3, 1, 1)
    model.layer2[0].downsample[0] = PlanarConv2d(64, 128, 1, 2, 0)
    model.layer2[1].conv1 = PlanarConv2d(128, 128, 3, 1, 1)
    model.layer2[1].conv2 = PlanarConv2d(128, 128, 3, 1, 1)
    #
    model.layer3[0].conv1 = PlanarConv2d(128, 256, 3, 2, 1)
    model.layer3[0].conv2 = PlanarConv2d(256, 256, 3, 1, 1)
    model.layer3[0].downsample[0] = PlanarConv2d(128, 256, 1, 2, 0)
    model.layer3[1].conv1 = PlanarConv2d(256, 256, 3, 1, 1)
    model.layer3[1].conv2 = PlanarConv2d(256, 256, 3, 1, 1)
    #
    model.layer4[0].conv1 = PlanarConv2d(256, 512, 3, 2, 1)
    model.layer4[0].conv2 = PlanarConv2d(512, 512, 3, 1, 1)
    model.layer4[0].downsample[0] = PlanarConv2d(256, 512, 1, 2, 0)
    model.layer4[1].conv1 = PlanarConv2d(512, 512, 3, 1, 1)
    model.layer4[1].conv2 = PlanarConv2d(512, 512, 3, 1, 1)
    #
    return model

def rn18():
    model = resnet18()
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
    model.fc = nn.Linear(512, 10)
    return model