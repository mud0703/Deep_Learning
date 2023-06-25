import torch
import torch.nn as nn
from torchvision.models import EfficientNet, resnet18



def create_net(netname, num_channels, num_classes):
    if netname == 'efficientnet':
        net = EfficientNet()

    if netname == 'resnet18':
        net = resnet18(pretrained = True)
        net.conv1 = nn.conv2d(num_channels, net.conv1.out_channels, kernel_size = net.conv1.kernel_size, 
                              stride = net.conv1.stride, padding = net.conv1.padding, bias = net.conv1.bias)
        net.fc = nn.Sequential(
            nn.Linear(net.fc.in_features, 256),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )

    return net