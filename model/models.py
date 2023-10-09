import torch
import torch.nn as nn

from .module import norm, conv1x1, conv3x3, Flatten
from .resnet import ResBlock
from .odenet import ODEBlock, ODEfunc


def get_downsample_layer(downsampling_method):
    if downsampling_method == "res":
        layer = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        )
        return layer
    if downsampling_method == "conv":
        layer = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        )
        return layer


def get_feature_layer(is_odenet):
    if is_odenet:
        layer = ODEBlock(ODEfunc(64))
        return layer
    else:
        layer = nn.Sequential(*[ResBlock(64, 64) for _ in range(6)])
        return layer


class Model(nn.Module):
    def __init__(self, is_odenet=True, downsampling_method="conv") -> None:
        super(Model, self).__init__()
        self.downsampling_layer = get_downsample_layer(downsampling_method)
        self.feature_layer = get_feature_layer(is_odenet)
        self.fc_layers = nn.Sequential(
            norm(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.downsampling_layer(x)
        x = self.feature_layer(x)
        x = self.fc_layers(x)
        return x
