from typing import Tuple

import numpy as np
import torch
import torch.jit as jit
from torchvision import transforms
from torch import nn


def out_sz(in_size, k=3, pad=0, stride=1):
    return ((in_size - k + 2 * pad) / stride + 1).astype(np.int)


class LocalMapCNN(nn.Module):
    def __init__(self, in_shape: Tuple[int, int, int], out_channels: int, stride: int = 1, flatten: bool = True):
        super(LocalMapCNN, self).__init__()

        in_channels = in_shape[2]
        modules = [nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride),
                   nn.MaxPool2d(kernel_size=2),
                   nn.ReLU(),
                   nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=stride),
                   nn.MaxPool2d(kernel_size=2),
                   nn.ReLU(),
                   nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=stride),
                   nn.ReLU()]

        map_output_size = out_sz(out_sz(out_sz(np.array(in_shape[:2])) // 2) // 2)
        if flatten:
            modules.append(nn.Flatten())
            self.output_size = out_channels * np.prod(map_output_size)
        else:
            self.output_size = [out_channels] + map_output_size.tolist()
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


# NOTE: making this a jit.ScriptModule instead of an nn.Module makes it faster, but copying it will make the components
# non-leaf nodes, leading to an error from the optimizer when training
class LocalDoubleMapCNN(nn.Module):
    def __init__(self, in_shape: Tuple[int, int, int], out_channels: int, stride: int = 1, flatten: bool = True):
        super(LocalDoubleMapCNN, self).__init__()

        assert tuple(in_shape[:2]) in [tuple([120, 120]), tuple([150, 150])], "Not implemented for other local map resolutions yet"
        assert flatten, flatten

        in_channels = in_shape[2]
        activation = nn.ReLU
        reduced_size = 30
        low_res_modules = [transforms.Resize([reduced_size, reduced_size]),  # resize to a resolution of 0.1
                           nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1),
                           nn.MaxPool2d(kernel_size=2),
                           activation(),
                           nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=stride),
                           activation(),
                           nn.Conv2d(out_channels, out_channels // 2, kernel_size=(3, 3), stride=stride),
                           activation()]
        low_res_output_size = out_sz(out_sz(out_sz(np.array([reduced_size, reduced_size]), stride=2)))

        high_res_modules = [transforms.CenterCrop(reduced_size),  # take the center at the original resolution of 0.025
                            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1),
                            nn.MaxPool2d(kernel_size=2),
                            activation(),
                            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=stride),
                            activation(),
                            nn.Conv2d(out_channels, out_channels // 2, kernel_size=(3, 3), stride=stride),
                            activation()]
        high_res_output_size = out_sz(out_sz(out_sz(np.array([reduced_size, reduced_size]), stride=2)))

        if flatten:
            low_res_modules.append(nn.Flatten())
            high_res_modules.append(nn.Flatten())
            self.output_size = out_channels // 2 * np.prod(low_res_output_size) + out_channels // 2 * np.prod(high_res_output_size)
        else:
            assert low_res_output_size == high_res_output_size, (low_res_output_size, high_res_output_size)
            self.output_size = [2 * (out_channels // 2)] + low_res_output_size.tolist()

        self.low_res_model = nn.Sequential(*low_res_modules)
        self.high_res_model = nn.Sequential(*high_res_modules)

    def forward(self, x):
        low_res_features = self.low_res_model(x)
        high_res_features = self.high_res_model(x)
        return torch.cat([low_res_features, high_res_features], dim=1)
