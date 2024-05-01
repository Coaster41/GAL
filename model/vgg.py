import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from collections import OrderedDict

norm_mean, norm_var = 0.0, 1.0

defaultcfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 512]
defaultcfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]

class ChannelSelection(nn.Module):
    def __init__(self, indexes, fc=False):
        super(ChannelSelection, self).__init__()
        self.indexes = indexes
        self.fc = fc

    def forward(self, input_tensor):
        if self.fc:
            return input_tensor[:, self.indexes]

        if len(self.indexes) == input_tensor.size()[1]:
            return input_tensor

        return input_tensor[:, self.indexes, :, :]

class Mask(nn.Module):
    def __init__(self, init_value=[1], fc=False):
        super().__init__()
        self.weight = Parameter(torch.Tensor(init_value))
        self.fc = fc

    def forward(self, input):
        if self.fc:
            weight = self.weight
        else:
            weight = self.weight[None, :, None, None]
        return input * weight

class VGG(nn.Module):
    def __init__(self, num_classes=10, init_weights=True, is_sparse=False, affine=True, cfg=None, index=None):
        super(VGG, self).__init__()
        self.feature = nn.Sequential()
        self._AFFINE = affine

        if cfg is None:
            cfg = defaultcfg
        print(is_sparse)
        self._AFFINE = affine
        self.feature = self.make_layers(cfg, True)
        num_classes = 10
        self.classifier = nn.Linear(cfg[-1], num_classes)

        if is_sparse:
            self.feature = self.make_sparse_layers(cfg, True)
            self.classifier = nn.Linear(cfg[-1], num_classes)
        else:
            self.feature = self.make_layers(cfg, True)
            self.classifier = nn.Linear(cfg[-1], num_classes)
        
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, affine=self._AFFINE), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    def make_sparse_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, affine=self._AFFINE), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                m = Normal(torch.tensor([norm_mean]*int(v)), torch.tensor([norm_var]*int(v))).sample()
                init_value = m
                layers += [Mask(init_value)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def vgg_19_bn(**kwargs):
    model = VGG(**kwargs)
    return model

def vgg_19_bn_sparse(**kwargs):
    model = VGG(is_sparse=True, **kwargs)
    return model


    