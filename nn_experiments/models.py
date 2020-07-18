import functools
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

import layers as rpn


class MNISTFCNet(torch.nn.Module):
    kCompatibleDataset = 'mnist'
    def __init__(self, hidden_size, rand_relu=False, rp_args={}):
        super(MNISTFCNet, self).__init__()
        self.rand_relu = rand_relu
        kept_keys = ['keep_frac', 'full_random', 'sparse']
        kept_dict = {key: rp_args[key] for key in kept_keys}

        self.fc1 = rpn.RandLinear(784, hidden_size, **kept_dict)
        self.relu1 = rpn.RandReLULayer(**kept_dict)
        self.fc2 = rpn.RandLinear(hidden_size, hidden_size, **kept_dict)
        self.relu2 = rpn.RandReLULayer(**kept_dict)
        self.fc3 = rpn.RandLinear(hidden_size, hidden_size, **kept_dict)
        self.relu3 = rpn.RandReLULayer(**kept_dict)
        self.fc4 = rpn.RandLinear(hidden_size, 10, **kept_dict)

    def forward(self, x, retain=False, skip_rand=False):
        if self.rand_relu:
            skip_relu = False
        else:
            skip_relu = True

        x = nn.Flatten()(x)
        x = self.fc1(x, retain=retain, skip_rand=skip_rand)
        x = self.relu1(x, skip_rand=skip_relu)
        x = self.fc2(x, retain=retain, skip_rand=skip_rand)
        x = self.relu2(x, skip_rand=skip_relu)
        x = self.fc3(x, retain=retain, skip_rand=skip_rand)
        x = self.relu3(x, skip_rand=skip_relu)
        x = self.fc4(x, retain=retain, skip_rand=skip_rand)
        output = F.log_softmax(x, dim=1)
        return output


class CIFARConvNet(torch.nn.Module):
    kCompatibleDataset = 'cifar10'

    def __init__(self, rand_relu=False, rp_args={}):
        super(CIFARConvNet, self).__init__()
        self.rand_relu = rand_relu
        kept_keys = ['keep_frac', 'full_random', 'sparse']
        kept_dict = {key: rp_args[key] for key in kept_keys}

        self.conv1 = rpn.RandConv2dLayer(3, 16, 5, padding=2, **kept_dict)
        self.relu1 = rpn.RandReLULayer(**kept_dict)
        self.conv2 = rpn.RandConv2dLayer(16, 32, 5, padding=2, **kept_dict)
        self.relu2 = rpn.RandReLULayer(**kept_dict)
        self.conv3 = rpn.RandConv2dLayer(32, 32, 5, padding=2, **kept_dict)
        self.relu3 = rpn.RandReLULayer(**kept_dict)
        self.conv4 = rpn.RandConv2dLayer(32, 32, 5, padding=2, **kept_dict)
        self.relu4 = rpn.RandReLULayer(**kept_dict)

        self.fc5 = rpn.RandLinear(2048, 10, **kept_dict)

    def forward(self, x, retain=False, skip_rand=False):
        if self.rand_relu:
            skip_relu = False
        else:
            skip_relu = True

        x = self.conv1(x, retain=retain, skip_rand=skip_rand)
        x = self.relu1(x, skip_rand=skip_relu)

        x = self.conv2(x, retain=retain, skip_rand=skip_rand)
        x = self.relu2(x, skip_rand=skip_relu)

        x = F.avg_pool2d(x, 2)

        x = self.conv3(x, retain=retain, skip_rand=skip_rand)
        x = self.relu3(x, skip_rand=skip_relu)

        x = self.conv4(x, retain=retain, skip_rand=skip_rand)
        x = self.relu4(x, skip_rand=skip_relu)

        x = F.avg_pool2d(x, 2)

        x = torch.flatten(x, start_dim=1)
        x = self.fc5(x, retain=retain, skip_rand=skip_rand)
        output = F.log_softmax(x, dim=1)
        return output
