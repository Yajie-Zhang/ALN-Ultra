import math
import copy
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from efficientnet_pytorch import EfficientNet


class efficientNet(nn.Module):
    def __init__(self,
                 num_classes: int = 1000,
                 ):
        super(efficientNet, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        # 替换最后的全连接层，以适应新的类别数
        self.efficientnet._fc = nn.Linear(self.efficientnet._fc.in_features, num_classes)


    def forward(self, x):
        out=self.efficientnet(x)
        return out


def efficientnet_b0(num_classes=1000):
    # input image size 224x224
    return efficientNet(num_classes=num_classes)



