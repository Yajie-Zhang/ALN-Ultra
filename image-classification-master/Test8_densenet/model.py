import re
from typing import Any, List, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
import torchvision.models as models



class DenseNet(nn.Module):

    def __init__(self,
                 num_classes: int = 1000):
        super(DenseNet, self).__init__()
        self.densenet = models.densenet121(pretrained=False)
        # 替换最后的全连接层，以适应新的类别数
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)


    def forward(self, x):
        out=self.densenet(x)
        return out


def densenet121(num_classes):
    # Top-1 error: 25.35%
    # 'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth'
    return DenseNet(num_classes)
