import torch
import torch.nn as nn
from torchvision.models import resnet

class Model(nn.Module):
    def __init__(self, num_classes=2,dim=2048):
        super(Model, self).__init__()
        self.num_classes=num_classes
        ResNet=resnet.resnet50(pretrained=True)
        # ResNet=densenet121(pretrained=True)
        self.conv1=ResNet.conv1
        self.bn1=ResNet.bn1
        self.relu=ResNet.relu
        self.maxpool=ResNet.maxpool
        self.layer1=ResNet.layer1
        self.layer2=ResNet.layer2
        self.layer3=ResNet.layer3
        self.layer4=ResNet.layer4
        self.avgpool=ResNet.avgpool
        self.dim=dim
        # self.cls = nn.Linear(self.dim,num_classes)
        self.cls = nn.Linear(2048*7*4, num_classes)

    def forward(self, x,pro_att_ori=None,label_oh=None,N=3):
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x);
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = self.layer3(x)
        x = self.layer4(x_3)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x=x.view(x.size(0),-1)
        y = self.cls(x)
        return y


if __name__ == '__main__':
    input_tensor=torch.rand(2,3,128,200)
    model=Model()
    out=model(input_tensor)