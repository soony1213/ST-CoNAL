'''ResNet & VGG in PyTorch.

'''
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.distributions import Normal

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            out += self.downsample(x)
        else:
            out += self.shortcut(x)
        out = F.relu(out)
        return out
class BasicBlock_img(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock_img, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            out += self.downsample(x)
        else:
            out += x
        out = F.relu(out)
        return out

class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.dropout(F.relu(self.bn1(self.conv1(x))), p=0.3, training=True)
        out = F.dropout(self.bn2(self.conv2(out)), p=0.3, training=True)
        out +=  self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        # self.linear2 = nn.Linear(1000, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, 4)
        outf = out.view(out.size(0), -1)
        # outl = self.linear(outf)
        out = self.linear(outf)
        return out, outf, [out1, out2, out3, out4]


class ResNet_img(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_img, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.avgpool = nn.AvgPool2d(10)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes

        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out4 = self.avgpool(out4)
        outf = out4.view(out4.size(0), -1)
        out = self.dropout(outf)
        out = self.linear(out)

        return out, outf, [out1, out2, out3, out4]

class ResNetfm(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetfm, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        # self.linear2 = nn.Linear(1000, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, 4)
        outf = out.view(out.size(0), -1)
        # outl = self.linear(outf)
        out = self.linear(outf)
        return out, outf, [out1, out2, out3, out4]



def ResNet18(num_classes = 10, pretrained=False, no_grad=False):
    if no_grad:
        model = ResNet(BasicBlock, [2,2,2,2], num_classes)
        for param in model.parameters():
            param.detach_()
    else:
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        print("Warning: We allow training on layer 4 and linear.")
        should_train = lambda param_name: param_name.startswith('layer4') or param_name.startswith('linear')
        for name, param in model.named_parameters():
            if not should_train(name):
                param.requires_grad_(False)
            else:
                print("Allow gradient on:", name)

    return model

def ResNet18_img(num_classes = 10, pretrained=False, no_grad=False):
    if no_grad:
        model = ResNet_img(BasicBlock_img, [2,2,2,2], num_classes)
        for param in model.parameters():
            param.detach_()
    else:
        model = ResNet_img(BasicBlock_img, [2, 2, 2, 2], num_classes)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
        print("Warning: We allow training on layer 4 and linear.")
        should_train = lambda param_name: param_name.startswith('layer4') or param_name.startswith('linear')
        for name, param in model.named_parameters():
            if not should_train(name):
                param.requires_grad_(False)
            else:
                print("Allow gradient on:", name)
    return model

def ResNet18fm(num_classes = 10, no_grad=False):
    if no_grad:
        model = ResNetfm(BasicBlock, [2,2,2,2], num_classes)
        for param in model.parameters():
            param.detach_()
        return model
    else:
        return ResNetfm(BasicBlock, [2,2,2,2], num_classes)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50(num_classes = 10, pretrained=False, no_grad=False):
    if no_grad:
        model = ResNet(BasicBlock, [3,4,6,3], num_classes)
        for param in model.parameters():
            param.detach_()
    else:
        model = ResNet(BasicBlock, [3,4,6,3], num_classes)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        print("Warning: We allow training on layer 4 and linear.")
        should_train = lambda param_name: param_name.startswith('layer4') or param_name.startswith('linear')
        for name, param in model.named_parameters():
            if not should_train(name):
                param.requires_grad_(False)
            else:
                print("Allow gradient on:", name)

    return model

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])