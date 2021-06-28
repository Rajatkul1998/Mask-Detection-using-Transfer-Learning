from torchvision import models
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
vgg16model = models.vgg16_bn(pretrained=True)
for param in vgg16model.parameters():
    param.requires_grad = False
vgg16model.classifier[6] = nn.Sequential(
    nn.Linear(4096, 2) 
)
for param in vgg16model.classifier[6].parameters():
    param.requires_grad = True




