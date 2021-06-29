from torchvision import models
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

resnet = models.resnet18(pretrained=True)

for param in resnet.parameters():
    param.requires_grad = False

num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 2)

total_trainable_params = sum(
    p.numel() for p in resnet.parameters() if p.requires_grad)

print(total_trainable_params)




