import torch.nn as nn
import torchvision.models as tm
import torch

from models.resnet18_se import resnet18_se
import numpy as np
import random

class Flatten(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = torch.flatten(x, self.dim)
        return x



class SURF_Multi(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)

        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

    def forward(self, img_rgb, img_depth):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_depth = self.special_bone_depth(img_depth)

        x = torch.cat((x_rgb, x_depth), dim=1)
        layer3 = self.shared_bone[0](x)
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)
        return x
