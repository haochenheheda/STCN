"""
modules.py - This file stores the rathering boring network blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from model import mod_resnet
from model import cbam
import timm

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class FeatureFusionBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()

        self.block1 = ResBlock(indim, outdim)
        self.attention = cbam.CBAM(outdim)
        self.block2 = ResBlock(outdim, outdim)

    def forward(self, x, f16):
        x = torch.cat([x, f16], 1)
        x = self.block1(x)
        r = self.attention(x)
        x = self.block2(x + r)

        return x


# Single object version, used only in static image pretraining
# This will be loaded and modified into the multiple objects version later (in stage 1/2/3)
# See model.py (load_network) for the modification procedure
class ValueEncoderSO(nn.Module):
    def __init__(self, value_encoder_type='resnet18'):
        super().__init__()
        
        if value_encoder_type == 'resnet18':
            resnet = mod_resnet.resnet18(pretrained=True, extra_chan=1)
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu  # 1/2, 64
            self.maxpool = resnet.maxpool

            self.layer1 = resnet.layer1 # 1/4, 64
            self.layer2 = resnet.layer2 # 1/8, 128
            self.layer3 = resnet.layer3 # 1/16, 256

            self.fuser = FeatureFusionBlock(1024 + 256, 512)

        elif value_encoder_type == 'resnet50':
            resnet = mod_resnet.resnet50(pretrained=True, extra_chan=1)
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu  # 1/2, 64
            self.maxpool = resnet.maxpool

            self.layer1 = resnet.layer1 # 1/4, 64
            self.layer2 = resnet.layer2 # 1/8, 128
            self.layer3 = resnet.layer3 # 1/16, 256

            self.fuser = FeatureFusionBlock(1024 + 1024, 512)

        #########################################################################
        elif value_encoder_type == 'resnest101':
            m = timm.create_model('resnest101e', features_only=True, pretrained=True)
            self.conv1 = m.conv1
            self.extra_conv = nn.Conv2d(4, 64, kernel_size=(3,3), stride=(2,2),padding=(1,1),bias=False)
            self.extra_conv.weight.data[:,:3,:,:] = self.conv1[0].weight.data
            nn.init.orthogonal_(self.extra_conv.weight[:,3:4,:,:])
            self.conv1[0] = self.extra_conv
            self.bn1 = m.bn1
            self.relu = m.act1  # 1/2, 64
            self.maxpool = m.maxpool

            self.layer1 = m.layer1 # 1/4, 64
            self.layer2 = m.layer2 # 1/8, 128
            self.layer3 = m.layer3 # 1/16, 256

            self.fuser = FeatureFusionBlock(1024 + 1024, 512)
        ########################################################################

    def forward(self, image, key_f16, mask):
        # key_f16 is the feature from the key encoder

        f = torch.cat([image, mask], 1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)   # 1/4, 64
        x = self.layer2(x) # 1/8, 128
        x = self.layer3(x) # 1/16, 256

        x = self.fuser(x, key_f16)

        return x


# Multiple objects version, used in other times
class ValueEncoder(nn.Module):
    def __init__(self, value_encoder_type='resnet18'):
        super().__init__()
    
        if value_encoder_type == 'resnet18':
            resnet = mod_resnet.resnet18(pretrained=True, extra_chan=2)
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu  # 1/2, 64
            self.maxpool = resnet.maxpool

            self.layer1 = resnet.layer1 # 1/4, 64
            self.layer2 = resnet.layer2 # 1/8, 128
            self.layer3 = resnet.layer3 # 1/16, 256

            self.fuser = FeatureFusionBlock(1024 + 256, 512)

        elif value_encoder_type == 'resnet50':
            resnet = mod_resnet.resnet50(pretrained=True, extra_chan=2)
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu  # 1/2, 64
            self.maxpool = resnet.maxpool

            self.layer1 = resnet.layer1 # 1/4, 64
            self.layer2 = resnet.layer2 # 1/8, 128
            self.layer3 = resnet.layer3 # 1/16, 256

            self.fuser = FeatureFusionBlock(1024 + 1024, 512)

        #########################################################################
        elif value_encoder_type == 'resnest101':
            m = timm.create_model('resnest101e', features_only=True, pretrained=True)
            self.conv1 = m.conv1
            self.extra_conv = nn.Conv2d(5, 64, kernel_size=(3,3), stride=(2,2),padding=(1,1),bias=False)
            self.extra_conv.weight.data[:,:3,:,:] = self.conv1[0].weight.data
            nn.init.orthogonal_(self.extra_conv.weight[:,3:5,:,:])
            self.conv1[0] = self.extra_conv
            self.bn1 = m.bn1
            self.relu = m.act1  # 1/2, 64
            self.maxpool = m.maxpool

            self.layer1 = m.layer1 # 1/4, 64
            self.layer2 = m.layer2 # 1/8, 128
            self.layer3 = m.layer3 # 1/16, 256

            self.fuser = FeatureFusionBlock(1024 + 1024, 512)
        ########################################################################

    def forward(self, image, key_f16, mask, other_masks):
        # key_f16 is the feature from the key encoder

        f = torch.cat([image, mask, other_masks], 1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)   # 1/4, 64
        x = self.layer2(x) # 1/8, 128
        x = self.layer3(x) # 1/16, 256

        x = self.fuser(x, key_f16)

        return x
 

class KeyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        m = timm.create_model('resnest101e', features_only=True, pretrained=True)
        self.stage0 = nn.Sequential(m.conv1,m.bn1,m.act1)
        self.stage1 = nn.Sequential(m.maxpool,m.layer1)
        self.stage2 = m.layer2
        self.stage3 = m.layer3


    def forward(self, f):

        f4 = self.stage1(self.stage0(f))
        f8 = self.stage2(f4)
        f16 = self.stage3(f8)
        

        return f16, f8, f4


class UpsampleBlock(nn.Module):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv(skip_f)
        x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x


class KeyProjection(nn.Module):
    def __init__(self, indim, keydim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)
    
    def forward(self, x):
        return self.key_proj(x)
