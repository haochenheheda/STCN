"""
network.py - The core of the neural network
Defines the structure and memory operations
Modifed from STM: https://github.com/seoungwugoh/STM

The trailing number of a variable usually denote the stride
e.g. f16 -> encoded features with stride 16
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.general_modules import *


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)

    def forward(self, x):
        x = self.atrous_conv(x)
        return F.relu(x,inplace=True)


class ASPP(nn.Module):
    def __init__(self):
        super(ASPP, self).__init__()
        dilations = [1, 2, 4, 8]

        self.aspp1 = _ASPPModule(1024, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(1024, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(1024, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(1024, 256, 3, padding=dilations[3], dilation=dilations[3])
        self.conv1 = nn.Conv2d(1024, 256, 1, bias=False)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv1(x)
        return F.dropout(F.relu(x),p = 0.5,training=self.training)


class Decoder(nn.Module):
    def __init__(self, key_encoder_type, aspp):
        super().__init__()
        self.aspp = aspp

        if key_encoder_type == 'resnet50' or key_encoder_type == 'wide_resnet50' or key_encoder_type == 'resnest101' or key_encoder_type == 'resnet50_v2':
            up_16_8_indim = 512
            up_8_4_indim = 256
        elif key_encoder_type == 'convext':
            up_16_8_indim = 256
            up_8_4_indim = 128
        elif key_encoder_type == 'regnet':
            up_16_8_indim = 192
            up_8_4_indim = 96

        if self.aspp:
            indim = 256
        else:
            indim = 1024

        if self.aspp:
            self.ASPP = ASPP()

        self.compress = ResBlock(indim, 512)
        self.up_16_8 = UpsampleBlock(up_16_8_indim, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(up_8_4_indim, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, f16, f8, f4):
        if self.aspp:
            f16 = self.ASPP(f16)
        x = self.compress(f16)
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)

        x = self.pred(F.relu(x))
        
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x


class MemoryReader(nn.Module):
    def __init__(self):
        super().__init__()
 
    def get_affinity(self, mk, qk):
        B, CK, T, H, W = mk.shape
        mk = mk.flatten(start_dim=2)
        qk = qk.flatten(start_dim=2)

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk

        affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, THW, HW
        
        # softmax operation; aligned the evaluation style
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum 

        return affinity

    def readout(self, affinity, mv, qv):
        B, CV, T, H, W = mv.shape

        mo = mv.view(B, CV, T*H*W) 
        mem = torch.bmm(mo, affinity) # Weighted-sum B, CV, HW
        mem = mem.view(B, CV, H, W)

        mem_out = torch.cat([mem, qv], dim=1)

        return mem_out


class STCN(nn.Module):
    def __init__(self, single_object, value_encoder_type = 'resnet18', key_encoder_type = 'resnest101', aspp = False):
        super().__init__()
        self.single_object = single_object
        self.value_encoder_type = value_encoder_type
        self.key_encoder_type = key_encoder_type
        self.aspp = aspp


        self.key_encoder = KeyEncoder(self.key_encoder_type)
        if single_object:
            self.value_encoder = ValueEncoderSO(self.value_encoder_type, self.key_encoder_type) 
        else:
            self.value_encoder = ValueEncoder(self.value_encoder_type, self.key_encoder_type) 

        if self.key_encoder_type == 'resnest101' or self.key_encoder_type == 'resnet50' or self.key_encoder_type == 'wide_resnet50' or self.key_encoder_type == 'resnet50_v2':
            key_proj_indim = 1024
        elif self.key_encoder_type == 'convext':
            key_proj_indim = 512
        elif self.key_encoder_type == 'regnet':
            key_proj_indim = 384
        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(key_proj_indim, keydim=64)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(key_proj_indim, 512, kernel_size=3, padding=1)

        self.memory = MemoryReader()
        self.decoder = Decoder(self.key_encoder_type, self.aspp)

    def aggregate(self, prob):
        new_prob = torch.cat([
            torch.prod(1-prob, dim=1, keepdim=True),
            prob
        ], 1).clamp(1e-7, 1-1e-7)
        logits = torch.log((new_prob /(1-new_prob)))
        return logits

    def encode_key(self, frame): 
        # input: b*t*c*h*w
        b, t = frame.shape[:2]

        f16, f8, f4 = self.key_encoder(frame.flatten(start_dim=0, end_dim=1))
        k16 = self.key_proj(f16)
        f16_thin = self.key_comp(f16)

        # B*C*T*H*W
        k16 = k16.view(b, t, *k16.shape[-3:]).transpose(1, 2).contiguous()

        # B*T*C*H*W
        f16_thin = f16_thin.view(b, t, *f16_thin.shape[-3:])
        f16 = f16.view(b, t, *f16.shape[-3:])
        f8 = f8.view(b, t, *f8.shape[-3:])
        f4 = f4.view(b, t, *f4.shape[-3:])

        return k16, f16_thin, f16, f8, f4

    def encode_value(self, frame, kf16, mask, other_mask=None): 
        # Extract memory key/value for a frame
        if self.single_object:
            f16 = self.value_encoder(frame, kf16, mask)
        else:
            f16 = self.value_encoder(frame, kf16, mask, other_mask)
        return f16.unsqueeze(2) # B*512*T*H*W

    def segment(self, qk16, qv16, qf8, qf4, mk16, mv16, selector=None): 
        # q - query, m - memory
        # qv16 is f16_thin above
        affinity = self.memory.get_affinity(mk16, qk16)
        
        if self.single_object:
            logits = self.decoder(self.memory.readout(affinity, mv16, qv16), qf8, qf4)
            prob = torch.sigmoid(logits)
        else:
            logits = torch.cat([
                self.decoder(self.memory.readout(affinity, mv16[:,0], qv16), qf8, qf4),
                self.decoder(self.memory.readout(affinity, mv16[:,1], qv16), qf8, qf4),
            ], 1)

            prob = torch.sigmoid(logits)
            prob = prob * selector.unsqueeze(2).unsqueeze(2)

        logits = self.aggregate(prob)
        prob = F.softmax(logits, dim=1)[:, 1:]

        return logits, prob

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError


