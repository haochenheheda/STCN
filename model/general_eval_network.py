"""
eval_network.py - Evaluation version of the network
The logic is basically the same
but with top-k and some implementation optimization

The trailing number of a variable usually denote the stride
e.g. f16 -> encoded features with stride 16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.general_modules import *
from model.general_network import Decoder

class STCN(nn.Module):
    def __init__(self, value_encoder_type = 'resnet18', key_encoder_type = 'resnest101', aspp = False):
        super().__init__()
        self.key_encoder = KeyEncoder(key_encoder_type) 
        self.value_encoder = ValueEncoder(value_encoder_type, key_encoder_type) 


        if key_encoder_type in ['resnet50', 'wide_resnet50', 'resnest101', 'resnet50_v2', 'resnet200d', 'seresnet152d','resnest269e', 'ecaresnet269d']:
            key_proj_indim = 1024
        elif key_encoder_type == 'convext':
            key_proj_indim = 512
        elif key_encoder_type == 'regnet':
            key_proj_indim = 384

        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(key_proj_indim, keydim=64)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(key_proj_indim, 512, kernel_size=3, padding=1)

        self.decoder = Decoder(key_encoder_type, aspp)

    def encode_value(self, frame, kf16, masks): 
        k, _, h, w = masks.shape

        # Extract memory key/value for a frame with multiple masks
        frame = frame.view(1, 3, h, w).repeat(k, 1, 1, 1)
        # Compute the "others" mask
        if k != 1:
            others = torch.cat([
                torch.sum(
                    masks[[j for j in range(k) if i!=j]]
                , dim=0, keepdim=True)
            for i in range(k)], 0)
        else:
            others = torch.zeros_like(masks)

        f16 = self.value_encoder(frame, kf16.repeat(k,1,1,1), masks, others)

        return f16.unsqueeze(2)

    def encode_key(self, frame):
        f16, f8, f4 = self.key_encoder(frame)
        k16 = self.key_proj(f16)
        f16_thin = self.key_comp(f16)

        return k16, f16_thin, f16, f8, f4

    def segment_with_query(self, mem_bank, qf8, qf4, qk16, qv16): 
        k = mem_bank.num_objects

        readout_mem = mem_bank.match_memory(qk16)
        qv16 = qv16.expand(k, -1, -1, -1)
        qv16 = torch.cat([readout_mem, qv16], 1)

        return torch.sigmoid(self.decoder(qv16, qf8, qf4))
