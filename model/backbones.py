import torch
import torch.nn as nn
import timm


input_tensor = torch.randn(2, 3, 224, 224)

###########
print('regnetz_e8')
m = timm.create_model('regnetz_e8', features_only=True, pretrained=True)
stage0 = nn.Sequential(m.stem_conv1,m.stem_conv2)
stage1 = nn.Sequential(m.stem_conv3,m.stages_0)
stage2 = m.stages_1
stage3 = m.stages_2
stage4 = nn.Sequential(m.stages_3, m.final_conv)

o = m(input_tensor)
for x in o:
    print(x.shape)

###########
print('wide_resnet50_2')
m = timm.create_model('wide_resnet50_2', features_only=True, pretrained=True)
stage0 = nn.Sequential(m.conv1,m.bn1,m.act1)
stage1 = nn.Sequential(m.maxpool,m.layer1)
stage2 = m.layer2
stage3 = m.layer3
stage4 = m.layer4

o = m(input_tensor)
for x in o:
    print(x.shape)

###########
print('resnest101e')
m = timm.create_model('resnest101e', features_only=True, pretrained=True)
stage0 = nn.Sequential(m.conv1,m.bn1,m.act1)
stage1 = nn.Sequential(m.maxpool,m.layer1)
stage2 = m.layer2
stage3 = m.layer3
stage4 = m.layer4

o = m(input_tensor)
for x in o:
    print(x.shape)

import pdb
pdb.set_trace()

###########
print('xception65')
m = timm.create_model('xception65', features_only=True, pretrained=True)
o = m(input_tensor)
for x in o:
    print(x.shape)
import pdb
pdb.set_trace()
