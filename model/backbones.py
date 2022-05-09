import torch
import torch.nn as nn
import timm


input_tensor = torch.randn(2, 3, 224, 224)

#########
m = timm.create_model('ecaresnet269d', features_only=True, pretrained=True)
o = m(input_tensor)

stage0 = nn.Sequential(m.conv1,m.bn1,m.act1)
stage1 = nn.Sequential(m.maxpool,m.layer1)
stage2 = m.layer2
stage3 = m.layer3
stage4 = m.layer4

for x in o:
    print(x.shape)

import pdb
pdb.set_trace()
#########
m = timm.create_model('resnest269e', features_only=True, pretrained=True)
o = m(input_tensor)

stage0 = nn.Sequential(m.conv1,m.bn1,m.act1)
stage1 = nn.Sequential(m.maxpool,m.layer1)
stage2 = m.layer2
stage3 = m.layer3
stage4 = m.layer4

for x in o:
    print(x.shape)

import pdb
pdb.set_trace()

#########
m = timm.create_model('resnet200d', features_only=True, pretrained=True)
o = m(input_tensor)

stage0 = nn.Sequential(m.conv1,m.bn1,m.act1)
stage1 = nn.Sequential(m.maxpool,m.layer1)
stage2 = m.layer2
stage3 = m.layer3
stage4 = m.layer4

for x in o:
    print(x.shape)

import pdb
pdb.set_trace()

#########
m = timm.create_model('seresnet152d', features_only=True, pretrained=True)
o = m(input_tensor)

stage0 = nn.Sequential(m.conv1,m.bn1,m.act1)
stage1 = nn.Sequential(m.maxpool,m.layer1)
stage2 = m.layer2
stage3 = m.layer3
stage4 = m.layer4

for x in o:
    print(x.shape)

import pdb
pdb.set_trace()
##########
m = timm.create_model('resnetv2_50x1_bit_distilled', features_only=True, pretrained=True)
o = m(input_tensor)

for x in o:
    print(x.shape)

###########
m = timm.create_model('resnet50', features_only=True, pretrained=True)
o = m(input_tensor)
stage0 = nn.Sequential(m.conv1,m.bn1,m.act1)
stage1 = nn.Sequential(m.maxpool,m.layer1)
stage2 = m.layer2
stage3 = m.layer3
stage4 = m.layer4
for x in o:
    print(x.shape)

###########
print('convext')
m = timm.create_model('convnext_base_in22ft1k', features_only=True, pretrained=True)
o = m(input_tensor)
stage0 = nn.Sequential(m.stem_0,m.stem_1)
stage1 = m.stages_0
stage2 = m.stages_1
stage3 = m.stages_2
stage4 = m.stages_3
for x in o:
    print(x.shape)


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

import pdb
pdb.set_trace()
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


