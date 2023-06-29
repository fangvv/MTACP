from thop import profile
from models.vgg_cifar import VGG
vgg = VGG('vgg16')
import torch
input = torch.randn(1, 3, 32, 32)
macs, params = profile(vgg, inputs=(input, ))
print(macs/1e6, params/1e6)