import torch
import torch.nn as nn
import torch.optim as optim
import sys
from typing import Tuple
import torch.nn.init as init

sys.path.append('../')
sys.path.append('./')
from model.MaxVit import MaxViT
from model.upscaler import UpscaleTransformModule

class Generator(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 depths: Tuple[int, ...] = (4, 4, 4, 4),
                 channels: Tuple[int, ...] = (128, 128, 128, 128)):
        
        super(Generator, self).__init__()
        
        self.in_channels = in_channels
        self.depth = depths
        self.channels = channels
        self.vit = MaxViT(in_channels=self.in_channels, depths=self.depth, channels=self.channels)
        self.upscaler = UpscaleTransformModule(in_channels=64)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)
            
    def forward(self, input,tp_features):
        # input should be of shape (B, 3, 16, 64)
        # and output should be of shape (B, 3, 32, 128)

        vit_features = self.vit(input, tp_features)
        upscaled_image = self.upscaler(vit_features)
        return upscaled_image
