import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T


import sys
sys.path.append('./')
sys.path.append('../')

class TP_transformer(nn.Module):
    def __init__(self):
        super(TP_transformer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.act = nn.PReLU()
        # Second Conv Layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
    
        
    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act((self.bn2(self.conv2(x))))
        return x

class TP_generator(nn.Module):
    def __init__(self) -> None:
        super(TP_generator,self).__init__()
        self.parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).train()
        
        
    def forward(self,input):
        out = self.parseq(input)
        return out
    
    def decode(self, preds):
        return self.parseq.tokenizer.decode(preds)
    

class TP_module(nn.Module):
    def __init__(self, ) -> None:
        super(TP_module,self).__init__()
        self.tp_generator = TP_generator()
        self.tp_transformer = TP_transformer()
    
    def generate_tp(self,interpolated_lr_image):
        out = self.tp_generator(interpolated_lr_image)
        probs = out.log_softmax(-1)
        return probs
        
    def forward(self,interpolated_lr_image):
        probs = self.generate_tp(interpolated_lr_image)
        
        out2 = probs.view(-1,1,probs.shape[1],probs.shape[2])
        
        tp_features = self.tp_transformer(out2)
        
        return probs, tp_features
