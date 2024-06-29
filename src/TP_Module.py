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
        
        
    def forward(self,x):
        x = self.parseq(x)
        return x
    
    def decode(self, preds):
        return self.parseq.tokenizer.decode(preds)
    

class TP_module(nn.Module):
    def __init__(self, ) -> None:
        super(TP_module,self).__init__()
        self.tp_generator = TP_generator()
        self.tp_transformer = TP_transformer()
    
    def generate_tp(self,x):
        x = self.tp_generator(x).log_softmax(-1)
        return x
        
    def forward(self,x):
        x = self.generate_tp(x)
        shape1 = x.shape[1]
        shape2 = x.shape[2]
        x = x.view(-1,1,shape1,shape2)
        
        tp_features = self.tp_transformer(x)
        del shape1, shape2
        return x, tp_features
