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

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3,5), stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(4)
        self.act1 = nn.PReLU(num_parameters=4)

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3,5), stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(8)
        self.act2 = nn.PReLU(num_parameters=8)

        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,5), stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(16)
        self.act3 = nn.PReLU(num_parameters=16)

        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,5), stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(16)
        self.act4 = nn.PReLU(num_parameters=16)

        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,5), stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(16)
        self.act5 = nn.PReLU(num_parameters=16)

        self.conv6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,5), stride=1, padding=(1,0))
        self.bn6 = nn.BatchNorm2d(16)
        self.act6 = nn.PReLU(num_parameters=16)

        self.conv7 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,5), stride=1, padding=(1,0))
        self.bn7 = nn.BatchNorm2d(32)
        self.act7 = nn.PReLU(num_parameters=32)

        self.conv8 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,6), stride=1, padding=(1,1))
        self.bn8 = nn.BatchNorm2d(32)
        self.act8 = nn.PReLU(num_parameters=32)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))
        x = self.act5(self.bn5(self.conv5(x)))
        x = self.act6(self.bn6(self.conv6(x)))
        x = self.act7(self.bn7(self.conv7(x)))
        x = self.act8(self.bn8(self.conv8(x)))
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Define the layers
        # First Conv Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.act = nn.PReLU()
        # Second Conv Layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
    def forward(self, x):
        # Forward pass through the layers
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
        self.tp_transformer = ConvNet()
    
    def generate_tp(self,interpolated_lr_image):
        out = self.tp_generator(interpolated_lr_image)
        out = out.log_softmax(-1)
        return out.unsqueeze(1)
        
    def forward(self,interpolated_lr_image):
        probs = self.generate_tp(interpolated_lr_image)

        tp_features = self.tp_transformer(probs)
        
        return probs, tp_features
