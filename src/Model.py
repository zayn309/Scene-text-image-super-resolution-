from TP_Module import TP_module
from model.Generator import Generator
import torch
import torch.nn as nn

class model_sr(nn.Module):
    def __init__(self) -> None:
        super(model_sr,self).__init__()
        self.tp_module = TP_module()
        self.generator = Generator()
    def forward(self,input_img):
        probs, tp_features = self.tp_module(input_img)
        sr_output = self.generator(input_img,tp_features)
        return sr_output , probs # returning the probs to calc the KL loss
