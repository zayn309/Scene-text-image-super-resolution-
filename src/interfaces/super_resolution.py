from base import TextBase
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model import model_sr


class TextSR(TextBase):
    def __init__(self, config, args):
        super().__init__(config, args)
        self.model = model_sr()
        self.opt = self.optimizer_init(self.model)
        self.eval_models_dic = {'crnn': self.CRNN_init(),
                                'moran':self.MORAN_init(),
                                'aster': self.Aster_init()}
        
        
    def train():
        pass
    def eval():
        pass 
    def test():
        pass
        
        
    