from base import TextBase
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model import model_sr
from Loss.general_loss import TotalLoss 
from LR_schedualer.LR_scheduler import LR_Scheduler


class TextSR(TextBase):
    def __init__(self, config, args):
        super().__init__(config, args)
        self.model = model_sr().to(self.device)
        self.opt = self.optimizer_init(self.model)
        self.eval_models_dic = {'crnn': self.CRNN_init(),
                                'moran':self.MORAN_init(),
                                'aster': self.Aster_init()}
        self.train_loader = self.get_train_data()
        self.val_dataset, self.val_loader = self.get_val_data()
        self.cri = TotalLoss(self.config)
        self.scheduler = LR_Scheduler(self.opt,self.config)
        
    def train():
        pass
    def eval():
        pass 
    def test():
        pass
        
