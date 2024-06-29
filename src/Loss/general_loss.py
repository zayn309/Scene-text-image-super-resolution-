import torch
import torch.nn as nn 
from Loss.KL_loss import TP_KlLoss
from Loss.SR_loss import CharbonnierLoss
from Loss.TP_L1_loss import TP_L1Loss

class TotalLoss(nn.Module):
    def __init__(self,config):
        super(TotalLoss, self).__init__()
        self.config = config
        self.charbonnier_loss = CharbonnierLoss()
        self.tp_kl_loss = TP_KlLoss()
        self.tp_l1_loss = TP_L1Loss()
        self.gamma = self.config.TRAIN.loss.gamma
        self.beta = self.config.TRAIN.loss.beta
        self.alpha = self.config.TRAIN.loss.alpha 
        

    def forward(self, sr, hr, TP_lr, TP_hr):
        charbonnier_loss = self.charbonnier_loss(sr, hr)
        tp_kl_loss = self.tp_kl_loss(TP_lr, TP_hr)
        tp_l1_loss = self.tp_l1_loss(TP_lr, TP_hr)
        
        total_loss = (self.gamma * charbonnier_loss) + (self.alpha * tp_kl_loss) + (self.beta * tp_l1_loss)
        return {'ch_loss': charbonnier_loss.item(),
                'kl_loss': tp_kl_loss.item(),
                'l1_loss': tp_l1_loss.item(),
                'total_loss': total_loss}
        
    def __str__(self):
        return (f"TotalLoss with configuration:\n"
                f"  Charbonnier Loss weight (gamma): {self.gamma}\n"
                f"  KL Divergence Loss weight (alpha): {self.alpha}\n"
                f"  L1 Loss weight (beta): {self.beta}\n"
                f"  Configuration: {self.config}")