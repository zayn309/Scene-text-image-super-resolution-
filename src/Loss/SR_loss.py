import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    """Pixel-wise L1 Loss"""

    def __init__(self):
        super(CharbonnierLoss, self).__init__()

    def forward(self, x, y):
        loss = torch.mean(torch.abs(x - y))
        return loss

