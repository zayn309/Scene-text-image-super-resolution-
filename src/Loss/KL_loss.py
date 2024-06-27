import torch
import torch.nn as nn
import torch.nn.functional as F

class TP_KlLoss(nn.Module):
    def __init__(self, reduction='batchmean',log_target = True):
        super(TP_KlLoss, self).__init__()
        self.reduction = reduction
        self.log_target = log_target

    def forward(self, input, target):

        kl_div = F.kl_div(input, target, reduction=self.reduction,log_target=self.log_target)

        return kl_div

# Example usage
if __name__ == "__main__":
    batch_size = 4
    sequence_length = 26
    vocab_size = 95
    
    input = torch.randn(batch_size, sequence_length, vocab_size, requires_grad=True)
    input = F.log_softmax(input, dim=-1) 
    target = torch.randn(batch_size, sequence_length, vocab_size)
    target = F.log_softmax(target, dim=-1)
    criterion = TP_KlLoss()
    loss = criterion(input, target)
    print(f'KL Divergence Loss: {loss.item()}')
