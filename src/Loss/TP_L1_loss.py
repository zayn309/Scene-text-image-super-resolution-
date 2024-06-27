import torch
import torch.nn as nn

import torch
import torch.nn as nn

class TP_L1Loss(nn.Module):
    def __init__(self):
        super(TP_L1Loss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, predictions, targets):
        """
        Forward pass for the L1 loss computation.
        
        Args:
        predictions (torch.Tensor): The predicted values, shape (B, 26, 95)
        targets (torch.Tensor): The ground truth values, shape (B, 26, 95)
        
        Returns:
        torch.Tensor: The L1 loss value
        """
        # Ensure the predictions and targets have the same shape
        assert predictions.shape == targets.shape, "Shapes of predictions and targets must match"
        
        # Compute the L1 loss
        loss = self.l1_loss(predictions, targets)
        
        return loss

# Example usage:
if __name__ == '__main__':
    # Suppose we have the following random predictions and targets tensors:
    B = 32  # Batch size
    seq_len = 26  # Sequence length
    vocab_size = 95  # Vocabulary size

    predictions = torch.randn(B, seq_len, vocab_size)
    targets = torch.randn(B, seq_len, vocab_size)

    # Initialize the custom L1 loss
    criterion = TP_L1Loss()

    # Calculate the L1 loss
    loss = criterion(predictions, targets)
    print("L1 Loss:", loss.item())
