#farghal task

import torch

class KL_loss():
    def __init__(self, epsilon=1e-5):
        """
        Initializes the KLDivergence with a small epsilon value to avoid numeric errors.
        
        Parameters:
        epsilon (float): A small positive number to avoid numeric error in division and logarithm.
        """
        self.epsilon = epsilon

    def compute(self, tp_l, tp_gt):
        """
        Computes the KL divergence between two tensors t_L and t_gt.
        
        Parameters:
        tp_L (torch.Tensor): The lower tensor with shape (batch size, 26, 95).
        tp_gt (torch.Tensor): The ground truth tensor with shape (batch size, 26, 95).
        
        Returns:
        float: The KL divergence value.
        """
        if tp_l.shape != tp_gt.shape:
            raise ValueError("The input tensors t_L and t_gt must have the same shape.")
        
        # Add epsilon to avoid numerical errors
        tp_l += self.epsilon
        tp_gt += self.epsilon
        
        # Compute the KL divergence
        kl_divergence = torch.sum(tp_gt * torch.log(tp_gt / tp_l)).item()
        
        return kl_divergence

# Example 
# tp_l = torch.rand(2, 26, 95)
# tp_gt = torch.rand(2, 26, 95)
# loss_calculator = KL_loss()
# loss = loss_calculator.compute(tp_l, tp_gt)
# print("KL Divergence Loss:", loss)

