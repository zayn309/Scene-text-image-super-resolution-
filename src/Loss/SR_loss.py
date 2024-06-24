import torch

class SR_loss():
    def __init__(self):
        pass
    
    def compute(self, h_r, g_t):
        """
        Computes the loss between two tensors h_r and g_t.
        
        Parameters:
        h_r (torch.Tensor): The generated tensor with shape (batch size, 3, 32, 128).
        g_t (torch.Tensor): The ground_truth tensor with shape (batch size, 3, 32, 128).
        
        Returns:
        float: loss.
        """
        if h_r.shape != g_t.shape:
            raise ValueError("The input tensors h_r and g_t must have the same shape.")
        
        loss = torch.sum(torch.abs(h_r - g_t)).item()

        # batch_size, channels, height, width = h_r.shape
        # loss = 0.0

        # for b in range(batch_size):
        #     for c in range(channels):
        #         for i in range(height):
        #             for j in range(width):
        #                 h_r_ij = h_r[b, c, i, j]
        #                 g_t_ij = g_t[b, c, i, j] 
        #                 loss += abs(h_r_ij - g_t_ij)
        
      
        return loss

# Example 
h_r = torch.tensor([[[[1], [2]]], [[[3], [1]]]])
g_t = torch.tensor([[[[2], [1]]], [[[3], [2]]]])
print('h_r',h_r)
print('g_t',g_t)

loss_calculator = SR_loss()
loss = loss_calculator.compute(h_r, g_t)
print("Loss:", loss)

