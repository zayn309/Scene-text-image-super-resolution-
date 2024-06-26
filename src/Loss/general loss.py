from Loss import SR_loss,KL_loss,TP_L1_loss

class general_loss():
    def __init__(self, alpha=1e-3, beta=1e-3):
        """
        Initializes the KLDivergence with a small epsilon value to avoid numeric errors.
        
        Parameters:
        alpha (float): 
        beta (float): 
        """
        self.alpha = alpha
        self.beta = beta
    
    def compute(self, tp_L, tp_gt, t_h , g_t):
        """
        Computes the KL divergence between two matrices t_L and t_H.
        
        Parameters:
        tp_L (numpy.ndarray): The generated tp (from the lower) matrix.
        t_H (numpy.ndarray): The higher generated matrix.
        g_t : the ground truth matrix
        tp_gt : The generated tp from ground truth matrix


        Returns:
        float: The total loss value.
        """
        sr_loss = SR_loss.compute(t_h , g_t)
        tp_l1_loss = generation_loss.compute(tp_l,tp_gt)
        kl_loss = TP_loss.compute(tp_L, tp_gt)

        total_loss = ls + self.alpha*sr_loss + self.beta * kl_loss
        
        return total_loss

