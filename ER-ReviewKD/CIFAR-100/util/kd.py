import torch.nn.functional as F
from torch import nn
import torch

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, entropy, t=4):
        p_s = F.softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s.log(), p_t, reduction="none")

        _p_t = F.softmax(y_t / t, dim=1)
        entropy = -torch.sum(_p_t * torch.log(_p_t.clamp(min=1e-10)), dim=1)

        loss = (loss * entropy.unsqueeze(1)).sum(1).mean() * self.T**2 
        return loss
