from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self):
        super(DistillKL, self).__init__()

    def forward(self, y_s, y_t, temp):
        T = temp.cuda()
        _p_t = F.softmax(y_t / 4, dim=1)
        entropy = -torch.sum(_p_t * torch.log(_p_t.clamp(min=1e-10)), dim=1)
        KD_loss = nn.KLDivLoss(reduction='none')(F.log_softmax(y_s/T, dim=1),
                                F.softmax(y_t/T, dim=1)) * entropy.unsqueeze(1)  
        
        return KD_loss.sum(1).mean() * T**2
