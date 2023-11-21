import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

from ._base import Distiller


def kd_loss(logits_student, logits_teacher, temperature):
    p_s = F.softmax(logits_student / temperature, dim=1)
    p_t = F.softmax(logits_teacher / temperature, dim=1)
    loss = F.kl_div(p_s.log(), p_t, reduction="none").sum(1).mean() * temperature**2
    return loss


def er_kd_loss(logits_student, logits_teacher, temperature, t=4):
    p_s = F.softmax(logits_student / temperature, dim=1)
    p_t = F.softmax(logits_teacher / temperature, dim=1)
    loss = F.kl_div(p_s.log(), p_t, reduction="none")

    _p_t = F.softmax(logits_teacher / t, dim=1)
    entropy = -torch.sum(_p_t * torch.log(_p_t.clamp(min=1e-10)), dim=1)

    loss = (loss * entropy.unsqueeze(1)).sum(1).mean() * temperature**2 
    return loss


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg, t, er):
        super(KD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.t = t
        self.er = er
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.tea_name = cfg.DISTILLER.TEACHER

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)

        with torch.no_grad():
            if self.tea_name in ['deit', 'vit', 'swin']:
                image = transforms.Resize(384, interpolation=InterpolationMode.BICUBIC)(image)
                logits_teacher = self.teacher(image)
            else:
                logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        if self.er == True:
            loss_kd = self.kd_loss_weight * er_kd_loss(
                logits_student, logits_teacher, self.temperature
            )
        else:
            loss_kd = self.kd_loss_weight * kd_loss(
                logits_student, logits_teacher, self.temperature
            )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict