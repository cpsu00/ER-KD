import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def entropy_DIST_loss(logits_student, logits_teacher, temperature, beta, gamma):
    p_s = F.softmax(logits_student / temperature, dim=1)
    p_t = F.softmax(logits_teacher / temperature, dim=1)

    _p_t = F.softmax(logits_teacher / 4, dim=1)
    entropy = -torch.sum(_p_t * torch.log(_p_t.clamp(min=1e-10)), dim=1)

    inter_loss = temperature**2 * inter_class_relation(p_s, p_t) * entropy.unsqueeze(1)
    intra_loss = temperature**2 * intra_class_relation(p_s, p_t) * entropy.unsqueeze(1)
    kd_loss = beta * inter_loss.mean() + gamma * intra_loss.mean()

    return kd_loss


def DIST_loss(logits_student, logits_teacher, temperature, beta, gamma):
    p_s = F.softmax(logits_student / temperature, dim=1)
    p_t = F.softmax(logits_teacher / temperature, dim=1)
    inter_loss = temperature**2 * inter_class_relation(p_s, p_t)
    intra_loss = temperature**2 * intra_class_relation(p_s, p_t)

    kd_loss = beta * inter_loss.mean() + gamma * intra_loss.mean()

    return kd_loss

def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)

def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t)


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(Distiller):

    def __init__(self, student, teacher, cfg):
        super(DIST, self).__init__(student, teacher)
        self.temperature = cfg.DIST.TEMPERATURE
        self.ce_loss_weight = cfg.DIST.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.DIST.LOSS.KD_WEIGHT
        self.beta = cfg.DIST.LOSS.beta
        self.gamma = cfg.DIST.LOSS.gamma

    def forward_train(self, image, target, **kwargs):
        torch.set_printoptions(threshold=10_000)
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = entropy_DIST_loss(logits_student, logits_teacher, self.temperature, self.beta, self.gamma)
        loss_kd *= self.kd_loss_weight
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict, 0
