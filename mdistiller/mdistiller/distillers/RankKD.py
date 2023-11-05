import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def rankkd_reallocate_loss(logits_student, logits_teacher, temperature, rank_weight):
    p_s = F.softmax(logits_student / temperature, dim=1)
    p_t = F.softmax(logits_teacher / temperature, dim=1)
    loss = nn.KLDivLoss(reduction='none')(p_s.log(), p_t)

    prob = 0.5  # split into top 30 probs and bot 70 probs
    k = int(prob * p_t.size(1))

    # take top and bot loss based on the index of top 30 probs and bot 70 probs
    # topk function will sort the value
    top_p_t, top_idx = torch.topk(p_t, k, dim=1)
    bot_p_t, bot_idx = torch.topk(p_t, (p_t.size(1)-k), dim=1, largest=False)
    top_loss = torch.gather(loss, 1, index=top_idx)
    bot_loss = torch.gather(loss, 1, index=bot_idx)

    with torch.no_grad():
        # assign ranks ratio
        top_ranks = 1 - torch.arange(k) / ((k-1) * 2)
        bot_ranks = torch.arange((p_t.size(1)-k)) / ((p_t.size(1)-k-1) * 2) + 0.5
        top_ranks = top_ranks.cuda()
        bot_ranks = bot_ranks.cuda()

        # sum up bot_loss to be allocate
        tba_loss = bot_loss * (1 - bot_ranks)
        tba_loss = tba_loss.sum(1).unsqueeze(1)
        tba_loss = tba_loss * (top_ranks / top_ranks.sum())

        # allocate tba_loss from bot_loss to top_loss
        idx = top_loss != 0
        ratio = (top_loss[idx] + tba_loss[idx]) / top_loss[idx] + 1e-6

    top_loss[idx] *= ratio
    bot_loss *= bot_ranks

    loss = torch.cat((top_loss, bot_loss), 1)

    if torch.isnan(loss).any():
        n = torch.isnan(loss)
        n = (n == True).nonzero(as_tuple=False)[0][0]
        print(n)
        print(logits_student[n]/temperature)
        print(p_t[n])
        print(p_s[n])
        print(loss[n])
        assert(False)

    loss = loss.sum(1).mean() * temperature**2

    return loss, rank_weight


def rankkd_loss(logits_student, logits_teacher, temperature):
    p_s = F.softmax(logits_student / temperature, dim=1)
    p_t = F.softmax(logits_teacher / temperature, dim=1)
    entropy = -torch.sum(p_t * torch.log(p_t + 1e-10), dim=1) 
    
    ranks = p_t.argsort(axis=1).argsort(axis=1)
    ranks = (ranks / (p_t.size(1)-1)) 
    
    loss = nn.KLDivLoss(reduction='none')(p_s.log(), p_t)
    loss = loss * ranks * entropy.unsqueeze(1)
    
    return loss.sum(1).mean() * temperature**2, 0

def exp_rankkd_loss(logits_student, logits_teacher, temperature):
    p_s = F.softmax(logits_student / temperature, dim=1)
    p_t = F.softmax(logits_teacher / temperature, dim=1)
    entropy = -torch.sum(p_t * torch.log(p_t + 1e-10), dim=1) 
    
    ranks = p_t.argsort(axis=1).argsort(axis=1)
    ranks = (ranks / (p_t.size(1)-1)) 
    ranks = torch.exp(ranks)

    loss = nn.KLDivLoss(reduction='none')(p_s.log(), p_t)
    loss = loss * (ranks + entropy.unsqueeze(1))
    
    return loss.sum(1).mean() * temperature**2, 0


def old_rankkd_loss(logits_student, logits_teacher, temperature, upscale_ranks):
    p_s = F.softmax(logits_student / temperature, dim=1)
    p_t = F.softmax(logits_teacher / temperature, dim=1)
    
    ranks = p_t.argsort(axis=1).argsort(axis=1)
    ranks = (ranks / (p_t.size(1)-1)) * 2
    
    entropy = -torch.sum(p_t * torch.log(p_t + 1e-10), dim=1) 
    
    loss = nn.KLDivLoss(reduction='none')(p_s.log(), p_t)
    loss = loss * ranks * entropy.unsqueeze(1)  # Expand entropy tensor to match the shape of loss tensor
    # loss = nn.functional.relu(loss)
    
    return loss.sum(1).mean() * temperature**2, upscale_ranks


def rankkd_upscale_loss(logits_student, logits_teacher, temperature, rank_weight):
    p_s = F.softmax(logits_student / temperature, dim=1)
    p_t = F.softmax(logits_teacher / temperature, dim=1)
    loss = nn.KLDivLoss(reduction='none')(p_s.log(), p_t)
    ranks = p_t.argsort(axis=1).argsort(axis=1)
    # get the ranks of t's prob e.g. ranks[target] = 99
    ranks = (ranks / (logits_teacher.size(1)-1)) * 3
    loss = loss * ranks
    loss = loss.sum(1).mean() * temperature**2
    return loss, rank_weight


def rankmodkd_loss(logits_student, logits_teacher, temperature, rank_weight):
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = nn.KLDivLoss(reduction='none')(pred_student.log(), pred_teacher).abs()

    order = pred_teacher.argsort(axis=1, descending=False)
    ranks = order.argsort(axis=1)
    ranks = ranks / (logits_teacher.size(1)-1) * rank_weight
    loss_kd = loss_kd * ranks

    loss_kd = loss_kd.sum(1).mean() * temperature**2
    return loss_kd, rank_weight


def rankkd_rescale_loss(logits_student, logits_teacher, temperature, _):
    p_s = F.softmax(logits_student / temperature, dim=1)
    p_t = F.softmax(logits_teacher / temperature, dim=1)
    print(p_t[0], p_s[0])

    loss = nn.KLDivLoss(reduction='none')(p_s.log(), p_t)

    # sort loss according to p_t (sorted_loss[0] => loss of class with smallest p_t)
    order = p_t.argsort(axis=1)
    sorted_loss = loss.gather(1, order)

    # rescale idx above and below to 2.0-1.0 and 1.0-0.0
    ranks = torch.linspace(0, 1, p_t.size(1)).repeat(p_t.size(0), 1).cuda()
    prob = torch.tensor([0.67]).unsqueeze(1).cuda()
    mid_idx = (prob*p_t.size(1) - 1).long()
    mid = torch.gather(ranks, 1, index=mid_idx)
    mid2 = torch.gather(ranks, 1, index=mid_idx+1)
    # mid = prob.unsqueeze(1)

    top_ranks = ((ranks - mid2) / (1.0 - mid2)) * (1 / (1-prob) - 1) + 1
    top_ranks[top_ranks < 1] = 0
    bot_ranks = ranks / mid
    bot_ranks[bot_ranks > 1] = 0

    ranks = top_ranks + bot_ranks
    ranks = ranks
    loss = sorted_loss * ranks
    loss = loss.sum(1).mean() * temperature**2

    return loss, 0


def rankkd_cheb_loss(logits_student, logits_teacher, temperature, _):
    p_s = F.softmax(logits_student / temperature, dim=1)
    p_t = F.softmax(logits_teacher / temperature, dim=1)

    with torch.no_grad():
        t_var = p_t.var(dim=1)

        # a = p_t.std(1) - p_s.std(1)
        # target_idx = p_t.argmax(1)
        # tt = p_t[range(p_t.size(0)), target_idx]
        # ts = p_s[range(p_s.size(0)), target_idx]
        # a = nn.SmoothL1Loss(reduction='none')(tt, ts)
        a = torch.norm(p_t - p_s, float("inf"), dim=1)

        prob = (t_var / (t_var + a**2))
        rank_weight = 1 / (prob+1e-6)
        rank_weight = rank_weight.mean()
        print(rank_weight)

        ranks = p_t.argsort(axis=1).argsort(axis=1)
        ranks = (ranks / (p_t.size(1)-1)) * 2
        ranks *= rank_weight

    loss = nn.KLDivLoss(reduction='none')(p_s.log(), p_t)
    loss = loss * ranks

    if loss.isnan().any():
        print('p_t:\t', temp_p_t[torch.argmax(temp_rw)])
        print('p_s:\t', temp_p_s[torch.argmax(temp_rw)])
        print('rank_w:\t', temp_rw)
        print('loss:\t', loss)
        assert(False)

    temp_p_t = p_t
    temp_p_s = p_s
    temp_rw = rank_weight

    loss = loss.sum(1).mean() * temperature**2
    return loss, rank_weight.mean().item()


def rankkd_plus_loss(logits_student, logits_teacher, temperature, rank_weight):
    p_s = F.softmax(logits_student / temperature, dim=1)
    p_t = F.softmax(logits_teacher / temperature, dim=1)
    loss = nn.KLDivLoss(reduction='none')(p_s.log(), p_t)

    with torch.no_grad():
        # sort loss according to p_t (sorted_loss[0] => loss of class with smallest p_t)
        order = p_t.argsort(axis=1)
        sorted_loss = loss.gather(1, order)

        # set the index prob*100 to 1.0, increase greater indexs' loss and decrease smaller indexs' loss
        prob = 0.34
        mid_idx = (prob*p_t.size(1) - 1)
        ranks = torch.cat([torch.linspace(0, 1, int(mid_idx)+1)[:-1], torch.linspace(1, 4, 100-int(mid_idx))]).cuda()

    loss = loss * ranks
    loss = nn.functional.relu(loss)
    loss = loss.sum(1).mean() * temperature**2
    return loss, 0.34

class RankKD(Distiller):

    def __init__(self, student, teacher, cfg, upscale_ranks):
        super(RankKD, self).__init__(student, teacher)
        self.temperature = cfg.RankKD.TEMPERATURE
        self.ce_loss_weight = cfg.RankKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.RankKD.LOSS.KD_WEIGHT
        self.rank_weight = cfg.RankKD.LOSS.RK_WEIGHT
        self.upscale_ranks = upscale_ranks

    def forward_train(self, image, target, raw_data=False, **kwargs):
        torch.set_printoptions(threshold=10_000)
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd, upscale_ranks = exp_rankkd_loss(logits_student, logits_teacher, self.temperature)
        loss_kd *= self.kd_loss_weight
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict, upscale_ranks
    