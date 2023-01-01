from .models import *
from .modules import *
import numpy as np


def factory_model(model_type, model_name, *args, **kwargs):
    if model_type == "timm":
        return TimmBackbone(model_name, *args, **kwargs)
    elif model_type == "timm_lf":
        return TimmBackboneLowFeatures(model_name, *args, **kwargs)
    else:
        raise ValueError()


def factory_loss(loss_type, unbalance, unbalance_perc, df_col):
    n_tot = len(df_col)
    n_val = len(df_col.unique())
    counts = df_col.value_counts()

    print(counts)
    if loss_type == "focal":
        if unbalance:
            alpha = 1.0 - counts[1] / n_tot
            alpha = unbalance_perc*alpha
        else:
            alpha = 0.5
        print(f"Factory Focal loss '{df_col.name}' with alpha={alpha}")
        return FocalLoss(alpha=alpha, gamma=2)
    elif loss_type == "bce":
        if unbalance:
            pos_weight = unbalance_perc*n_tot/counts[1]
            pos_weight = torch.tensor([pos_weight])
        else:
            pos_weight  = None
        print(f"Factory BCEWithLogitsLoss '{df_col.name}' with pos_weight={pos_weight}")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == "ce":
        if unbalance:
            weight = [(n_tot/counts[i]) for i in range(n_val)]
            m_w = max(weight)
            weight = torch.tensor([w / m_w / unbalance_perc for w in weight])
        else:
            weight = None
        print(f"Factory CrossEntropyLoss '{df_col.name}' with weight={weight}")
        return nn.CrossEntropyLoss(weight=weight, ignore_index=100)
    else:
        raise ValueError()



def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or np.any([v in name.lower()  for v in skip_list]):
            # print(name, 'no decay')
            no_decay.append(param)
        else:
            # print(name, 'decay')
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]



def factory_optimizer(opt_type, model, cfg):
    # TODO : optimizer factory
    if opt_type == "adamw":
        return torch.optim.AdamW(add_weight_decay(model, weight_decay=cfg.weight_decay, skip_list=['bias']),
                                 lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    if opt_type == "adam":
        return torch.optim.Adam(add_weight_decay(model, weight_decay=cfg.weight_decay, skip_list=['bias']),
                                 lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    elif opt_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.opt_sgd_momentum)
    else:
        raise ValueError()


def factory_lr_scheduler(lr_type, total_iter, opt, cfg):
    if lr_type == "cosineannealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_iter, eta_min=cfg.lr_cosineannealing_eta_mul*cfg.lr)
    elif lr_type == "step":
        return torch.optim.lr_scheduler.MultiStepLR(opt, milestones=cfg.lr_step_milestones, gamma=cfg.lr_step_gamma)
    else:
        raise ValueError()