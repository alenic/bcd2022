from .models import *
from .modules import *

def factory_model(model_type, model_name, *args, **kwargs):
    if model_type == "timm":
        return TimmBackbone(model_name, *args, **kwargs)
    elif model_type == "timm_lf":
        return TimmBackboneLowFeatures(model_name, *args, **kwargs)
    else:
        raise ValueError()


def factory_loss(loss_type, cfg):
    if loss_type == "focal":
        return FocalLoss(alpha=cfg.focal_loss_alpha, gamma=cfg.focal_loss_gamma)
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_type == "bce_weighted":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.bce_pos_weight))
    else:
        raise ValueError()



def factory_optimizer(opt_type, model, cfg):
    # TODO : optimizer factory
    if opt_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
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