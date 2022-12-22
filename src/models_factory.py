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