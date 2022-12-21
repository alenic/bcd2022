from .models import *

def factory_model(model_type, model_name, *args, **kwargs):
    if model_type == "timm":
        return TimmBackbone(model_name, *args, **kwargs)
    else:
        raise ValueError()