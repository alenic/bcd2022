from .models import *

def factory_model(model_type, model_name, *args, **kwargs):
    if model_type == "timm":
        return Timm(model_name, *args, **kwargs)
    elif model_type == "timm_mh":
        return MultiHeadTimm(model_name, *args, **kwargs)
    else:
        raise ValueError()