from .models import *

def factory_model(model_type, model_name, num_classes, *args, **kwargs):
    if model_type == "timm":
        return Timm(model_name, num_classes=num_classes, *args, **kwargs)
    
    else:
        raise ValueError()