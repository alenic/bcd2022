import plotly.express as px
import pandas as pd
import numpy as np
import torch.nn as nn
import random
import os
import torch
from collections.abc import MutableMapping
import datetime
import yaml
import matplotlib.pyplot as plt
from easydict import EasyDict
import torchvision.transforms as T
import cv2

def seed_all(random_state):
    random.seed(random_state)
    os.environ['PYTHONHASHSEED'] = str(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_interactive_precision_recall_curve(precision, recall, threshold):
    df = pd.DataFrame()
    df["threshold"] = np.concatenate([np.array([0]), threshold])
    df["precision"] = precision[::-1]
    df["recall"] = recall[::-1]
    fig = px.area(
        data_frame=df, 
        x="recall", 
        y="precision",
        hover_data=["threshold"], 
        title='Precision-Recall Curve'
    )
    fig.update_layout(
        autosize=False,
        width=500,
        height=500,
        margin=dict(l=30, r=30, b=30, t=30, pad=4),
        title_x=.5, # Centre title
        hovermode = 'closest',
        xaxis=dict(hoverformat='.4f'),
        yaxis=dict(hoverformat='.4f')
    )
    hovertemplate = 'Recall=%{x}<br>Precision=%{y}<br>Threshold=%{customdata[0]:.4f}<extra></extra>'
    fig.update_traces(hovertemplate=hovertemplate)
    
    # Add dashed line with a slope of 1
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    fig.show()


def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str ='.') -> MutableMapping:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_state_dict_improved(state_dict, model: nn.Module, replace_str=None, prepend=None):
    model_state_dict = model.state_dict()
    ckpt_state_dict = {}

    for key in state_dict:
        keyr = key
        if replace_str is not None:
            keyr = keyr.replace(replace_str[0], replace_str[1])
        if prepend is not None:
            keyr = prepend + keyr
        ckpt_state_dict[keyr] = state_dict[key]
    

    n_load = 0
    for key in model_state_dict:
        if key in ckpt_state_dict.keys():
            model_state_dict[key] = ckpt_state_dict[key]
            n_load += 1
        else:
            print(f"model {key} is not in checkpoint")

    for key in ckpt_state_dict:
        if key not in model_state_dict.keys():
            print(f"checkpoint {key} is not in model")
    
    return model.load_state_dict(model_state_dict)




def get_output_folder(cfg, root="outputs"):
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = cfg.model_name.replace("/","_").replace("@","_")
    folder_out = f"{date}_{model_name}"
    folder_out = os.path.join(root, folder_out)
    return folder_out

def get_config(config_file):
    with open(config_file, "r") as fp:
        cfg = yaml.load(fp, yaml.loader.SafeLoader)
    
    cfg = EasyDict(cfg)
    cfg.yaml_file = config_file

    if not isinstance(cfg.loss_aux.weights, list):
        cfg.loss_aux.weights = [cfg.loss_aux.weights]*len(cfg.aux_cols)
    
    if not isinstance(cfg.loss_aux.unbalance_perc, list):
        cfg.loss_aux.unbalance_perc = [cfg.loss_aux.unbalance_perc]*len(cfg.aux_cols)

    cfg.aux_cols_name = [c[0] for c in cfg.aux_cols]
    cfg.aux_cols_type = [c[1] for c in cfg.aux_cols]
    cfg.aux_cols_balance = [c[2] for c in cfg.aux_cols]

    print(cfg)
    return cfg

def optimize_metric(metric_func, y_true, y_prob, N=100, dtype=int):
    best_score = 0
    for thr in np.linspace(0, 1, N):
        y_pred = (y_prob>=thr).astype(dtype)
        score = metric_func(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_thr = thr
    
    return best_score, best_thr


def show_batch(image_tensor, label=None, mean=None, std=None):
    img = torch.clone(image_tensor)
    if mean is not None:
        if std is not None:
            img = (img*std + mean)
    plt.imshow(T.ToPILImage()(img))
    if label is not None:
        plt.title(str(label))
    plt.show()


def cv2_loader(path, in_chans=1):
    if in_chans == 1:
        return cv2.imread(path, 0)
    else:
        return cv2.cvtColor(cv2.imread(path, 0), cv2.COLOR_GRAY2RGB)

def sigmoid(x):
    def _positive_sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def _negative_sigmoid(x):
        # Cache exp so you won't have to calculate it twice
        exp = np.exp(x)
        return exp / (exp + 1)
    x = np.array(x)
    positive = x >= 0
    # Boolean array inversion is faster than another comparison
    negative = ~positive

    # empty contains junk hence will be faster to allocate
    # Zeros has to zero-out the array after allocation, no need for that
    # See comment to the answer when it comes to dtype
    result = np.empty_like(x)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])

    return result

if __name__ == "__main__":
    print(sigmoid(1000), sigmoid(-1000))