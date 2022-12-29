import gc
import os
import sys
import cv2
import glob
import json
import shutil
import random
import pydicom
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import timm
from timm.data import create_transform
from timm import create_model, list_models

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
from joblib import Parallel, delayed


class CFG:
    """
    Parameters used for training
    """
    seed = 42
    verbose = 1
    save_weights = True
    
    img_size = (1024, 512)
    batch_size = 8
    epochs = 5
    use_fp16 = True
    n_folds = 5
    train_folds = [0, 1]
    
    weight_decay = 0.024
    one_cycle_max_lr = 4e-4  # 8e-4
    
    # Model
    model_name = "efficientnet_b2"
    pretrained_weights = None
    num_classes = 1
    n_channels = 3
    
    pos_target_weight = 20
    target = "cancer"
    tta = True


def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')
    

class BreastCancerDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.paths = df['path'].values
        self.transforms = transforms
        self.targets = df['cancer'].values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Item accessor

        Args:
            idx (int): Index.

        Returns:
            np array [H x W x C]: Image.
            torch tensor [1]: Label.
            torch tensor [1]: Sample weight.
        """ 
        try:
            image = np.asarray(Image.open(self.paths[idx]).convert('RGB'))
        except Exception as ex:
            print(self.paths[idx], ex)
            return None
        
        if self.transforms:
            image = self.transforms(image=image)["image"]

        if CFG.target in self.df.columns:
            target = torch.as_tensor(self.df.iloc[idx].cancer)
            return image, target

        return image

def transformer(stage):
    if stage == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=5),
            A.augmentations.crops.RandomResizedCrop(height=CFG.img_size[0], width=CFG.img_size[1], scale=(0.8, 1), ratio=(0.45, 0.55)),
            A.Normalize(),
            A.pytorch.transforms.ToTensorV2()
        ])
    else:
        return A.Compose([
                A.Resize(CFG.img_size[0], CFG.img_size[1]),
                A.Normalize(),
                A.pytorch.transforms.ToTensorV2()
            ])


def plot_df(df):
    fig,ax = plt.subplots(2,2,figsize=(20,10))
    
    ax[0, 0].plot(df['train_loss'])
    ax[0, 0].plot(df['valid_loss'])
    ax[0, 0].legend()
    ax[0, 0].set_title('Loss')
    
    ax[1, 0].plot(df['pF1'])
    ax[1, 0].legend()
    ax[1, 0].set_title('pF1')
    
    ax[1, 1].plot(df['thres'])
    ax[1, 1].legend()
    ax[1, 1].set_title('Threshold')

def load_model_weights(model, filename, verbose=1, cp_folder="", strict=True):
    """
    Loads the weights of a PyTorch model. The exception handles cpu/gpu incompatibilities.

    Args:
        model (torch model): Model to load the weights to.
        filename (str): Name of the checkpoint.
        verbose (int, optional): Whether to display infos. Defaults to 1.
        cp_folder (str, optional): Folder to load from. Defaults to "".

    Returns:
        torch model: Model with loaded weights.
    """
    state_dict = torch.load(os.path.join(cp_folder, filename), map_location="cpu")

    try:
        model.load_state_dict(state_dict["model"], strict=strict)
    except BaseException:
        try:
            del state_dict['logits.weight'], state_dict['logits.bias']
            model.load_state_dict(state_dict, strict=strict)
        except BaseException:
            del state_dict['encoder.conv_stem.weight']
            model.load_state_dict(state_dict, strict=strict)

    if verbose:
        print(f"\n -> Loading encoder weights from {os.path.join(cp_folder,filename)}\n")

    return model, state_dict["threshold"], state_dict["model_type"]


def save_model(name, model, thres, model_type=CFG.model_name):
    torch.save({'model': model.state_dict(), 'threshold': thres, 'model_type': model_type}, f'{name}')


class BreastCancerModel(nn.Module):
    def __init__(
        self,
        model,
        num_classes=1,
        num_classes_aux=0,
        n_channels=3,
    ):
        """
        Constructor.

        Args:
            encoder (timm model): Encoder.
            num_classes (int, optional): Number of classes. Defaults to 1.
            num_classes_aux (int, optional): Number of aux classes. Defaults to 0.
            n_channels (int, optional): Number of image channels. Defaults to 3.
        """
        super().__init__()

        self.model = model
        self.backbone_dim = self.model(torch.randn(1, 3, CFG.img_size[0], CFG.img_size[1])).shape[-1]

        self.num_classes = num_classes
        self.n_channels = n_channels

        self.logits = nn.Linear(self.backbone_dim, num_classes)
        
        self._update_num_channels()

    def _update_num_channels(self):
        if self.n_channels != 3:
            for n, m in self.model.named_modules():
                if n:
                    # print("Replacing", n)
                    old_conv = getattr(self.model, n)
                    new_conv = nn.Conv2d(
                        self.n_channels,
                        old_conv.out_channels,
                        kernel_size=old_conv.kernel_size,
                        stride=old_conv.stride,
                        padding=old_conv.padding,
                        bias=old_conv.bias is not None,
                    )
                    setattr(self.model, n, new_conv)
                    break

    def forward(self, x, return_fts=False):
        """
        Forward function.

        Args:
            x (torch tensor [batch_size x n_channels x h x w]): Input batch.

        Returns:
            torch tensor [batch_size x num_classes]: logits.
            torch tensor [batch_size x num_classes_aux]: logits aux.
        """
        x = self.model(x)
        logits = self.logits(x).squeeze()

        return logits


def define_model(
    name,
    num_classes=1,
    num_classes_aux=0,
    n_channels=3,
    pretrained_weights="",
    pretrained=True,
):
    """
    Loads a pretrained model & builds the architecture.
    Supports timm models.

    Args:
        name (str): Model name
        num_classes (int, optional): Number of classes. Defaults to 1.
        num_classes_aux (int, optional): Number of aux classes. Defaults to 0.
        n_channels (int, optional): Number of image channels. Defaults to 3.
        pretrained_weights (str, optional): Path to pretrained encoder weights. Defaults to ''.
        pretrained (bool, optional): Whether to load timm pretrained weights.

    Returns:
        torch model -- Pretrained model.
    """
    # Load pretrained model
    encoder = create_model(CFG.model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=0.)
    encoder.name = name

    # Tile Model
    model = BreastCancerModel(
        encoder,
        num_classes=num_classes,
        num_classes_aux=num_classes_aux,
        n_channels=n_channels,
    )

    if pretrained_weights:
        model = load_model_weights(model, pretrained_weights, verbose=1, strict=False)

    return model


def pfbeta(labels, predictions, beta=1.):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / max(y_true_count, 1)  # avoid / 0
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0

def optimal_f1(labels, predictions):
    thres = np.linspace(0, 1, 101)
    f1s = [pfbeta(labels, predictions > thr) for thr in thres]
    idx = np.argmax(f1s)
    return f1s[idx], thres[idx]

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


def get_optimizer_and_scheduler(model, dataloader, optim="adam"):
    if optim == "adamw":
         optimizer = torch.optim.AdamW(
             add_weight_decay(model,
                              weight_decay=CFG.weight_decay,
                              skip_list=['bias']),
             lr=CFG.one_cycle_max_lr,
             betas=(0.9, 0.999),
             weight_decay=CFG.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters())
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1,
                                              max_lr=CFG.one_cycle_max_lr, epochs=CFG.epochs, steps_per_epoch=len(dataloader))
    return optimizer, scheduler




def train_one_epoch(dataloader, model, scheduler, optimizer, scaler, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Train: Epoch {epoch + 1}", total=len(dataloader), mininterval=5)
    
    for img, target in pbar:
        optimizer.zero_grad()
        img = img.to(device)

        # Using mixed precision training
        with autocast():
            outputs = model(img)
            loss = binary_cross_entropy_with_logits(
                outputs,
                target.to(float).to(device),
                pos_weight=torch.tensor([CFG.pos_target_weight]).to(device)
            )
            
            if np.isinf(loss.item()) or np.isnan(loss.item()):
                print(f'Bad loss, skipping the batch')
                del loss, outputs
                gc.collect()
                continue
        
        # scaler is needed to prevent "gradient underflow"
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        if scheduler is not None:
            scheduler.step()
        scaler.update()
        
        lr = scheduler.get_last_lr()[0] if scheduler else CFG.one_cycle_max_lr
        loss = loss.item()
        
        pbar.set_postfix({"loss": loss, "lr": lr})
        total_loss += loss
    
    total_loss /= len(dataloader)
    gc.collect()
    torch.cuda.empty_cache()
    return total_loss


def valid_one_epoch(dataloader, model, epoch):
    model.eval()
    pred_cancer = []
    with torch.no_grad():
        total_loss = 0
        targets = []
        pbar = tqdm(dataloader, desc=f'Eval: {epoch + 1}', total=len(dataloader), mininterval=5)

        for img, target in pbar:
            with autocast(enabled=True):
                img = img.to(device)

                outputs = model(img)
                if CFG.tta:
                    outputs2 = model(torch.flip(img, dims=[-1])) # horizontal mirror
                    outputs = (outputs + outputs2) / 2

                loss = binary_cross_entropy_with_logits(
                            outputs, 
                            target.to(float).to(device),
                            pos_weight=torch.tensor([CFG.pos_target_weight]).to(device)
                        ).item()
                
                pbar.set_postfix({"loss": loss})
                
                pred_cancer.append(torch.sigmoid(outputs))
                total_loss += loss
                targets.append(target.cpu().numpy())
             
    targets = np.concatenate(targets)
    pred = torch.concat(pred_cancer).cpu().numpy()
    pf1, thres = optimal_f1(targets, pred)

    total_loss /= len(dataloader)
    gc.collect()
    torch.cuda.empty_cache()
    return total_loss, pf1, thres, pred


def train_fnc(train_dataloader, valid_dataloader, model, fold, optimizer, scheduler):
    train_losses = []
    valid_losses = []
    valid_scores = []
    thresholds   = []
    
    scaler = GradScaler()
    best_loss = 999
    best_score = -1
    for epoch in range(CFG.epochs):
        train_loss = train_one_epoch(train_dataloader, model, scheduler, optimizer, scaler, epoch)
        valid_loss, valid_score, thres, pred = valid_one_epoch(valid_dataloader, model, epoch)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_scores.append(valid_score)
        thresholds.append(thres)
        
        if valid_score > best_score:
            best_score = valid_score
            save_model(f"fold{fold}_best_score.pth", model, thres)
            print("New Best Score")
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            save_model(f"fold{fold}_best_loss.pth", model, thres)
            print("New Best Loss")
        print()
        
        print(f"-------- Epoch {epoch + 1} --------")
        print("Train Loss: ", train_loss)
        print("Valid Loss: ", valid_loss)
        print("pF1: ", valid_score)
        print("Best Score: ", best_score)
        print("Best Loss: ", best_loss)
        print()
        
    column_names = ['train_loss','valid_loss', 'pF1', 'thres']
    df = pd.DataFrame(np.stack([train_losses, valid_losses, valid_scores, thresholds],
                               axis=1),columns=column_names)

    plot_df(df)



SAVE_FOLDER = "./"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(CFG.seed)

train = pd.read_csv("data/train.csv")
root = os.path.join(os.environ["DATASET_ROOT"], "bcd2022")
IMG_PATH = os.path.join(root, "images_1024")

train["path"] = [os.path.join(IMG_PATH, f"{p}_{im}.png") for p, im in zip(train["patient_id"].values, train["image_id"].values)]

skf = StratifiedKFold(CFG.n_folds)
train["fold"] = -1

for fold, (train_idx, val_idx) in enumerate(skf.split(train, train["cancer"])):
    train.loc[val_idx, "fold"] = fold

train.groupby("fold")["cancer"].value_counts()



for fold in range(CFG.n_folds):
    if fold not in CFG.train_folds: continue
        
    print("*"*10, f"Fold: {fold}", "*"*10)
    train_df = train[train["fold"] != fold]
    valid_df = train[train["fold"] == fold]
    
    train_dataset = BreastCancerDataset(train_df, transformer("train"))
    valid_dataset = BreastCancerDataset(valid_df, transformer("valid"))
    
    train_dataloader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=CFG.batch_size * 2, shuffle=False, pin_memory=True)

    model = define_model(CFG.model_name).to(device)
    model = nn.DataParallel(model)
    optimizer, scheduler = get_optimizer_and_scheduler(model, train_dataloader, "adamw")
    
    train_fnc(train_dataloader, valid_dataloader, model, fold, optimizer, scheduler)


def gen_predictions(models, train):
    train_predictions = []
    pbar = tqdm(enumerate(models), total=len(models), desc='Folds')
    for fold, model in pbar:
        if model is not None:
            eval_dataset = BreastCancerDataset(train.query('fold == @fold'), transformer("valid"))
            eval_dataloader = DataLoader(eval_dataset, batch_size=CFG.batch_size, shuffle=False)
            
            eval_loss, pF1, thres, pred = valid_one_epoch(eval_dataloader, model, -1)
            
            pbar.set_description(f'Eval fold:{fold} pF1:{pF1:.02f}')
            pred_df = pd.DataFrame(data=pred,
                                          columns=['cancer_pred_proba'])
            pred_df['cancer_pred'] = pred_df.cancer_pred_proba > thres

            df = pd.concat(
                [train.query('fold == @fold').reset_index(drop=True), pred_df],
                axis=1
            ).sort_values(['patient_id', 'image_id'])
            train_predictions.append(df)
    train_predictions = pd.concat(train_predictions)
    return train_predictions


print("-"*15, " BEST LOSS ", "-"*15)
models_path = glob.glob(f"./*best_loss.pth")


models = [load_model_weights(define_model(CFG.model_name).to(device), model)[0] for model in models_path]
pred_df = gen_predictions(models, train)
pred_df.to_csv('train_predictions.csv', index=False)

pred_df = pd.read_csv('train_predictions.csv')
print('F1 CV score (multiple thresholds):', f1_score(pred_df.cancer, pred_df.cancer_pred))    
pred_df = pred_df.groupby(['patient_id', 'laterality']).agg(
    cancer_max=('cancer_pred_proba', 'max'), cancer_mean=('cancer_pred_proba', 'mean'), cancer=('cancer', 'max')
)
print('pF1 CV score. Mean aggregation, single threshold:', optimal_f1(pred_df.cancer.values, pred_df.cancer_mean.values))
print('pF1 CV score. Max aggregation, single threshold:', optimal_f1(pred_df.cancer.values, pred_df.cancer_max.values))

print("-"*15, " BEST SCORE ", "-"*15)
models_path = glob.glob(f"./*best_score.pth")


models = [load_model_weights(define_model(CFG.model_name).to(device), model)[0] for model in models_path]
pred_df = gen_predictions(models, train)
pred_df.to_csv('train_predictions.csv', index=False)


pred_df = pd.read_csv('train_predictions.csv')
print('F1 CV score (multiple thresholds):', f1_score(pred_df.cancer, pred_df.cancer_pred))    
pred_df = pred_df.groupby(['patient_id', 'laterality']).agg(
    cancer_max=('cancer_pred_proba', 'max'), cancer_mean=('cancer_pred_proba', 'mean'), cancer=('cancer', 'max')
)
print('pF1 CV score. Mean aggregation, single threshold:', optimal_f1(pred_df.cancer.values, pred_df.cancer_mean.values))
print('pF1 CV score. Max aggregation, single threshold:', optimal_f1(pred_df.cancer.values, pred_df.cancer_max.values))