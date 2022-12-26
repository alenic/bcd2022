import os
import numpy as np
import random
import torch
import yaml
from easydict import EasyDict
import datetime
import shutil
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as T
import sklearn.metrics as skm
import pandas as pd
from .custom_metrics import *
from .models_factory import *
from .utils import *

def seed_all(random_state):
    random.seed(random_state)
    os.environ['PYTHONHASHSEED'] = str(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_output_folder(cfg, root="outputs"):
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = cfg.model_name.replace("/","_").replace("@","_")
    folder_out = f"{date}_{model_name}"
    folder_out = os.path.join(root, folder_out)
    return folder_out

def get_config(config_file, output_folder="outputs"):
    with open(config_file, "r") as fp:
        cfg = yaml.load(fp, yaml.loader.SafeLoader)
    
    cfg = EasyDict(cfg)
    cfg.yaml_file = config_file

    if not isinstance(cfg.loss_aux.unbalance, list):
        cfg.loss_aux.unbalance = [cfg.loss_aux.unbalance]*len(cfg.aux_cols)
    if not isinstance(cfg.loss_aux.unbalance_perc, list):
        cfg.loss_aux.unbalance_perc = [cfg.loss_aux.unbalance_perc]*len(cfg.aux_cols)
    if not isinstance(cfg.loss_aux_weights, list):
        cfg.loss_aux_weights = [cfg.loss_aux_weights]*len(cfg.aux_cols)
    
    
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

class CVMHEval:
    def __init__(self, cfg, df_val, model, val_dataset):
        self.cfg = cfg
        self.df_val = df_val

        val_data_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=cfg.test_batch_size,
            num_workers=cfg.num_workers,
            shuffle=False
        )

        self.val_data_loader = val_data_loader
        
        self.model = model
        self.device = cfg.device

        model.to(self.device)

    def eval(self, tta=False, criterion=None):
        self.model.eval()

        if criterion is not None:
            loss_target_list = []

        eval_cols = [self.cfg.target] + self.cfg.aux_cols
        loss_type = [self.cfg.loss_target.loss_type] + self.cfg.loss_aux.loss_type

        y_true_dict = {c: [] for c in eval_cols}
        y_prob_dict = {c: [] for c in eval_cols}

        for iter, (image, label) in enumerate(tqdm(self.val_data_loader)):
            image = image.to(self.device)
            with torch.no_grad():
                output = self.model(image)

                if criterion is not None:
                    label = label.to(self.device)
                    loss_target_list += [criterion[0](output[0].squeeze(-1), label[:,0].type(torch.float32)).item()]
                if tta:
                    output_tta = self.model(image.flip(-1))

            for i in range(len(eval_cols)):
                if loss_type[i] in ["bce", "focal"]:
                    y_prob = torch.sigmoid(output[i].squeeze(-1)).cpu().numpy().flatten()
                    
                    if tta:
                        y_prob += torch.sigmoid(output_tta[i].squeeze(-1)).cpu().numpy().flatten()
                        y_prob /= 2.0
                else:
                    y_prob = torch.softmax(output[i], 1).cpu().numpy()
                    
                    if tta:
                        y_prob += torch.sigmoid(output_tta[i], 1).cpu().numpy()
                        y_prob /= 2.0

                y_prob_dict[eval_cols[i]] += list(y_prob)
                y_true_dict[eval_cols[i]] += list(label[:, i].cpu().numpy())

        for i in range(len(eval_cols)):
            y_true_dict[eval_cols[i]] = np.array(y_true_dict[eval_cols[i]] )
            y_prob_dict[eval_cols[i]] = np.array(y_prob_dict[eval_cols[i]] )
        

        if criterion is not None:
            return y_true_dict, y_prob_dict, np.mean(loss_target_list)

        return y_true_dict, y_prob_dict
    
    
    def eval_metrics(self, y_true: dict, y_prob: dict):
        metrics = {m: {} for m in ["f1score", "pf1_mean", "pf1_max", "pf1_majority", "precision", "recall", "pr_thr"]}
        thresholds = {}
        
        # Eval target
        c = self.cfg.target
        metrics["f1score"][c], best_thr = optimize_metric(f1, y_true[c], y_prob[c])
        metrics["precision"][c], metrics["recall"][c], metrics["pr_thr"][c] = skm.precision_recall_curve(y_true[c], y_prob[c])
        
        thresholds[c] = best_thr
        y_pred = (y_prob[c] >= best_thr).astype(int)

        metrics["pf1_max"][c] = grouped_reduced(y_true[c], y_pred, self.df_val, reduce="max")
        metrics["pf1_majority"][c] = grouped_reduced(y_true[c], y_pred, self.df_val, reduce="majority")
        metrics["pf1_mean"][c] = grouped_mean(y_true[c], y_prob[c], self.df_val, thr=best_thr)

        # Eval aux
        for i, c in enumerate(self.cfg.aux_cols):
            if self.cfg.loss_aux.loss_type[i] in ["bce", "focal"]:
                y_pred = (y_prob[c] >= best_thr).astype(int)
                metrics["f1score"][c] = f1(y_true[c], y_pred)
            else:
                y_pred = np.argmax(y_prob[c], 1)
                metrics["f1score"][c] = f1(y_true[c], y_pred, average="macro")

        return metrics, thresholds


class CVMHTrainer:
    def __init__(self,
                cfg,
                model: torch.nn.Module,
                train_dataset: torch.utils.data.Dataset,
                val_dataset: torch.utils.data.Dataset,
                df_val: pd.DataFrame,
                criterion: list,
                maximize: bool,
                show_dataset=False,
                output_folder="outputs",
                imb_callback=None,
                save_pth=True,
                ):
        # Get config
        self.cfg = cfg
        self.output_folder = output_folder

        self.summary = SummaryWriter(self.output_folder)
        # Copy configuration to outputs
        shutil.copy(self.cfg.yaml_file, os.path.join(self.output_folder, "config.yaml"))
        # Save config text to tensorboard

        cfg_dict = dict(self.cfg)
        cfg_dict = flatten_dict(cfg_dict)
        dict_str = ""
        for k in cfg_dict:
            cfg_dict[k] = str(cfg_dict[k])
            dict_str += f"**{k}** : {cfg_dict[k]}  \n"
        self.summary.add_text("config", dict_str)

        if cfg.imbalance_sampler:
            from .imbalanced import ImbalancedDatasetSampler
            assert imb_callback is not None
            sampler = ImbalancedDatasetSampler(train_dataset,
                                               callback_get_label=imb_callback,
                                               perc_sample=cfg.perc_sample)
            train_data_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                sampler = sampler,
                shuffle=False
            )
        else:
            train_data_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                shuffle=True
            )
            
        val_data_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=cfg.test_batch_size,
            num_workers=cfg.num_workers,
            shuffle=False
        )

        model.to(cfg.device)
        criterion = [c.to(cfg.device) for c in criterion]
        self.evaluator = CVMHEval(cfg, df_val, model, val_dataset)

        n_iter_per_epoch = len(train_data_loader)
        total_iter = n_iter_per_epoch * cfg.n_epochs
        opt = factory_optimizer(cfg.optimizer, model, cfg)
        lr_scheduler = factory_lr_scheduler(cfg.lr_scheduler, total_iter, opt, cfg)

        self.global_iter = 0

        self.cfg = cfg
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        
        self.model = model
        self.opt = opt
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.device = cfg.device
        self.maximize = maximize
        self.save_pth = save_pth
        self.show_dataset = show_dataset
        self.mean = cfg.mean
        self.std = cfg.std

        
    def is_better(self, score, best_score):
        if self.maximize:
            return score > best_score
        return score < best_score
    
    def train(self):
        # Train Cycle
        for epoch in range(1, self.cfg.n_epochs+1):
            t_epoch = time.time()
            self.train_epoch(epoch, data_loader=self.train_data_loader, model=self.model, optimizer=self.opt, scheduler=self.lr_scheduler, criterion=self.criterion)
            y_true_dict, y_prob_dict, val_loss = self.evaluator.eval(tta=self.cfg.tta, criterion=self.criterion)

            self.summary.add_scalar(f"Train/ValLoss{self.cfg.target}", val_loss, epoch)

            metrics, thresholds = self.evaluator.eval_metrics(y_true_dict, y_prob_dict)

            # Add Target metrics to summary
            c = self.cfg.target
            for m in ["f1score", "pf1_mean", "pf1_max", "pf1_majority"]:
                print(f"Epoch {epoch} - Val {c} -> {m} {metrics[m][c]}")
                self.summary.add_scalar(f"Val_{c}/{m}", metrics[m][c], epoch)
            
            self.summary.add_scalar(f"Val_{c}/best_f1_thr", thresholds[c], epoch)
            print(f"best_f1score_thr: {thresholds[c]}")
            self.summary.add_pr_curve(f"Val_{c}/pr", y_true_dict[c], y_prob_dict[c], epoch)

            for c in self.cfg.aux_cols:
                for m in ["f1score"]:
                    print(f"Epoch {epoch} - Val {c} -> {m} {metrics[m][c]}")
                    self.summary.add_scalar(f"Val_{c}/{m}", metrics[m][c], epoch)

  
            epoch_time = time.time()-t_epoch
            self.summary.add_scalar(f"Time/EpochTime", epoch_time, epoch)
            print(f"Epoch {epoch} Time: {epoch_time}")
            
            if epoch == 1:
                os.makedirs(self.output_folder, exist_ok=True)
                best_score = {m: -np.inf if self.maximize else np.inf for m in metrics}
                best_pth = {m: None for m in metrics}
            
            if self.save_pth:
                for m in ["f1score", "pf1_mean", "pf1_max", "pf1_majority"]:
                    score = metrics[m][self.cfg.target]
                    if self.is_better(score, best_score[m]):
                        pth = os.path.join(self.output_folder, f"E{epoch:04d}_{m}_thr{thresholds[self.cfg.target]:.4f}_{100*score:.4f}.pth")
                        if best_pth[m] is not None:
                            os.remove(best_pth[m])
                        
                        best_pth[m] = pth
                        best_score[m] = score
                        torch.save(self.model.state_dict(), pth)



    def train_epoch(self, epoch, data_loader, model, optimizer, scheduler, criterion):
        model.train()
        y_train_true = []
        y_train_prob = []
        pbar = tqdm(total=len(data_loader))
        n_iter = len(data_loader)
        for iter, (image, labels) in enumerate(data_loader):

            y_train_true += list(labels[:,0].cpu().numpy())

            if self.show_dataset:
                show_batch(image[0], labels[0][0], mean=self.mean, std=self.std)
            
            optimizer.zero_grad()

            image = image.to(self.device)
            labels = labels.to(self.device)

            outputs = model(image)
            
            y_train_prob += list(torch.sigmoid(outputs[0]).detach().squeeze(-1).cpu().numpy().flatten())
            loss = criterion[0](outputs[0].squeeze(-1), labels[:,0].type(torch.float32))
            for i in range(1, len(criterion)):

                if self.model.heads_num[i] == 1:
                    loss += self.cfg.loss_aux_weights[i-1]*criterion[i](outputs[i].squeeze(-1), labels[:,i].type(torch.float32))
                else:
                    loss += self.cfg.loss_aux_weights[i-1]*criterion[i](outputs[i].squeeze(-1), labels[:,i])
            loss.backward()

            optimizer.step()
            
            if iter % max(1, int(0.01*n_iter)) == 0:
                self.summary.add_scalar("Train/Loss", loss, self.global_iter)
                self.summary.add_scalar("Train/lr", optimizer.param_groups[0]['lr'], self.global_iter)
                self.global_iter += 1

            pbar.update()
            if scheduler is not None:
                scheduler.step()
        
        y_train_true = np.array(y_train_true)
        y_train_prob = np.array(y_train_prob)

        prob_pos = y_train_prob[y_train_true==1]
        prob_neg = y_train_prob[y_train_true==0]
        loss_pos = -np.mean(np.log(prob_pos))
        loss_neg = -np.mean(np.log(1-prob_neg))
        

        self.summary.add_pr_curve("Train/pr", y_train_true, y_train_prob, epoch)
        self.summary.add_scalar("Train/loss_pos", loss_pos, epoch)
        self.summary.add_scalar("Train/loss_neg", loss_neg, epoch)

        unique, counts = np.unique(y_train_true, return_counts=True)
        print("Train Pos Neg", unique, counts, counts/sum(counts))

