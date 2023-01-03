import os
import numpy as np
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
from .eval_mh import *
from pytorch_metric_learning.samplers import MPerClassSampler

import numpy as np
import torch
from tqdm import tqdm
from .custom_metrics import *
from .models_factory import *
from .utils import *

class CVMHGroupedEval:
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

    def eval(self, tta=False):
        self.model.eval()

        eval_cols = [self.cfg.target] + self.cfg.aux_cols_name
        loss_type = [self.cfg.loss_target.loss_type] + self.cfg.aux_cols_type

        y_true_dict = {c: [] for c in eval_cols}
        y_out_dict = {c: [] for c in eval_cols}

        n_iter = len(self.val_data_loader)
        pbar = tqdm(total=10)
        for iter, (image, label, breast_id) in enumerate(self.val_data_loader):
            image = image.to(self.device)
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    output = self.model(image)
            if tta:
                output_tta = self.model(image.flip(-1))

            for i in range(len(eval_cols)):
                if loss_type[i] in ["gbce", "bce", "focal"]:
                    y_out = output[i].squeeze(-1).cpu().numpy().flatten()
                    
                    if tta:
                        y_out += output_tta[i].squeeze(-1).cpu().numpy().flatten()
                        y_out /= 2.0
                    
                    y_out_dict[eval_cols[i]] += list(y_out)
                    y_true_dict[eval_cols[i]] += list(label[:, i].cpu().numpy().flatten())
                #else:
                #    y_out = output[i].cpu().numpy()

            if iter % max(1, int(0.1*n_iter)) == 0:
                pbar.update()
        
        for i in range(len(eval_cols)):
            y_true_dict[eval_cols[i]] = np.array(y_true_dict[eval_cols[i]] )
            y_out_dict[eval_cols[i]] = np.array(y_out_dict[eval_cols[i]] )

        return y_true_dict, y_out_dict

    
    def eval_metrics(self, y_true: dict, y_out: dict):
        metrics = {m: {} for m in ["pf1_mean", "pf1_max", "tpf1_mean", "tpf1_mean_thr", "tpf1_max", "tpf1_max_thr"]}
        #thresholds = {}
        
        # Eval target
        c = self.cfg.target
        df = pd.DataFrame()
        df["pid"] = self.df_val["patient_id"].astype(str) + "_" + self.df_val["laterality"].astype(str)
        df["y_out"] = y_out[c]
        df["y_true"] = y_true[c]
        df = df.groupby("pid")[["y_true","y_out"]].agg({'y_true':'max', 
                                                         'y_out':['mean','max']})
        
        df[("y_out", "mean")] = df[("y_out", "mean")].clip(-24, 1000)
        df[("y_out", "max")] = df[("y_out", "max")].clip(-24, 1000)
        df[("y_prob", "mean")] = 1 / (1 + np.exp(-df[("y_out", "mean")].values))
        df[("y_prob", "max")] = 1 / (1 + np.exp(-df[("y_out", "max")].values))

        metrics["pf1_mean"][c] = pf1(df["y_true"].values, df[("y_out", "mean")].values)
        metrics["pf1_max"][c] = pf1(df["y_true"].values, df[("y_out", "max")].values)

        best_score_mean = best_score_max = 0
        best_thr_max = best_thr_mean = 0
        print("Optimizing Score...")
        for thr in np.linspace(0.01, 0.99, 100):
            y_true = df["y_true"].values

            y_pred = (df[("y_prob", "mean")].values >= thr)
            score = pf1(y_true, y_pred)
            if score > best_score_mean:
                best_score_mean = score
                best_thr_mean = thr
            
            y_pred = (df[("y_prob", "max")].values >= thr)
            score = pf1(y_true, y_pred)
            if score > best_score_max:
                best_score_max = score
                best_thr_max = thr

        metrics["tpf1_mean"][c] = best_score_mean
        metrics["tpf1_mean_thr"][c] = best_thr_mean

        metrics["tpf1_max"][c] = best_score_max
        metrics["tpf1_max_thr"][c] = best_thr_max

        # # Eval aux
        # for i, c in enumerate(self.cfg.aux_cols_name):
        #     if self.cfg.aux_cols_type[i] in ["bce", "focal"]:
        #         y_pred = (y_prob[c] >= best_thr).astype(int)
        #         metrics["f1score"][c] = f1(y_true[c], y_pred)
        #     else:
        #         y_pred = np.argmax(y_prob[c], 1)
        #         metrics["f1score"][c] = f1(y_true[c], y_pred, average="macro")

        return metrics




class CVMHGroupedTrainer:
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
            sampler = MPerClassSampler(train_dataset.breast_id,
                                       batch_size=cfg.batch_size,
                                       m=2,
                                       length_before_new_iter=len(train_dataset))

            train_data_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                sampler=sampler
            )
            
        val_data_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=cfg.test_batch_size,
            num_workers=cfg.num_workers,
            shuffle=False
        )

        model.to(cfg.device)
        criterion = [c.to(cfg.device) for c in criterion]
        self.evaluator = CVMHGroupedEval(cfg, df_val, model, val_dataset)

        n_iter_per_epoch = len(train_data_loader)
        total_iter = n_iter_per_epoch * cfg.n_epochs
        opt = factory_optimizer(cfg.optimizer, model, cfg)
        lr_scheduler = factory_lr_scheduler(cfg.lr_scheduler, total_iter, opt, cfg)

        self.global_iter = 0
        self.df_val = df_val
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

            y_true_dict, y_out_dict = self.evaluator.eval(tta=self.cfg.tta)

            val_loss = self.criterion[0](torch.tensor(y_out_dict[self.cfg.target].astype(np.float32)),
                                         torch.tensor(y_true_dict[self.cfg.target].astype(np.float32)),
                                         torch.tensor(self.df_val["breast_id"].values.astype(int)))
            self.summary.add_scalar(f"Train/ValLoss{self.cfg.target}", val_loss, epoch)

            # This is to prevent soft labeling
            y_true_dict[self.cfg.target] = self.df_val["target"].values
            metrics = self.evaluator.eval_metrics(y_true_dict, y_out_dict)

            # Add Target metrics to summary
            c = self.cfg.target

            for m in ["pf1_mean", "pf1_max", "tpf1_mean", "tpf1_mean_thr", "tpf1_max", "tpf1_max_thr"]:
                print(f"Epoch {epoch} - Val {c} -> {m} {metrics[m][c]}")
                self.summary.add_scalar(f"Val_{c}/{m}", metrics[m][c], epoch)

            for c in self.cfg.aux_cols_name:
                for m in ["f1score"]:
                    print(f"Epoch {epoch} - Val {c} -> {m} {metrics[m][c]}")
                    self.summary.add_scalar(f"Val_{c}/{m}", metrics[m][c], epoch)
            
            if epoch == 1:
                os.makedirs(self.output_folder, exist_ok=True)
                best_score = {m: -np.inf if self.maximize else np.inf for m in metrics}
                best_pth = {m: None for m in metrics}
            
            if self.save_pth:
                for m in ["pf1_mean", "pf1_max", "tpf1_mean", "tpf1_max"]:
                    score = metrics[m][self.cfg.target]
                    if self.is_better(score, best_score[m]):
                        pth = os.path.join(self.output_folder, f"E{epoch:04d}_{m}_{100*score:.4f}.pth")
                        if best_pth[m] is not None:
                            os.remove(best_pth[m])
                        
                        best_pth[m] = pth
                        best_score[m] = score
                        torch.save(self.model.state_dict(), pth)

            epoch_time = time.time()-t_epoch
            self.summary.add_scalar(f"Time/EpochTime", epoch_time, epoch)
            print(f"Epoch {epoch} Time: {epoch_time}")

    def train_epoch(self, epoch, data_loader, model, optimizer, scheduler, criterion):
        model.train()
        y_train_true = []
        y_train_prob = []
        n_iter = len(data_loader)
        pbar = tqdm(total=10)
        for iter, (image, labels, breast_id) in enumerate(data_loader):
            y_train_true += list(labels[:,0].cpu().numpy())

            if self.show_dataset:
                show_batch(image[0], labels[0][0], mean=self.mean, std=self.std)
            
            optimizer.zero_grad()

            image = image.to(self.device)
            labels = labels.to(self.device)
        
            with torch.cuda.amp.autocast():
                outputs = model(image)
            
            y_train_prob += list(torch.sigmoid(outputs[0]).detach().squeeze(-1).cpu().numpy().flatten())

            loss = criterion[0](outputs[0].squeeze(-1), labels[:,0].type(torch.float32), breast_id)

            for i in range(1, len(criterion)):
                if self.model.heads_num[i] == 1:
                    loss += self.cfg.loss_aux.weights[i-1]*criterion[i](outputs[i].squeeze(-1), labels[:,i].type(torch.float32))
                else:
                    loss += self.cfg.loss_aux.weights[i-1]*criterion[i](outputs[i].squeeze(-1), labels[:,i].type(torch.long))

            loss.backward()

            optimizer.step()
            
            if iter % max(1, int(0.01*n_iter)) == 0:
                self.summary.add_scalar("Train/Loss", loss, self.global_iter)
                self.summary.add_scalar("Train/lr", optimizer.param_groups[0]['lr'], self.global_iter)
                self.global_iter += 1
            
            if iter % max(1, int(0.1*n_iter)) == 0:
                pbar.update()

            if scheduler is not None:
                scheduler.step()
            
        
        y_train_true = np.array(y_train_true)
        y_train_prob = np.array(y_train_prob)

        prob_pos = y_train_prob[y_train_true>0.5]
        prob_neg = y_train_prob[y_train_true<0.5]
        loss_pos = -np.mean(np.log(prob_pos + 1e-6))
        loss_neg = -np.mean(np.log(1-prob_neg+ 1e-6))
        

        self.summary.add_pr_curve("Train/pr", y_train_true, y_train_prob, epoch)
        self.summary.add_scalar("Train/loss_pos", loss_pos, epoch)
        self.summary.add_scalar("Train/loss_neg", loss_neg, epoch)

        unique, counts = np.unique(y_train_true, return_counts=True)
        print(f"Tain Epoch {epoch} finished - Train Pos Neg", unique, counts, counts/sum(counts))

