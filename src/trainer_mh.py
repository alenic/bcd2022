import os
import numpy as np
import random
import torch
import yaml
from easydict import EasyDict
import datetime
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as T
from .metrics import *
import pandas as pd
from .trainer import *

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

    def eval(self, tta=False):
        self.model.eval()
        multi_cols = self.cfg.multi_cols

        y_true_dict = {c: [] for c in multi_cols}
        y_prob_dict = {c: [] for c in multi_cols}

        for iter, (image, label) in enumerate(tqdm(self.val_data_loader)):
            image = image.to(self.device)
            with torch.no_grad():
                output = self.model(image)
                if tta:
                    output_tta = self.model(image.flip(-1))

                for i in range(len(multi_cols)):
                    y_prob = torch.sigmoid(output[i].squeeze(-1)).cpu().numpy().flatten()
                    
                    if tta:
                        y_prob += torch.sigmoid(output_tta[i].squeeze(-1)).cpu().numpy().flatten()
                        y_prob /= 2.0

                    y_prob_dict[multi_cols[i]] += list(y_prob)
                    y_true_dict[multi_cols[i]] += list(label[:, i].cpu().numpy())

        for i in range(len(multi_cols)):
            y_true_dict[multi_cols[i]] = np.array(y_true_dict[multi_cols[i]] )
            y_prob_dict[multi_cols[i]] = np.array(y_prob_dict[multi_cols[i]] )
        
        return y_true_dict, y_prob_dict
    
    
    def eval_metrics(self,  tta=False, return_y=False, provided_y: tuple = None):
        if provided_y is None:
            y_true, y_prob = self.eval(tta)
        else:
            y_true, y_prob = provided_y

        multi_cols = self.cfg.multi_cols

        best_thr = {}
        f1score = {}
        pf1_mean = {}
        pf1_max = {}
        pf1_majority = {}
        y_pred = {}
        for c in multi_cols:
            f1score[c], best_thr[c] = optimize_f1_metric(f1, y_true[c], y_prob[c])
            y_pred[c] = (y_prob[c]>=best_thr[c]).astype(int)

            df = pd.DataFrame()
            df["pid"] = self.df_val["patient_id"].astype(str) + "_" + self.df_val["laterality"]
            df["y_pred"] = y_pred[c]
            df["y_true"] = y_true[c]
            df["y_prob"] = y_prob[c]

            df_true = df.groupby("pid")["y_true"].apply(lambda x: x.sum()>0).astype(int)
            # pf1 by mean
            df_pred = df.groupby("pid")["y_prob"].mean()
            pf1_mean[c] = pf1(df_true.values, df_pred.values)
            # pf1 by max
            df_pred = df.groupby("pid")["y_pred"].max()
            pf1_max[c] = pf1(df_true.values, df_pred.values)
            # pf1 by majority
            df_pred = df.groupby("pid")["y_pred"].apply(lambda x: x.sum() >= len(x)*0.5).astype(int)
            pf1_majority[c] = pf1(df_true.values, df_pred.values)

        if return_y:
            return f1score, pf1_mean, pf1_max, pf1_majority, best_thr, y_prob, y_pred, y_true

        return f1score, pf1_mean, pf1_max, pf1_majority, best_thr

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
        self.output_folder = get_output_folder(cfg, output_folder)

        if cfg.imbalanced:
            from .imbalanced import ImbalancedDatasetSampler
            assert imb_callback is not None
            sampler = ImbalancedDatasetSampler(train_dataset, callback_get_label=imb_callback)
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

        self.evaluator = CVMHEval(cfg, df_val, model, val_dataset)

        n_iter_per_epoch = len(train_data_loader)
        total_iter = n_iter_per_epoch * cfg.n_epochs
        
        # TODO : optimizer factory
        if cfg.optimizer == "adamw":
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        # TODO : lr factory
        if cfg.lr_scheduler == "cosineannealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_iter, eta_min=0.01*cfg.lr)
        elif cfg.lr_scheduler is None:
            lr_scheduler = None
        else:
            raise ValueError()

        self.summary = SummaryWriter(self.output_folder)
        # TODO dict(cfg) \r
        self.summary.add_text("config", yaml.dump(dict(cfg)))

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

        self.model.to(self.device)
        
    def is_better(self, score, best_score):
        if self.maximize:
            return score > best_score
        return score < best_score
    
    def train(self):
        # Train Cycle
        best_score = -np.inf if self.maximize else np.inf
        best_pth = None
        for epoch in range(1, self.cfg.n_epochs+1):
            t_epoch = time.time()
            self.train_epoch(data_loader=self.train_data_loader, model=self.model, optimizer=self.opt, scheduler=self.lr_scheduler, criterion=self.criterion)
            f1score, pf1_mean, pf1_max, pf1_majority, best_thr = self.evaluator.eval_metrics()

            score = max(pf1_mean["cancer"], pf1_max["cancer"], pf1_majority["cancer"])
            
            for c in self.cfg.multi_cols:
                self.summary.add_scalar(f"Val/{c}/f1score", f1score[c], epoch)
                print(f"Epoch {epoch} - Val f1score {f1score[c]}", "best thr:", best_thr[c])
                self.summary.add_scalar(f"Val/{c}/pf1_mean", pf1_mean[c], epoch)
                print(f"Epoch {epoch} - Val pf1_mean {pf1_mean[c]}")
                self.summary.add_scalar(f"Val/{c}/pf1_max", pf1_max[c], epoch)
                print(f"Epoch {epoch} - Val pf1_max {pf1_max[c]}")
                self.summary.add_scalar(f"Val/{c}/pf1_majority", pf1_majority[c], epoch)
                print(f"Epoch {epoch} - Val pf1_majority {pf1_majority[c]}")
            
            epoch_time = time.time()-t_epoch
            self.summary.add_scalar(f"Train/EpochTime", epoch_time, epoch)
            print(f"Epoch {epoch} Time: {epoch_time}")
            
            if epoch == 1:
                os.makedirs(self.output_folder, exist_ok=True)
                with open(os.path.join(self.output_folder, "config.yaml"), "w") as fp:
                    yaml.dump(dict(self.cfg), fp)
            
            if self.save_pth:
                if self.is_better(score, best_score):
                    pth = os.path.join(self.output_folder, f"E{epoch:04d}_{score}.pth")
                    if best_pth is not None:
                        os.remove(best_pth)
                    
                    best_pth = pth
                    best_score = score
                    torch.save(self.model.state_dict(), pth)



    def train_epoch(self, data_loader, model, optimizer, scheduler, criterion):
        model.train()

        pbar = tqdm(total=len(data_loader))
        n_iter = len(data_loader)
        for iter, (image, labels) in enumerate(data_loader):

            if self.show_dataset:
                show_batch(image[0], labels[0][0], mean=self.mean, std=self.std)
            
            optimizer.zero_grad()

            image = image.to(self.device)
            labels = labels.to(self.device)

            outputs = model(image)
            
            loss = criterion[0](outputs[0].squeeze(-1), labels[:,0].type(torch.float32))
            for i in range(1, len(outputs)):
                if self.model.heads_num[i] == 1:
                    loss += self.cfg.loss_weight*criterion[i](outputs[i].squeeze(-1), labels[:,i].type(torch.float32))
                else:
                    loss += self.cfg.loss_weight*criterion[i](outputs[i].squeeze(-1), labels[:,i])
            loss.backward()

            optimizer.step()
            
            if iter % max(1, int(0.01*n_iter)) == 0:
                self.summary.add_scalar("Train/Loss", loss, self.global_iter)
                self.summary.add_scalar("Train/lr", optimizer.param_groups[0]['lr'], self.global_iter)
                self.global_iter += 1

            pbar.update()
            if scheduler is not None:
                scheduler.step()


