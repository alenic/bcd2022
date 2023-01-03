import numpy as np
import torch
from tqdm import tqdm
from .custom_metrics import *
from .models_factory import *
from .utils import *
import sklearn.metrics as skm

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

        eval_cols = [self.cfg.target] + self.cfg.aux_cols_name
        loss_type = [self.cfg.loss_target.loss_type] + self.cfg.aux_cols_type

        y_true_dict = {c: [] for c in eval_cols}
        y_out_dict = {c: [] for c in eval_cols}

        n_iter = len(self.val_data_loader)
        pbar = tqdm(total=10)
        for iter, (image, label) in enumerate(self.val_data_loader):
            image = image.to(self.device)
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    output = self.model(image)
            if tta:
                output_tta = self.model(image.flip(-1))

            for i in range(len(eval_cols)):
                y_out = output[i].cpu().numpy()
                if tta:
                    y_out += output_tta[i].cpu().numpy()
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
        metrics = {m: {} for m in ["f1score", "precision", "recall", "pr_thr", "f1score_thr", "pf1_mean", "pf1_max", "tpf1_mean", "tpf1_mean_thr", "tpf1_max", "tpf1_max_thr"]}

        # Eval target
        c = self.cfg.target
        y_prob_target = sigmoid(y_out[c].flatten())
        metrics["f1score"][c], best_thr = optimize_metric(f1, y_true[c], y_prob_target)
        metrics["precision"][c], metrics["recall"][c], metrics["pr_thr"][c] = skm.precision_recall_curve(y_true[c], y_prob_target)
        metrics["f1score_thr"][c] = best_thr

        df = pd.DataFrame()
        df["pid"] = self.df_val["patient_id"].astype(str) + "_" + self.df_val["laterality"].astype(str)
        df["y_out"] = y_out[c]
        df["y_true"] = y_true[c]
        df = df.groupby("pid")[["y_true","y_out"]].agg({ 'y_true':'max', 
                                                         'y_out':['mean','max']})
        
        df[("y_prob", "mean")] = sigmoid(df[("y_out", "mean")].values)
        df[("y_prob", "max")] = sigmoid(df[("y_out", "max")].values)

        metrics["pf1_mean"][c] = pf1(df["y_true"].values, df[("y_out", "mean")].values)
        metrics["pf1_max"][c] = pf1(df["y_true"].values, df[("y_out", "max")].values)

        best_score_mean = best_score_max = 0
        best_thr_max = best_thr_mean = 0
        print("Optimizing Score...")
        for thr in np.linspace(0.01, 0.99, 100):
            target = df["y_true"].values

            y_thr = (df[("y_prob", "mean")].values >= thr)
            score = pf1(target, y_thr)
            if score > best_score_mean:
                best_score_mean = score
                best_thr_mean = thr
            
            y_thr = (df[("y_prob", "max")].values >= thr)
            score = pf1(target, y_thr)
            if score > best_score_max:
                best_score_max = score
                best_thr_max = thr

        metrics["tpf1_mean"][c] = best_score_mean
        metrics["tpf1_mean_thr"][c] = best_thr_mean

        metrics["tpf1_max"][c] = best_score_max
        metrics["tpf1_max_thr"][c] = best_thr_max

        # Eval aux
        for i, c in enumerate(self.cfg.aux_cols_name):
            if self.cfg.aux_cols_type[i] in ["bce", "focal"]:
                y_prob = sigmoid(y_out[c])
                y_pred = (y_prob >= best_thr).astype(int)
                metrics["f1score"][c] = f1(y_true[c], y_pred)
            else:
                # TODO
                y_prob = torch.softmax(torch.tensor(y_out[c].astype(np.float32)), 1).cpu().numpy()
                y_pred = np.argmax(y_prob, 1)
                metrics["f1score"][c] = f1(y_true[c], y_pred, average="macro")

        return metrics