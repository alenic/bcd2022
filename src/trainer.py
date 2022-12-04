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


def seed_all(random_state):
    random.seed(random_state)
    os.environ['PYTHONHASHSEED'] = str(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_config(config_file, output_folder="outputs"):
    with open(config_file, "r") as fp:
        cfg = yaml.load(fp, yaml.loader.SafeLoader)

    cfg = EasyDict(cfg)
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = cfg.model_name.replace("/","_").replace("@","_")
    folder_out = f"{date}_{model_name}"
    folder_out = os.path.join(output_folder, folder_out)

    return cfg, folder_out

def show_batch(image_tensor, label=None, mean=None, std=None):
    img = torch.clone(image_tensor)
    if mean is not None:
        if std is not None:
            img = T.Normalize([-x for x in mean], [1/s for s in std])
    plt.imshow(T.ToPILImage()(img))
    if label is not None:
        plt.title(str(label))
    plt.show()

class CVTrainer:
    def __init__(self,
                config_file,
                model,
                train_dataset,
                val_dataset,
                criterion,
                metric,
                maximize,
                show_dataset=False,
                output_folder="outputs",
                imb_callback=None,
                device="cuda:0",
                save_pth=True,
                ):
        # Get config
        cfg, output_folder = get_config(config_file, output_folder)
        self.cfg = cfg
        self.output_folder = output_folder

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

        model.to(device)

        n_iter_per_epoch = len(train_data_loader)
        total_iter = n_iter_per_epoch * cfg.n_epochs
        
        # TODO : optimizer factory
        if cfg.optimizer == "adamw":
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "adabelief":
            from adabelief_pytorch import AdaBelief
            opt = AdaBelief(model.parameters(), lr=cfg.lr, eps=1e-8, weight_decay=cfg.weight_decay, weight_decouple=False, rectify=False, fixed_decay=False, amsgrad=False)
        
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
        self.device = device
        self.metric = metric
        self.maximize = maximize
        self.save_pth = save_pth
        self.show_dataset = show_dataset
        
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
            self.train_epoch(data_loader=self.train_data_loader, model=self.model, optimizer=self.opt, scheduler=self.lr_scheduler, criterion=self.criterion,  device=self.device)
            y_pred, y_true, y_prob = self.eval(device=self.device)

            if isinstance(self.metric, list):
                score_list = []
                for m in self.metric:
                    score, metric_name = m(y_true, y_pred, y_prob)
                    self.summary.add_scalar(f"Val/{metric_name}", score, epoch)
                    print(f"Epoch {epoch} - Val {metric_name} {score}")
                    score_list.append(score)
                # take the first score list
                score = score_list[0]
            else:
                score, metric_name = self.metric(y_true, y_pred, y_prob)
                self.summary.add_scalar(f"Val/{metric_name}", score, epoch)
                print(f"Epoch {epoch} - Val {metric_name} {score}")
            
            print(f"Epoch {epoch} Time: {time.time()-t_epoch}")
            
            if epoch == 0:
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



    def train_epoch(self, data_loader, model, optimizer, scheduler, criterion, device="cuda:0"):
        model.train()

        pbar = tqdm(total=len(data_loader))
        n_iter = len(data_loader)
        for iter, (image, label) in enumerate(data_loader):
        
            if self.show_dataset:
                show_batch(image[0], label[0])
            optimizer.zero_grad()
            image = image.to(device)
            label = label.to(device)
            output = model(image).squeeze(-1)

            loss = criterion(output, label)
            loss.backward()

            optimizer.step()
            
            if iter % max(1, int(0.01*n_iter)) == 0:
                self.summary.add_scalar("Train/Loss", loss, self.global_iter)
                self.summary.add_scalar("Train/lr", optimizer.param_groups[0]['lr'], self.global_iter)
                self.global_iter += 1

            pbar.update()
            if scheduler is not None:
                scheduler.step()


    def eval(self, device="cuda:0"):
        self.model.eval()
        
        y_true = []
        y_pred = []
        y_prob = []
        
        for iter, (image, label) in enumerate(tqdm(self.val_data_loader)):
            image = image.to(device)
            with torch.no_grad():
                output = self.model(image)
                y_true += list(label.cpu().numpy())
                y_pred += list(torch.argmax(output,1).cpu().numpy())
                y_prob += list(torch.softmax(output,1).cpu().numpy())


        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        y_prob = np.vstack(y_prob)
        return y_pred, y_true, y_prob
