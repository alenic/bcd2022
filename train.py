import timm
import torch
import torchvision.transforms as T
import os
import numpy as np
import pandas as pd
from src import *

config_file = "config/default.yaml"

root = os.path.join(os.environ["DATASET_ROOT"], "Breast-Cancer-Detection-2022")
root_images = os.path.join(os.environ["DATASET_ROOT"], "Breast-Cancer-Detection-2022", "train_images_processed_512")


if __name__ == "__main__":
    cfg, _ = get_config(config_file)
    seed_all(cfg.random_state)

    train_tr = T.Compose([
                        T.RandomApply([T.RandomRotation(10)], p=0.8),
                        T.Resize((cfg.test_input_size, cfg.test_input_size)),
                        #T.RandomResizedCrop((cfg.input_size, cfg.input_size), scale=(0.85, 1), ratio=(0.9, 1.1)),
                        T.RandomHorizontalFlip(),
                        #T.RandomVerticalFlip(),
                        T.ColorJitter(0.1, 0.1),
                        T.ToTensor(),
                        T.RandomErasing(scale=(0.001,0.01), ratio=(0.8, 1.2)),
                        T.RandomErasing(scale=(0.001,0.01), ratio=(0.8, 1.2)),
                        T.RandomErasing(scale=(0.001,0.01), ratio=(0.8, 1.2)),
                        T.RandomErasing(scale=(0.001,0.01), ratio=(0.8, 1.2)),
                        T.RandomErasing(scale=(0.001,0.01), ratio=(0.8, 1.2)),
                        T.RandomErasing(scale=(0.001,0.01), ratio=(0.8, 1.2)),
                        T.RandomErasing(scale=(0.001,0.01), ratio=(0.8, 1.2)),
                        T.RandomErasing(scale=(0.001,0.01), ratio=(0.8, 1.2)),
                        T.RandomErasing(scale=(0.001,0.01), ratio=(0.8, 1.2)),
                        ])

    val_tr = T.Compose([    T.Resize((cfg.test_input_size, cfg.test_input_size)),
                            T.ToTensor(),
                            ])


    model = factory_model(cfg.model_type, cfg.model_name, num_classes=2, drop_rate=cfg.drop_rate, hidden=cfg.hidden)
    if cfg.model_ckpt is not None:
        print(model.load_state_dict(torch.load(cfg.model_ckpt)))
    

    for i in range(4):
        df_train = pd.read_csv(os.path.join(root, "train_split.csv"))
        df_val = pd.read_csv(os.path.join(root, "val_split.csv"))

        df_train = df_train[(df_train["difficult_negative_case"] == False)]
        df_val = df_val[(df_val["difficult_negative_case"] == False)]

        df_train_neg = df_train[df_train["cancer"] == 0].sample(10000)
        df_train_pos = df_train[df_train["cancer"] == 1]
        df_train = pd.concat([df_train_neg, df_train_pos])

        print(df_train["cancer"].value_counts())
        print(df_val["cancer"].value_counts())

        train_dataset = BCDDataset(root_images, df_train, transform=train_tr)
        val_dataset = BCDDataset(root_images, df_val, transform=val_tr)

        criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        trainer = CVTrainer(cfg,
                            model,
                            train_dataset,
                            val_dataset,
                            criterion,
                            [pf1, f1],
                            maximize=True,
                            show_dataset=False,
                            output_folder="outputs",
                            imb_callback=lambda dataset, idx: dataset.label[idx],
                            save_pth=True)
        #y_pred, y_true, y_prob = trainer.eval()
        #score, metric_name = pf1(y_true, y_pred, y_prob)
        #print(f"Start Val {metric_name} {score}")
        trainer.train()

        cfg.lr = cfg.lr*0.9