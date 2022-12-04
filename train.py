import timm
import torch
import torchvision.transforms as T
import os
import numpy as np
import pandas as pd
from src import *

config_file = "config/deafult.yaml"

root = os.path.join(os.environ["DATASET_ROOT"], "Breast-Cancer-Detection-2022")
root_images = os.path.join(os.environ["DATASET_ROOT"], "Breast-Cancer-Detection-2022", "train_images_processed_512")


if __name__ == "__main__":
    cfg, _ = get_config(config_file)
    seed_all(cfg.random_state)

    train_tr = T.Compose([
                        T.RandomRotation(10),
                        T.RandomResizedCrop((cfg.input_size, cfg.input_size), scale=(0.85, 1), ratio=(0.9, 1.1)),
                        T.RandomHorizontalFlip(),
                        T.RandomVerticalFlip(),
                        T.ColorJitter(0.1, 0.1),
                        T.ToTensor(),
                        T.RandomErasing(scale=(0.01,0.03)),
                        T.RandomErasing(scale=(0.01,0.03)),
                        ])

    val_tr = T.Compose([    T.Resize((cfg.test_input_size, cfg.test_input_size)),
                            T.ToTensor(),
                            ])


    df_train = pd.read_csv(os.path.join(root, "train_split.csv"))
    df_val = pd.read_csv(os.path.join(root, "val_split.csv"))

    df_train = df_train[(df_train["difficult_negative_case"] == False)]
    df_val = df_val[(df_val["difficult_negative_case"] == False)]


    train_dataset = BCDDataset(root_images, df_train, transform=train_tr)
    val_dataset = BCDDataset(root_images, df_val, transform=val_tr)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    model = timm.create_model(cfg.model_name, pretrained=True, num_classes=2, in_chans=1, drop_rate=cfg.drop_rate)
    if cfg.model_ckpt is not None:
        print(model.load_state_dict(torch.load(cfg.model_ckpt)))


    trainer = CVTrainer(config_file,
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