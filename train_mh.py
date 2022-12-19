import torch
import os
import pandas as pd
from src import *

config_file = "config/default_mh.yaml"
n_folds = 1

root = os.path.join(os.environ["DATASET_ROOT"], "bcd2022")
root_images = os.path.join(root, "images_1024")


if __name__ == "__main__":
    cfg = get_config(config_file)
    seed_all(cfg.random_state)
    df = pd.read_csv(os.path.join("data", "train_5fold.csv"))

    for fold in range(n_folds):
        df_train = df[df["fold"] != fold]
        df_val = df[df["fold"] == fold]
        print("-- Fold selected --")
        
        # Undersampling
        if cfg.max_examples is not None:
            df_train_neg = df_train[df_train["cancer"] == 0] .sample(cfg.max_examples)
            df_train_pos = df_train[df_train["cancer"] == 1]
            df_train = pd.concat([df_train_neg, df_train_pos])
            print("-- Fold undersampled --")

        print(df_train["cancer"].value_counts())
        print(df_val["cancer"].value_counts())

        heads_num = []
        criterion = []
        for col in cfg.multi_cols:
            uniq = df[col].unique()
            n = len(uniq)
            if n == 2:
                n = 1
            heads_num += [n]
            criterion += [nn.BCEWithLogitsLoss() if n==1 else nn.CrossEntropyLoss()]


        model = factory_model(cfg.model_type, cfg.model_name, hidden=cfg.hidden, heads_num=heads_num, drop_rate=cfg.drop_rate)
        
        if cfg.model_ckpt is not None:
            print(model.load_state_dict(torch.load(cfg.model_ckpt)))

        train_dataset = BCDDataset(root_images,
                                   df_train,
                                   multi_cols=cfg.multi_cols,
                                   transform=transform_albumentations(get_train_tr(cfg.input_size, cfg.severity, cfg.mean, cfg.std)))
                                   
        val_dataset = BCDDataset(root_images,
                                 df_val,
                                 multi_cols=cfg.multi_cols,
                                 transform=transform_albumentations(get_val_tr(cfg.test_input_size, cfg.mean, cfg.std)))

        trainer = CVMHTrainer(cfg,
                            model,
                            train_dataset,
                            val_dataset,
                            df_val,
                            criterion,
                            maximize=True,
                            show_dataset=False,
                            output_folder="outputs",
                            imb_callback=lambda dataset, idx: dataset.label[idx],
                            save_pth=True)
        trainer.train()
