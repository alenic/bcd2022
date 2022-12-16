import torch
import os
import pandas as pd
from src import *

config_file = "config/default.yaml"
n_folds = 5

root = os.path.join(os.environ["DATASET_ROOT"], "bcd2022")
root_images = os.path.join(root, "images_1024")


if __name__ == "__main__":
    cfg = get_config(config_file)
    seed_all(cfg.random_state)
    df = pd.read_csv(os.path.join("data", "train_5fold.csv"))

    for fold in range(n_folds):
        model = factory_model(cfg.model_type, cfg.model_name, num_classes=1, drop_rate=cfg.drop_rate, hidden=cfg.hidden)
        
        if cfg.model_ckpt is not None:
            print(model.load_state_dict(torch.load(cfg.model_ckpt)))
        
        df_train = df[df["fold"] != fold]
        df_val = df[df["fold"] == fold]
        print("-- Fold selected --")
        
        # Undersampling
        df_train_neg = df_train[df_train["cancer"] == 0].sample(cfg.max_examples)
        df_train_pos = df_train[df_train["cancer"] == 1]
        df_train = pd.concat([df_train_neg, df_train_pos])
        df_train.index = np.arange(len(df_train))
        df_val.index = np.arange(len(df_val))
        print("-- Fold undersampled --")

        print(df_train["cancer"].value_counts())
        print(df_val["cancer"].value_counts())

        train_dataset = BCDDataset(root_images, df_train, transform=transform_albumentations(get_train_tr(cfg.input_size, cfg.severity, cfg.mean, cfg.std)))
        val_dataset = BCDDataset(root_images, df_val, transform=transform_albumentations(get_val_tr(cfg.test_input_size, cfg.mean, cfg.std)))

        #criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(len(df_train_neg)/len(df_train_pos)))
        criterion = torch.nn.BCEWithLogitsLoss()

        trainer = CVTrainer(cfg,
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
