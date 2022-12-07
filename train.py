import torch
import os
import pandas as pd
from src import *

config_file = "config/default.yaml"

root = os.path.join(os.environ["DATASET_ROOT"], "bcd2022")
root_images = os.path.join(root, "images_512")


if __name__ == "__main__":
    cfg = get_config(config_file)
    seed_all(cfg.random_state)

    model = factory_model(cfg.model_type, cfg.model_name, num_classes=1, drop_rate=cfg.drop_rate, hidden=cfg.hidden)
    
    if cfg.model_ckpt is not None:
        print(model.load_state_dict(torch.load(cfg.model_ckpt)))
    
    df_train = pd.read_csv(os.path.join(root, "train_split.csv"))
    df_val = pd.read_csv(os.path.join(root, "val_split.csv"))

    df_train = df_train[(df_train["difficult_negative_case"] == False)]
    df_val = df_val[(df_val["difficult_negative_case"] == False)]

    df_train_neg = df_train[df_train["cancer"] == 0].sample(10000)
    df_train_pos = df_train[df_train["cancer"] == 1]
    df_train = pd.concat([df_train_neg, df_train_pos])

    print(df_train["cancer"].value_counts())
    print(df_val["cancer"].value_counts())

    train_dataset = BCDDataset(root_images, df_train, transform=transform_albumentations(get_train_tr(cfg.input_size, cfg.severity)))
    val_dataset = BCDDataset(root_images, df_val, transform=transform_albumentations(get_val_tr(cfg.input_size)))

    #criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(len(df_train_neg)/len(df_train_pos)))
    criterion = torch.nn.BCEWithLogitsLoss()

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
    trainer.train()
