import torch
import os
import pandas as pd
import argparse
from src import *
import sys

root = os.path.join(os.environ["DATASET_ROOT"], "bcd2022")
root_images = os.path.join(root, "images_1024")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config/default_mh.yaml", type=str)
    args = parser.parse_args()

    print("Loading ",args.cfg)
    cfg = get_config(args.cfg)

    output_folder = get_output_folder(cfg, root="outputs")
    os.makedirs(output_folder, exist_ok=True)

    stdout = open(os.path.join(output_folder , "stdout.log"), "w")
    stderr = open(os.path.join(output_folder , "stederr.log"), "w")
    sys.stdout = stdout
    sys.stderr = stderr

    seed_all(cfg.random_state)
    df = pd.read_csv(os.path.join("data", "train_5fold.csv"))
    df = df_preprocess(df)

    print(df.head())

    for fold in cfg.folds:
        df_train = df[df["fold"] != fold]
        df_val = df[df["fold"] == fold]
        print("-- Fold selected --")
        
        # Undersampling
        if cfg.max_negative_examples is not None:
            df_train_neg = df_train[df_train[cfg.target] == 0].sample(cfg.max_negative_examples)
            df_train_pos = df_train[df_train[cfg.target] == 1]
            df_train = pd.concat([df_train_neg, df_train_pos])
            print("-- Fold undersampled --")

        print(cfg.target, df_train[cfg.target].value_counts())
        print(cfg.target, df_val[cfg.target].value_counts())
        print("------------------------------------")

        # Target
        heads_num = [1]
        criterion = [
            factory_loss(cfg.loss_target.loss_type,
                        cfg.loss_target.unbalance,
                        cfg.loss_target.unbalance_perc,
                        df_train[cfg.target])
        ]
        
        for i, col in enumerate(cfg.aux_cols):
            # n = number of col values
            n = len(df[col].unique())
            if n == 2:
                n = 1
            heads_num += [n]

            criterion += [
                factory_loss(cfg.loss_aux.loss_type[i],
                            cfg.loss_aux.unbalance[i],
                            cfg.loss_aux.unbalance_perc[i],
                            df_train[col])
            ]

        backbone = factory_model(cfg.model_type,
                                 cfg.model_name,
                                 in_chans=cfg.in_chans,
                                 n_hidden=cfg.n_hidden,
                                 drop_rate_back=cfg.drop_rate_back
                                 )
    
        model = MultiHead(backbone, heads_num=heads_num, drop_rate_mh=cfg.drop_rate_mh)
    
        if cfg.model_ckpt is not None:
            print(model.load_state_dict(torch.load(cfg.model_ckpt)))

        train_dataset = BCDDataset(root_images,
                                   df_train,
                                   aux_cols=cfg.aux_cols,
                                   target=cfg.target,
                                   in_chans=cfg.in_chans,
                                   transform=transform_albumentations(get_train_tr(cfg.input_size, cfg.severity, cfg.mean, cfg.std)))
                                   
        val_dataset = BCDDataset(root_images,
                                 df_val,
                                 aux_cols=cfg.aux_cols,
                                 target=cfg.target,
                                 in_chans=cfg.in_chans,
                                 transform=transform_albumentations(get_val_tr(cfg.test_input_size, cfg.mean, cfg.std)))

        trainer = CVMHTrainer(cfg,
                            model,
                            train_dataset,
                            val_dataset,
                            df_val,
                            criterion,
                            maximize=True,
                            show_dataset=False,
                            output_folder=output_folder,
                            imb_callback=lambda dataset, idx: dataset.target[idx],
                            save_pth=True)
        trainer.train()

        stdout.close()
        stderr.close()
