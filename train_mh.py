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
    parser.add_argument("--file", action="store_true")
    args = parser.parse_args()

    print("Loading ",args.cfg)
    cfg = get_config("config/default_mh.yaml")
    cfg_custom = get_config(args.cfg)
    cfg.update(cfg_custom)

    output_folder = get_output_folder(cfg, root="outputs")
    os.makedirs(output_folder, exist_ok=True)

    if args.file:
        stdout = open(os.path.join(output_folder , "stdout.log"), "w")
        stderr = open(os.path.join(output_folder , "stederr.log"), "w")
        sys.stdout = stdout
        sys.stderr = stderr

    seed_all(cfg.random_state)
    df = pd.read_csv(os.path.join("data", "train_5fold.csv"))
    df = df_preprocess(df, cfg.preprocess_softlabel)

    print(df.head())

    for fold in cfg.folds:
        df_train = df[df["fold"] != fold]
        df_val = df[df["fold"] == fold]
        print("-- Fold selected --")
        
        # Undersampling
        if cfg.max_negative_examples is not None:
            if cfg.preprocess_softlabel == False:
                print("preprocess_softlabel is True,max_negative_examples can't be applied! ")
            else:
                df_train_neg = df_train[df_train[cfg.target] == 0].sample(cfg.max_negative_examples)
                df_train_pos = df_train[df_train[cfg.target] == 1]
                df_train = pd.concat([df_train_neg, df_train_pos])
                print("Negative undersampled --")

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
        
        for i, col in enumerate(cfg.aux_cols_name):
            # n = number of col values
            n = len(df[col].unique())
            if n == 2:
                n = 1
            heads_num += [n]

            criterion += [
                factory_loss(cfg.aux_cols_type[i],
                            cfg.aux_cols_balance[i],
                            cfg.loss_aux.unbalance_perc[i],
                            df_train[col])
            ]

        backbone = factory_model(cfg.model_type,
                                 cfg.model_name,
                                 in_chans=cfg.in_chans,
                                 n_hidden=cfg.n_hidden,
                                 drop_rate_back=cfg.drop_rate_back,
                                 pretrained=cfg.pretrained
                                 )
        
        if cfg.model_ckpt is not None:
            state_dict = torch.load(cfg.model_ckpt, map_location="cpu")

            print("ckpt state dict")
            for i,k in enumerate(state_dict.keys()):
                print(k)
                if i>=5: break
            
            print("model state dict")
            model_sd = backbone.state_dict()
            for i,k in enumerate(model_sd.keys()):
                print(k)
                if i>=5: break

            print(load_state_dict_improved(state_dict, backbone, replace_str="backbone."))
        
        if cfg.freeze:
            for p in backbone.parameters():
                p.requires_grad = False
        
        model = MultiHead(backbone, heads_num=heads_num, drop_rate_mh=cfg.drop_rate_mh)

        train_dataset = BCDDataset(root_images,
                                   df_train,
                                   aux_cols=cfg.aux_cols_name,
                                   target=cfg.target,
                                   in_chans=cfg.in_chans,
                                   breast_crop=cfg.breast_crop,
                                   transform=transform_albumentations(get_train_tr(cfg.input_size, cfg.severity, cfg.mean, cfg.std)))
                                   
        val_dataset = BCDDataset(root_images,
                                 df_val,
                                 aux_cols=cfg.aux_cols_name,
                                 target=cfg.target,
                                 in_chans=cfg.in_chans,
                                 breast_crop=cfg.breast_crop,
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

        if args.file:
            stdout.close()
            stderr.close()
