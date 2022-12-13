import torch
import torchvision.transforms as T
import os
from src import *
import glob
import pandas as pd

fold = 0
model_ckpt = "outputs/2022-12-08_23-32-18_efficientnet_b0/E0008_0.15743440233236153.pth"
config_file = os.path.join(os.path.dirname(model_ckpt), "config.yaml")

# ===========================================================
root = os.path.join(os.environ["DATASET_ROOT"], "bcd2022")
root_images = os.path.join(root, "images_512")

if __name__ == "__main__":
    cfg = get_config(config_file)
    df = pd.read_csv(os.path.join("data", "train_5fold.csv"))

    val_tr = transform_albumentations(get_val_tr(cfg.test_input_size))

    model = factory_model(cfg.model_type, cfg.model_name, num_classes=1, drop_rate=cfg.drop_rate, hidden=cfg.hidden)
    print(model.load_state_dict(torch.load(model_ckpt)))
 
    df_val = df[df["fold"] == fold]

    print(df_val["cancer"].value_counts())

    val_dataset = BCDDataset(root_images, df_val, transform=val_tr)

    eval = CVEval(cfg,
                  df_val,
                  model,
                  val_dataset)

    f1score, pf1_mean, pf1_max, pf1_majority = eval.eval_metrics()

    score = max(pf1_mean, pf1_max, pf1_majority)

    print(f"Val f1score {f1score}")

    print(f"Val pf1_mean {pf1_mean}")

    print(f"Val pf1_max {pf1_max}")

    print(f"Val pf1_majority {pf1_majority}")
