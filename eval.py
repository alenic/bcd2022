import torch
import torchvision.transforms as T
import os
from src import *
import glob
import pandas as pd

model_path = "outputs/2022-12-06_20-51-35_efficientnet_b0"

config_file = os.path.join(model_path, "config.yaml")
root = os.path.join(os.environ["DATASET_ROOT"], "Breast-Cancer-Detection-2022")
root_images = os.path.join(os.environ["DATASET_ROOT"], "Breast-Cancer-Detection-2022", "train_images_processed_512")

if __name__ == "__main__":
    cfg, _ = get_config(config_file)
    val_tr = T.Compose([    T.Resize((cfg.test_input_size, cfg.test_input_size)),
                            T.ToTensor(),
                            ])
    
    pths = glob.glob(os.path.join(model_path,"*.pth"))
    cfg.model_ckpt = pths[0]

    model = factory_model(cfg.model_type, cfg.model_name, num_classes=2, drop_rate=cfg.drop_rate, hidden=cfg.hidden)
    if cfg.model_ckpt is not None:
        print(model.load_state_dict(torch.load(cfg.model_ckpt)))
 

    df_val = pd.read_csv(os.path.join(root, "val_split.csv"))
    df_val = df_val[(df_val["difficult_negative_case"] == False)]

    print(df_val["cancer"].value_counts())

    val_dataset = BCDDataset(root_images, df_val, transform=val_tr)

    eval = CVEval(cfg,
                  model,
                  val_dataset,
                  [pf1, f1])

    y_pred, y_true, y_prob = eval.eval(tta=True)
    score, metric_name = pf1(y_true, y_pred, y_prob)
    print(f"Start Val {metric_name} {score}")
    score, metric_name = f1(y_true, y_pred, y_prob)
    print(f"Start Val {metric_name} {score}")
