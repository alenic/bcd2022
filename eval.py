import torch
import torchvision.transforms as T
import os
from src import *
from tqdm import tqdm
import pandas as pd

import fiftyone as fo
import fiftyone.zoo as foz

fold = 0
tta = False
model_ckpt = "outputs/2022-12-09_16-30-57_tf_efficientnetv2_b0/E0011_0.13076923076923078.pth"
config_file = os.path.join(os.path.dirname(model_ckpt), "config.yaml")

# ===========================================================
root = os.path.join(os.environ["DATASET_ROOT"], "bcd2022")
root_images = os.path.join(root, "images_768")

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

    f1score, pf1_mean, pf1_max, pf1_majority, best_thr, y_prob, y_pred, y_true = eval.eval_metrics(tta=tta, return_y=True)
    print("f1score", f1score)
    print("pf1_mean", pf1_mean)
    print("pf1_max", pf1_max)
    print("pf1_majority", pf1_majority)



    # Fifty One
    dataset = fo.Dataset("eval-result", overwrite=True)
    files = [os.path.join(root_images, f"{p}_{im}.png") for p, im in zip(df_val["patient_id"].values, df_val["image_id"].values)]

    samples = []
    for i, f in enumerate(tqdm(files)):
        sample = fo.Sample(filepath=f)
        sample["ground_truth"] = fo.Classification(label=str(int(y_true[i])))
        sample["predicted"] = fo.Classification(label=str(int(y_pred[i])), confidence=y_prob[i])
        #if y_true[i] != y_pred[i]:
        #    sample.tags.append("mistake")
        samples.append(sample)

    dataset.add_samples(samples)
    dataset.save()

    session = fo.launch_app(dataset, port=5152)
    session.wait()