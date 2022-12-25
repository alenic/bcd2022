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
model_ckpt = "outputs/2022-12-22_00-28-43_tf_efficientnetv2_m_in21ft1k/E0002_pf1_max_thr0.9091_24.3056.pth"
config_file = os.path.join(os.path.dirname(model_ckpt), "config.yaml")

# ===========================================================
root = os.path.join(os.environ["DATASET_ROOT"], "bcd2022")
root_images = os.path.join(root, "images_1024")

if __name__ == "__main__":
    cfg = get_config(config_file)
    df = pd.read_csv(os.path.join("data", "train_5fold.csv"))

    val_tr = transform_albumentations(get_val_tr(cfg.test_input_size))

    backbone = factory_model(cfg.model_type,
                                cfg.model_name,
                                in_chans=cfg.in_chans,
                                n_hidden=cfg.n_hidden,
                                drop_rate_back=cfg.drop_rate_back
                                )

    model = MultiHead(backbone, heads_num=[1], drop_rate_mh=cfg.drop_rate_mh)
    if model_ckpt is not None:
        print(model.load_state_dict(torch.load(model_ckpt, map_location="cpu"),strict=False))

    df_val = df[df["fold"] == fold]
    df_val.index = np.arange(len(df_val))

    print(df_val["cancer"].value_counts())

    val_dataset = BCDDataset(root_images,
                             df_val,
                             aux_cols=cfg.aux_cols,
                             transform=val_tr)

    evaluator = CVMHEval(cfg,
                    df_val,
                    model,
                    val_dataset)

    y_true_dict, y_prob_dict = evaluator.eval(tta=cfg.tta)
    metrics, thresholds = evaluator.eval_metrics(y_true_dict, y_prob_dict)

    for c in cfg.aux_cols:
        for m in ["f1score", "pf1_mean", "pf1_max", "pf1_majority"]:
            print(f"Val {c} -> {m} {metrics[m][c]}")
        print(f"best_f1score_thr: {thresholds[c]}")
    
    plot_interactive_precision_recall_curve(metrics["precision"]["cancer"], metrics["recall"]["cancer"], metrics["pr_thr"]["cancer"])


    # Fifty One
    dataset = fo.Dataset("eval-result", overwrite=True)
    files = [os.path.join(root_images, f"{p}_{im}.png") for p, im in zip(df_val["patient_id"].values, df_val["image_id"].values)]

    y_pred = (y_prob_dict["cancer"] >= thresholds["cancer"]).astype(int)

    df_val["laterality"] = df["laterality"].astype(str)
    df_val["view"] = df["view"].astype(str)
    df_val["density"] = df["density"].astype(str)
    df_val["machine_id"] = df["machine_id"].astype(str)

    samples = []
    for i, f in enumerate(tqdm(files)):
        sample = fo.Sample(filepath=f)
        sample["ground_truth"] = fo.Classification(label=str(int(y_true_dict["cancer"][i])))
        sample["predicted"] = fo.Classification(label=str(int(y_pred[i])), confidence=y_prob_dict["cancer"][i])
        row = df_val.loc[i, :]
        sample["l_site_id"] = row.site_id
        sample["l_laterality"] = row.laterality
        sample["l_view"] = row["view"]
        sample["l_age"] = row.age
        sample["l_biopsy"] = row.biopsy
        sample["l_invasive"] = row.invasive
        sample["l_BIRADS"] = row.BIRADS
        sample["l_implant"] = row.implant
        sample["l_density"] = row.density
        sample["l_machine_id"] = str(row.machine_id)
        sample["l_difficult_negative_case"] = row.difficult_negative_case


        #if y_true[i] != y_pred[i]:
        #    sample.tags.append("mistake")
        samples.append(sample)

    dataset.add_samples(samples)
    dataset.save()

    session = fo.launch_app(dataset, port=5152)
    session.wait()