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
model_ckpt = "outputs/2023-01-02_14-03-04_tf_efficientnetv2_s_in21ft1k/best.pth"
config_file = os.path.join(os.path.dirname(model_ckpt), "config.yaml")
SAVE_FIFTYONE = False
# ===========================================================
SAVE_FIFTYONE = False
root = os.path.join(os.environ["DATASET_ROOT"], "bcd2022")
root_images = os.path.join(root, "alenic_train_images_1024")

if __name__ == "__main__":
    cfg = get_config(config_file)
    df = pd.read_csv(os.path.join("data", "train_5fold_aug.csv"))
    df = df_preprocess(df, cfg.preprocess_softlabel)

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

    val_tr = transform_albumentations(get_val_tr(cfg.test_input_size, cfg.mean, cfg.std))
    val_dataset = BCDDatasetNPZ(root_images,
                                df_val,
                                aux_cols=cfg.aux_cols_name,
                                target=cfg.target,
                                in_chans=cfg.in_chans,
                                breast_crop=cfg.breast_crop,
                                transform=val_tr,
                                return_breast_id=True)

    evaluator = CVMHGroupedEval(cfg,
                                df_val,
                                model,
                                val_dataset)

    y_true_dict, y_out_dict = evaluator.eval(tta=cfg.tta)

    criterion = factory_loss(cfg.loss_target.loss_type,
                            cfg.loss_target.unbalance,
                            cfg.loss_target.unbalance_perc,
                            df_val["target"])
    
    val_loss = criterion(torch.tensor(y_out_dict[cfg.target].astype(np.float32)),
                         torch.tensor(y_true_dict[cfg.target].astype(np.float32)),
                         torch.tensor(df_val["breast_id"].values.astype(int)))
    print("Val loss", val_loss)

    metrics = evaluator.eval_metrics(y_true_dict, y_out_dict)


    c = cfg.target
    for m in ["pf1_mean", "pf1_max", "tpf1_mean", "tpf1_mean_thr", "tpf1_max", "tpf1_max_thr"]:
        print(f"Val {c} -> {m} {metrics[m][c]}")
    
    if SAVE_FIFTYONE:
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