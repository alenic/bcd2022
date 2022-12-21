import torch
import torchvision.transforms as T
import os
from src import *
from tqdm import tqdm
import pandas as pd
from PIL import Image
import fiftyone as fo
import fiftyone.zoo as foz

fold = 0
tta = False
model_ckpt = "outputs/2022-12-09_16-30-57_tf_efficientnetv2_b0/E0011_0.13076923076923078.pth"
best_thr = 0.69
target_layer = "back.bn2.act"

config_file = os.path.join(os.path.dirname(model_ckpt), "config.yaml")


# ===========================================================
root = os.path.join(os.environ["DATASET_ROOT"], "bcd2022")
root_images = os.path.join(root, "images_768")

if __name__ == "__main__":
    cfg = get_config(config_file)
    cfg.test_batch_size = 1

    df = pd.read_csv(os.path.join("data", "train_5fold.csv"))

    val_tr = transform_albumentations(get_val_tr(cfg.test_input_size))

    model = factory_model(cfg.model_type, cfg.model_name, num_classes=1, drop_rate=cfg.drop_rate, hidden=cfg.hidden)
    print(model.load_state_dict(torch.load(model_ckpt)))
 
    df_val = df[(df["fold"] != fold) & (df["cancer"]==1)]

    print(df_val["cancer"].value_counts())

    val_dataset = BCDDataset(root_images, df_val, transform=val_tr, return_path=True)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=cfg.test_batch_size,
        num_workers=cfg.num_workers,
        shuffle=False
    )
    model.eval()
    model.to(cfg.device)
    
    y_true_list = []
    y_prob_list = []

    grad_cam = GradCAM(model, candidate_layers=[target_layer], device=cfg.device)

    for iter, (image, label, path) in enumerate(tqdm(val_data_loader)):
        image = image.to(cfg.device)
        with torch.set_grad_enabled(True):
            scores = grad_cam.forward(image)

            grad_cam.backward(label)
            grad_cam_regions = grad_cam.generate(target_layer=target_layer)

            for i in range(len(path)):
                raw_image = np.array(Image.open(path[i]).convert("RGB"))

                mixed_image = grad_cam.image_heatmap_mix(
                    grad_cam_regions[i, 0].cpu().numpy(), raw_image
                )
                mixed_image_pil = Image.fromarray(mixed_image)
                plt.imshow(mixed_image_pil)
                print(label, scores)
                plt.show()

            y_true_list += list(label.cpu().numpy())
            y_prob = scores.detach().cpu().numpy().flatten()
            y_prob_list += list(y_prob)


    y_true = np.array(y_true_list)
    y_prob = np.array(y_prob_list)


    f1score, best_thr = optimize_metric(f1, y_true, y_prob)
    print("f1_score", f1score, "best thr ", best_thr)
    y_pred = (y_prob>=best_thr).astype(int)

    df = pd.DataFrame()
    df["pid"] = df_val["patient_id"].astype(str) + "_" + df_val["laterality"]
    df["y_pred"] = y_pred
    df["y_true"] = y_true
    df["y_prob"] = y_prob

    df_true = df.groupby("pid")["y_true"].apply(lambda x: x.sum()>0).astype(int)
    # pf1 by mean
    df_pred = df.groupby("pid")["y_prob"].mean()
    pf1_mean = pf1(df_true.values, df_pred.values)
    # pf1 by max
    df_pred = df.groupby("pid")["y_pred"].max()
    pf1_max = pf1(df_true.values, df_pred.values)
    # pf1 by majority
    df_pred = df.groupby("pid")["y_pred"].apply(lambda x: x.sum() >= len(x)*0.5).astype(int)
    pf1_majority = pf1(df_true.values, df_pred.values)

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