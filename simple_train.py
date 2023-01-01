import torch
import os
from src import *
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from sklearn.metrics import f1_score
import logging as log

log.basicConfig(filename="pretraining_out/stdout.txt",
                filemode='w',
                format='%(message)s',
                datefmt='%H:%M:%S',
                level=log.DEBUG)

random_state = 42
model_type = "timm"
model_name = "tf_efficientnetv2_s_in21ft1k"
drop_rate_back = 0.0
in_chans = 1
n_hidden = None
input_size = (160, 256)
mean = 0.5
std = 0.5
severity = 3
batch_size = 24
num_workers = 8
opt = "adam"
lr = 0.5e-3
n_epochs = 10
weight_decay=1e-8
fold = 0
breast_crop = True
pos_weight = 10.0

device = "cuda:0"

root = os.path.join(os.environ["DATASET_ROOT"], "bcd2022")
root_images = os.path.join(root, "images_1024")

seed_all(random_state)

criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight]).to(device))

backbone = factory_model(model_type,
                         model_name,
                         in_chans=in_chans,
                         n_hidden=n_hidden,
                         drop_rate_back=drop_rate_back,
                         pretrained=True
                        )

model = MultiHead(backbone=backbone, heads_num=[1])


df = pd.read_csv(os.path.join("data", "train_5fold_aug.csv"))
df = df_preprocess(df, False)

df_train = df[df["fold"] != fold]
df_val = df[df["fold"] == fold]

df_train_neg = df_train[df_train["target"] == 0].sample(10000)
df_train_pos = df_train[df_train["target"] == 1]
df_train = pd.concat([df_train_neg, df_train_pos])

train_dataset = BCDDataset(root_images,
                            df_train,
                            aux_cols=[],
                            target="cancer",
                            in_chans=in_chans,
                            breast_crop=breast_crop,
                            transform=transform_albumentations(get_train_tr(input_size, severity, mean, std)))
                            
val_dataset = BCDDataset(root_images,
                            df_val,
                            aux_cols=[],
                            target="cancer",
                            in_chans=in_chans,
                            breast_crop=breast_crop,
                            transform=transform_albumentations(get_val_tr(input_size, mean, std)))

train_data_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    pin_memory=True
)

val_data_loader = torch.utils.data.DataLoader(
val_dataset, 
batch_size=batch_size,
num_workers=num_workers,
shuffle=False,
pin_memory=True
)

n_iter_per_epoch = len(train_data_loader)
total_iter = n_iter_per_epoch * n_epochs
opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_iter, eta_min=0.01*lr)

model.to(device)
best_score = 0
best_epoch = 0
best_thr = 0
for epoch in range(n_epochs):
    model.train()
    for iter, (image, labels) in enumerate(tqdm(train_data_loader)):
        opt.zero_grad()
        image = image.to(device)
        labels = labels.to(device)
        outputs = model(image)[0]
        loss = criterion(outputs, labels.flatten())
        if epoch == 0 and iter == 0:
            input()
        loss.backward()
        opt.step()
        lr_scheduler.step()

        if iter % int(0.1*n_iter_per_epoch) == 0:
            log.info(f"train loss: {loss.item()}")

    model.eval()
    y_prob = []
    y_true = []
    for iter, (image, labels) in enumerate(tqdm(val_data_loader)):
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image)[0]
        y_prob += list(torch.sigmoid(outputs).cpu().numpy().flatten())
        y_true += list(labels.cpu().numpy().flatten())
    
    y_prob = np.array(y_prob)
    y_true = np.array(y_true)
    score, thr = optimize_metric(f1, y_true, y_prob)
    if score > best_score:
        best_score = score
        best_epoch = epoch
        best_thr = thr
        os.makedirs("pretraining_out", exist_ok=True)
        torch.save(backbone.state_dict(), "pretraining_out/backbone.pth")
        torch.save(model.state_dict(), "pretraining_out/model.pth")

    log.info(f"epoch {epoch}: score  {score}  thr {thr}")
    log.info(f"---> Best epoch {best_thr} with score {best_score}, with thr {best_thr}")
