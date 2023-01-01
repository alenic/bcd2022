import torch
import os
import argparse
from src import *
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from sklearn.metrics import accuracy_score

random_state = 42
model_type = "timm"
model_name = "tf_efficientnetv2_s_in21ft1k"
drop_rate_back = 0.25
in_chans = 1
n_hidden = None
input_size = (224, 224)
mean = 0.5
std = 0.5
severity = 3
batch_size = 32
num_workers = 8
opt = "adam"
lr = 1e-3
n_epochs = 40
weight_decay=1e-8

device = "cuda:0"

def cv2_loader(path):
    img = cv2.imread(path, 0)
    return img

root = os.path.join(os.environ["DATASET_ROOT"], "fractaldb-60")

seed_all(random_state)

criterion = torch.nn.CrossEntropyLoss()

backbone = factory_model(model_type,
                         model_name,
                         in_chans=in_chans,
                         n_hidden=n_hidden,
                         drop_rate_back=drop_rate_back,
                         pretrained=False
                        )

model = MultiHead(backbone=backbone, heads_num=[60])

train_dataset = ImageFolder(os.path.join(root,"train"),
                            transform=transform_albumentations(get_train_tr(input_size, severity, mean, std)),
                            loader=cv2_loader
                            )
                            
val_dataset = ImageFolder(os.path.join(root,"val"),
                            transform=transform_albumentations(get_train_tr(input_size, severity, mean, std)),
                            loader=cv2_loader
                            )

train_data_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True
)

val_data_loader = torch.utils.data.DataLoader(
val_dataset, 
batch_size=batch_size,
num_workers=num_workers,
shuffle=False
)

n_iter_per_epoch = len(train_data_loader)
total_iter = n_iter_per_epoch * n_epochs
opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_iter, eta_min=0.01*lr)


out_file = open("pretraining_out/log.txt", "w")

model.to(device)
best_score = 0
for epoch in range(n_epochs):
    model.train()
    for iter, (image, labels) in enumerate(tqdm(train_data_loader)):
        opt.zero_grad()
        image = image.to(device)
        labels = labels.to(device)
        outputs = model(image)[0]
        loss = criterion(outputs, labels)
        if epoch == 0 and iter == 0:
            input()
        loss.backward()
        opt.step()
        lr_scheduler.step()

        if iter % int(0.1*n_iter_per_epoch) == 0:
            print("train loss: ", loss.item())

    model.eval()
    y_pred = []
    y_true = []
    for iter, (image, labels) in enumerate(tqdm(val_data_loader)):
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image)[0]
        y_pred += list(torch.argmax(outputs, 1).squeeze().cpu().numpy())
        y_true += list(labels)
    acc = accuracy_score(y_true, y_pred)
    if acc > best_score:
        best_score = acc
        os.makedirs("pretraining_out", exist_ok=True)
        torch.save(backbone.state_dict(), "pretraining_out/best_model.pth")

    print("Val acc", acc)
    out_file.write(f"epoch {epoch} {acc}\n")

out_file.close()