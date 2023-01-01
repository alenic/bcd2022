
import pandas as pd
import matplotlib.pyplot as plt
from src import *
import torchvision.transforms as T
import time

root = os.path.join(os.environ["DATASET_ROOT"], "bcd2022")

df = pd.read_csv("data/train_5fold_aug.csv")

option = input("Profile (p/P) or Visualize (v/V): ")


tr = transform_albumentations(get_train_tr(input_size=(300, 512), severity=4))
tr_val = transform_albumentations(get_val_tr(input_size=(300, 512)))

for dataset_type in [(BCDDatasetNPZ, "alenic_train_images_1024"), (BCDDataset, "images_1024")]:
    root_images = os.path.join(root, dataset_type[1])

    dataset_train = dataset_type[0](root_images, df, transform=tr, in_chans=3)
    dataset_val = dataset_type[0](root_images, df, transform=tr_val, in_chans=3)
    
    if option.lower() == "v":
        
        for i in range(5):
            img, label = dataset_train[i]
            img_val, label = dataset_val[i]
            img_tr_train = T.ToPILImage()(img)
            img_tr_val = T.ToPILImage()(img_val)

            img_tr_train_np = np.array(img_tr_train)
            img_tr_val_np = np.array(img_tr_val)
            print("Train", img_tr_train_np.shape, img_tr_train_np.min(), img_tr_train_np.max())
            print("Val", img_tr_val_np.shape, img_tr_val_np.min(), img_tr_val_np.max())

            fig, ax = plt.subplots(1, 2, figsize=(10,5))
            ax[0].imshow(img_tr_train_np, vmin=0, vmax=255)
            ax[1].imshow(img_tr_val, vmin=0, vmax=255)
    else:
        t = time.time()
        for i in range(100):
            img, label = dataset_train[i]
        
        print("Time: ", time.time()-t)
    
    plt.show()