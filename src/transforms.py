import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.transforms as T
import cv2
import torch


def transform_albumentations(tr):
    return lambda x: tr(image=x)["image"]


def get_train_tr(input_size, severity=2, mean=0, std=1):
    rot = [2, 4, 7, 10, 12, 15][severity]
    tr = [A.ShiftScaleRotate(p=1, rotate_limit=rot, border_mode=cv2.BORDER_CONSTANT)]

    tr += [A.Resize(width=input_size[0], height=input_size[1], interpolation=cv2.INTER_LINEAR)]

    tr += [A.HorizontalFlip(p=0.5)]

    bl = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1][severity]
    cl = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1][severity]
    tr += [A.RandomBrightnessContrast(brightness_limit=bl, contrast_limit=cl, p=1)]
    
    mh = [12, 14, 16, 18, 20, 22][severity]

    tr += [A.CoarseDropout(p=1, max_holes=mh,
           min_height=input_size[1]//32, max_height=input_size[1]//32,
           min_width=input_size[1]//32,  max_width=input_size[1]//32)]


    tr += [A.Normalize(mean=mean, std=std), ToTensorV2()]

    return A.Compose(tr)

# Val
def get_val_tr(input_size, mean=0, std=1):
    return A.Compose([A.Resize(height=input_size[1], width=input_size[0], interpolation=cv2.INTER_LINEAR),
                      A.Normalize(mean=mean, std=std),
                      ToTensorV2()
                      ])


def crop_breast(img: np.array):
    im_mean = img.mean()*0.5
    img_mask= (img > im_mean).astype(np.uint8)
    contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        max_contour = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(max_contour)
        img = img[y:y+h, x:x+w]

    return img

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    root = os.environ["DATASET_ROOT"]
    root_images = os.path.join(os.environ["DATASET_ROOT"], "bcd2022", "images_1024")
    files = os.listdir(root_images)
    
    for f in files:
        image_file = os.path.join(root_images, f)
        img = cv2.imread(image_file, 0)

        try:
            img_crop = crop_breast(img)
        except:
            print("Error in image", image_file)
        
        tr = transform_albumentations(get_train_tr(severity=3, input_size=(256, 512)))
        img_crop_t = tr(img_crop)
        print(img_crop_t.min(), img_crop_t.max())
        img_crop = T.ToPILImage()(img_crop_t)

        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        ax[0].imshow(img, vmin=0, vmax=255)
        ax[1].imshow(img_crop, vmin=0, vmax=255)
        plt.show()

