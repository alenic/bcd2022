import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.transforms as T
import cv2


def transform_albumentations(tr):
    return lambda x: tr(image=x)["image"]


def get_train_tr(input_size, severity=2):
    tr = [A.Resize(input_size, input_size)]

    #rot = [2, 5, 10, 15, 20, 25][severity]
    #tr += [A.ShiftScaleRotate(p=1, rotate_limit=rot, border_mode=cv2.BORDER_CONSTANT)]
    
    if severity >= 3:
        tr += [A.Perspective(p=1)]
    
    tr += [A.HorizontalFlip(p=0.5)]

    bl = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25][severity]
    cl = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25][severity]
    tr += [A.RandomBrightnessContrast(brightness_limit=bl, contrast_limit=cl, p=0.5)]
    
    mh = [2, 4, 6, 8, 10, 12][severity]
    tr += [A.CoarseDropout(p=1, max_holes=mh,
           min_height=input_size//16, max_height=input_size//15,
           min_width=input_size//16,  max_width=input_size//15)]


    # Resize and tensor
    tr += [A.ToFloat(), ToTensorV2()]

    return A.Compose(tr)

# Val
def get_val_tr(input_size):
    return A.Compose([A.Resize(input_size, input_size), A.ToFloat(), ToTensorV2()])


def crop_breast(img: np.array):
    im_mean = img.mean()*0.5
    img_mask= (img > im_mean).astype(np.uint8)
    img_mask = cv2.UMat(img_mask)
    img_mask = cv2.UMat.get(cv2.erode(img_mask, kernel=np.ones((8, 8), np.uint8)))
    
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
    root_images = os.path.join(os.environ["DATASET_ROOT"], "bcd2022", "images_512")
    files = os.listdir(root_images)
    
    for f in files:
        image_file = os.path.join(root_images, f)
        img = cv2.imread(image_file, 0)

        img_crop = crop_breast(img)
        tr = transform_albumentations(get_train_tr(severity=2, input_size=512))
        img_crop_t = tr(img_crop)

        img_crop = T.ToPILImage()(img_crop_t)

        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        ax[0].imshow(img)
        ax[1].imshow(img_crop)
        plt.show()

