import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.transforms as T
import cv2
import torch


class CustomBrigthnessContrast(A.ImageOnlyTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5, brightness_limit=0.01, contrast_limit=0.01):
        super(CustomBrigthnessContrast, self).__init__(always_apply, p)
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
    
    def apply(self, img, **params):
        ct = 1+(2*np.random.rand()-1)*self.brightness_limit
        br = (2*np.random.rand()-1)*self.contrast_limit
        img = ct * img
        img = img + br*img.max()
        img = np.clip(img, 0, 255)

        return img

def transform_albumentations(tr):
    return lambda x: tr(image=x)["image"]


def get_train_tr(input_size, severity=2, mean=0, std=1):
    tr = []

    rot = [0, 2, 4, 7, 10, 12, 15][severity]
    if rot > 0:
        tr += [A.ShiftScaleRotate(p=0.5, rotate_limit=rot, scale_limit=(0, 0.2), border_mode=cv2.BORDER_CONSTANT)]
        #tr += [A.ElasticTransform(1, 10, rot, border_mode=cv2.BORDER_CONSTANT, p=0.5)]

    tr += [A.Resize(width=input_size[0], height=input_size[1], interpolation=cv2.INTER_LINEAR)]

    bl = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12][severity]
    cl = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12][severity]
    tr += [CustomBrigthnessContrast(brightness_limit=bl, contrast_limit=cl, p=0.5)]

    tr += [A.HorizontalFlip(p=0.5)]

    # mh = [0, 5, 8, 12, 15, 18, 22][severity]
    # if mh > 0:
    #     tr += [A.CoarseDropout(p=0.5, max_holes=mh,
    #         min_height=input_size[1]//24, max_height=input_size[1]//12,
    #         min_width=input_size[1]//24,  max_width=input_size[1]//12)]

    tr += [A.Normalize(mean=mean, std=std), ToTensorV2()]

    return A.Compose(tr)

# Val
def get_val_tr(input_size, mean=0, std=1):
    return A.Compose([A.Resize(height=input_size[1], width=input_size[0], interpolation=cv2.INTER_LINEAR),
                      A.Normalize(mean=mean, std=std),
                      ToTensorV2()
                      ])

def npz_preprocessing(image_in, quant=None, window=None):
    image = image_in.copy()
    # normalize
    if window is not None:
        c, w = window
        imin = c - w//2
        imax = c + w//2
        image[image<imin]=imin
        image[image>imax]=imax
    elif quant is not None:
        data = image.flatten()
        q1 = np.quantile(data, q=quant[0])
        q2 = np.quantile(data, q=quant[1])
        imin = q1
        imax = q2
        image[image<imin]=imin
        image[image>imax]=imax
    else:
        imin = image.min()
        imax = image.max()

    image = (image-imin).astype(np.float32)/(imax-imin)

    return (image*255).astype(np.uint8)


def crop_breast(img: np.array):
    im_mean = img.mean()*0.5
    img_mask= (img > im_mean).astype(np.uint8)
    contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        max_contour = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(max_contour)
        img = img[:, x:x+w]

    return img

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    root = os.environ["DATASET_ROOT"]
    root_images = os.path.join(os.environ["DATASET_ROOT"], "bcd2022", "alenic_train_images_1024")
    files = os.listdir(root_images)
    
    for f in files:
        image_file = os.path.join(root_images, f)
        img = np.load(image_file)["data"]
        img = npz_preprocessing(img, "MONOCHROME2")
        img_crop=img
        try:
            img_crop = crop_breast(img)
        except:
            print("Error in image", image_file)
        
        tr = transform_albumentations(get_train_tr(severity=3, input_size=(256, 512)))
        img_crop_t = tr(img_crop)
        print(img_crop_t.min(), img_crop_t.max())
        img_crop = T.ToPILImage()(img_crop_t)

        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        ax[0].imshow(img, vmin=0, vmax=1)
        ax[1].imshow(img_crop, vmin=0, vmax=255)
        plt.show()

