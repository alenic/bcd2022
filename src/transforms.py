import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.transforms as T
import cv2
from functools import partial

class CustomBrigthnessContrast(A.ImageOnlyTransform):
    def __init__(self,
                 always_apply: bool = False,
                 p: float = 0.5,
                 brightness_limit=0.01,
                 contrast_limit=0.01,
                 min_value=0,
                 max_value=255):
        super(CustomBrigthnessContrast, self).__init__(always_apply, p)
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.min_value = min_value
        self.max_value = max_value
        
    def apply(self, img, **params):
        ct = 1+(2*np.random.rand()-1)*self.brightness_limit
        br = (2*np.random.rand()-1)*self.contrast_limit
        img = ct * img
        img = img + br*img.max()
        img = np.clip(img, self.min_value, self.max_value)
        return img

class PutCalcification(A.ImageOnlyTransform):
    def __init__(self,
                 always_apply: bool = False,
                 p: float = 0.5,
                 n=1,
                 min_value=0,
                 max_value=255):
        super(PutCalcification, self).__init__(always_apply, p)
        self.min_value = min_value
        self.max_value = max_value
        self.n = n
        
    def apply(self, img, **params):
        imax = img.max()
        imin = img.min()

        h,w = img.shape[:2]

        n = int(self.n*np.random.rand()) + 1
        for i in range(n):
            r = int((np.random.rand()*0.08 + 0.01)*max(h,w))
            pos_x = int(np.random.rand()*(w-2*r) + r)
            pos_y = int(np.random.rand()*(h-2*r) + r)
            axes_x = int((np.random.rand()*0.9+0.1)*r)
            axes_y = int((np.random.rand()*0.9+0.1)*r)
            angle = int(np.random.rand()*180)
            color = imin + (imax-imin)*(0.8 + np.random.rand()*0.2)
            cv2.ellipse(img,
                        (pos_x, pos_y),
                        (axes_x, axes_y),
                        angle=angle,
                        startAngle=0,
                        endAngle=360, 
                        color=color,
                        thickness=-1)


        return img

def transform_albumentations_pickle(tr, x):
    return tr(image=x)["image"]

def transform_albumentations(tr):
    return partial(transform_albumentations_pickle, tr)

def get_train_tr(input_size, severity=2, mean=0, std=1):
    tr = []

    rot = [0, 2, 4, 7, 10, 12, 15][severity]
    scale = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25][severity]
    if rot > 0:
        tr += [A.ShiftScaleRotate(p=1, rotate_limit=0, scale_limit=(0, scale))]
        #tr += [A.ElasticTransform(1, 10, rot, border_mode=cv2.BORDER_CONSTANT, p=0.5)]

    tr += [A.Resize(width=input_size[0], height=input_size[1], interpolation=cv2.INTER_LINEAR)]

    bl = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12][severity]
    cl = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12][severity]
    tr += [CustomBrigthnessContrast(brightness_limit=bl, contrast_limit=cl, p=1, min_value=0, max_value=255)]

    tr += [A.HorizontalFlip(p=0.5)]

    # mh = [0, 5, 8, 12, 15, 18, 22][severity]
    # if mh > 0:
    #     tr += [A.CoarseDropout(p=0.2, max_holes=mh,
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


def npz_preprocessing3(image_in, low_q=(0.1,0.5), hig_h=(0.95,1)):
    image = image_in.copy()
    image_high = np.empty_like(image)
    image_low = np.empty_like(image)

    imin = image.min()
    imax = image.max()

    data = image.flatten()

    q1 = np.quantile(data, q=low_q[0])
    q2 = np.quantile(data, q=low_q[1])
    image_low[image<q1]=q1
    image_low[image>q2]=q2

    q1 = np.quantile(data, q=hig_h[0])
    q2 = np.quantile(data, q=hig_h[1])
    image_high[image<q1]=q1
    image_high[image>q2]=q2

    image3 = np.zeros(list(image.shape)+[3], dtype=image.dtype)
    image3[:, :, 0] = image_low
    image3[:, :, 1] = image
    image3[:, :, 2] = image_high
    image3 = (image3-imin).astype(np.float32)/(imax-imin)

    return (image3*255).astype(np.uint8)


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
    img_mask = cv2.erode(img_mask, (8,8))
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
    root_images = os.path.join(os.environ["DATASET_ROOT"], "bcd2022", "alenic_train_images_1024")
    files = os.listdir(root_images)
    
    tr_calc = [PutCalcification(p=1, n=100)]
    tr_calc = A.Compose(tr_calc)

    for f in files:
        image_file = os.path.join(root_images, f)
        img = np.load(image_file)["data"]
        img = npz_preprocessing(img)
        img_t = tr_calc(image=img)["image"]

        fig, ax = plt.subplots(1, 1, figsize=(10,10))
        ax.imshow(img_t, vmin=img.min(), vmax=img.max())
        plt.show()

    exit()

    for f in files:
        image_file = os.path.join(root_images, f)
        img = np.load(image_file)["data"]
        img_crop = crop_breast(img)
        img = npz_preprocessing(img)
        img_crop = npz_preprocessing(img_crop)
        tr = transform_albumentations(get_train_tr(severity=6, input_size=(256, 512)))
        img_crop_t = tr(img_crop)
        print(img_crop_t.min(), img_crop_t.max())
        img_crop = T.ToPILImage()(img_crop_t)

        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        ax[0].imshow(img, vmin=img.min(), vmax=img.max())
        ax[1].imshow(img_crop, vmin=0, vmax=255)
        plt.show()

