import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.transforms as T
import cv2
import torch


'''
class CustomToTensor(A.ImageOnlyTransform):
    # Input:  ndarray image from 0 to 255
    def __init__(self, always_apply: bool = False, p: float = 1, mean=None, std=None, max_pixel_value=255):
        super(CustomToTensor, self).__init__(always_apply, p)
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value
    
    def apply(self, img, **params):
        img = img.astype(np.float32) / self.max_pixel_value 
        if self.mean is not None:
            if self.std is not None:
                img = img - self.mean
                img = img / self.std
        
        return torch.from_numpy(np.expand_dims(img, axis=0))
'''

'''
class CustomResizePad(A.ImageOnlyTransform):
    # Input:  ndarray image from 0 to 255
    def __init__(self, size, always_apply: bool = False, p: float = 1):
        super(CustomResizePad, self).__init__(always_apply, p)
        self.size = size
    
    def apply(self, img, **params):
        h, w = img.shape
        
        max_wh = max(w,h)
        p_left = (max_wh - w) // 2
        p_top = (max_wh - h) // 2
        p_right = max_wh - (w+p_left)
        p_bottom = max_wh - (h+p_top)
        padding = (p_left, p_top, p_right, p_bottom)
        return T.functional.pad(image, padding, 0, 'constant')

        if self.mean is not None:
            if self.std is not None:
                img = img - self.mean
                img = img / self.std
        
        return torch.from_numpy(np.expand_dims(img, axis=0))
'''

def transform_albumentations(tr):
    return lambda x: tr(image=x)["image"]


def get_train_tr(input_size, severity=2, mean=0, std=1):
    tr = [   A.PadIfNeeded(min_height=input_size[1], min_width=input_size[0])]
    tr += [A.Resize(width=input_size[0], height=input_size[1], interpolation=cv2.INTER_LINEAR)]

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
           min_height=input_size[0]//16, max_height=input_size[0]//15,
           min_width=input_size[1]//16,  max_width=input_size[1]//15)]


    # Resize and tensor
    #tr += [CustomToTensor(mean=mean, std=std, max_pixel_value=255)]
    tr += [A.Normalize(mean=mean, std=std), ToTensorV2()]

    return A.Compose(tr)

# Val
def get_val_tr(input_size, mean=0, std=1):
    return A.Compose([A.Resize(height=input_size[1], width=input_size[0], interpolation=cv2.INTER_LINEAR),
                      A.Normalize(mean=mean, std=std), ToTensorV2()
                      ])


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
    root_images = os.path.join(os.environ["DATASET_ROOT"], "bcd2022", "images_1024")
    files = os.listdir(root_images)
    
    for f in files:
        image_file = os.path.join(root_images, f)
        img = cv2.imread(image_file, 0)

        try:
            img_crop = crop_breast(img)
        except:
            print("Error in image", image_file)
        
        tr = transform_albumentations(get_train_tr(severity=2, input_size=(256,512)))
        img_crop_t = tr(img_crop)

        img_crop = T.ToPILImage()(img_crop_t)

        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        ax[0].imshow(img)
        ax[1].imshow(img_crop)
        plt.show()

