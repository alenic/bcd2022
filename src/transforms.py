import numpy as np
from PIL import Image
import cv2

def crop_breast(img: Image):
    img = np.array(img)
    im_mean = img.mean()*0.5
    img_mask= (img > im_mean).astype(np.uint8)
    img_mask = cv2.UMat(img_mask)
    img_mask = cv2.UMat.get(cv2.erode(img_mask, kernel=np.ones((8, 8), np.uint8)))
    
    contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        max_contour = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(max_contour)
        img = Image.fromarray(img).crop([x,y,x+w,y+h])

    return img



if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    root = os.environ["DATASET_ROOT"]
    root_images = os.path.join(os.environ["DATASET_ROOT"], "Breast-Cancer-Detection-2022", "train_images_processed_512")
    patid = os.listdir(root_images)
    
    for p in patid:
        pat_folder = os.path.join(root_images, p)
        files = os.listdir(pat_folder)
        for f in files:
            image_file = os.path.join(pat_folder, f)
            img = Image.open(image_file).convert("L")

            img_crop = crop_breast(img)


            fig, ax = plt.subplots(1, 2, figsize=(10,5))
            ax[0].imshow(img)
            ax[1].imshow(img_crop)
            plt.show()

