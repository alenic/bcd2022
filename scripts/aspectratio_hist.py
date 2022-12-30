from src import *
import os 
import matplotlib.pyplot as plt
import cv2
import tqdm

root = os.path.join(os.environ["DATASET_ROOT"], "bcd2022", "images_512")

files = os.listdir(root)
files = files[:10000]

width = []
height = []

for k, f in enumerate(tqdm.tqdm(files)):
    path = os.path.join(root, f)
    img_raw = cv2.imread(path, 0)
    img = crop_breast(img_raw)
    h,w = img.shape
    width.append(w)
    height.append(h)
    ar = w/h
    '''
    if ar > 0.9:
        fig, ax = plt.subplots(1, 2, figsize=(30,15))
        ax[0].imshow(img_raw)
        ax[1].imshow(img)
        plt.show()
    '''

width = np.array(width)
height = np.array(height)

fig, ax = plt.subplots(3, 1, figsize=(25,25))
ax[0].hist(width)
ax[0].title.set_text("width")
ax[1].hist(height)
ax[1].title.set_text("height")

ax[2].hist(width/height, bins=100)
ax[2].title.set_text("AR")



plt.show()