import os
import cv2
import glob
from src import *
from tqdm import tqdm

root = os.path.join(os.environ["DATASET_ROOT"], "bcd2022")
root_images = os.path.join(root, "images_512")
files = glob.glob(os.path.join(root_images, "*.png"))

mean = 0
std = 0

for f in tqdm(files):
    img = cv2.imread(f, 0)
    img = crop_breast(img)

    mean += img.mean()
    std += img.std()

n = len(files)

print(mean/n, std/n)
