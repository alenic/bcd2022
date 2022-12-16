from src import *
import os 
import matplotlib.pyplot as plt


root = os.path.join(os.environ["DATASET_ROOT"], "bcd2022", "image_512")

files = os.listdir(root)
files = files[:100]


for f in files:
    path = os.path.join(root, f)
    crop_breast(img)

