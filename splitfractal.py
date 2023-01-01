import os
import numpy as np
import shutil
import glob


root = os.path.join(os.environ["DATASET_ROOT"], "fractaldb-60")

val_out = os.path.join(root,"val")

files = glob.glob(root+"/**/*.png")

np.random.shuffle(files)

n = len(files)

nt = int(0.2*n)

os.makedirs(val_out, exist_ok=True)

for f in files[:nt]:
    src = f
    fn = f.split("/")[-1]
    c  = f.split("/")[-2]
    os.makedirs(os.path.join(val_out, c,), exist_ok=True)
    shutil.move(f, os.path.join(val_out, c, fn))





