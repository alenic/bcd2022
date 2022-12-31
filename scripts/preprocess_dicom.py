import os
import numpy as np
import matplotlib.pyplot as plt
import dicomsdl
import cv2
import math
import glob
import time
import multiprocessing as mp
import json
import tqdm

MAX_SIZE = 1024
NUM_PROCESS = 8

J = os.path.join

out_dataset = J(os.environ["DATASET_ROOT"], "bcd2022", f"alenic_train_images_{MAX_SIZE}")
out_json_dataset = J(os.environ["DATASET_ROOT"], "bcd2022", f"alenic_train_json_{MAX_SIZE}")
root = J(os.environ["DATASET_ROOT"], "rsna-bcd2022-dicom", "train_images")




def dicom2array(path, max_size=MAX_SIZE):
    ds = dicomsdl.open(path)
    data = ds.pixelData()
    info = ds.getPixelDataInfo()

    # Resize
    h, w = data.shape
    max_s = max(h, w)
    new_w = math.ceil((max_size/max_s)*w)
    new_h = math.ceil((max_size/max_s)*h)
    data = cv2.resize(data, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    return data.astype(np.uint16), info

def worker(paths, id_worker):
    info_dict = {}
    for k, p in enumerate(tqdm.tqdm(paths)):
        #data, info = dicom2array(p)
        ds = dicomsdl.open(p)
        info = ds.getPixelDataInfo()
        
        path_split = p.split("/")
        image_id = path_split[-1].replace(".dcm","")
        patient_id  = path_split[-2]
        id_name = patient_id+"_"+image_id
        info_dict[id_name] = info

        #np.savez_compressed(J(out_dataset, f"{id_name}.npz"), data=data)


    with open(J(out_json_dataset, f"{id_worker}.json"), "w") as fp:
        json.dump(info_dict, fp)
            

    



os.makedirs(out_dataset, exist_ok=True)
os.makedirs(out_json_dataset, exist_ok=True)

file_paths = glob.glob(J(root, "**/*.dcm"))
print(file_paths[:4])
process = []
n_p = NUM_PROCESS
n_w = math.ceil(len(file_paths)/n_p)
for i in range(n_p):
    i1 = n_w*i
    i2 = i1 + n_w
    process.append(mp.Process(target=worker, args=(file_paths[i1:i2], i)))
    process[i].start()


for i in range(n_p):
    process[i].join()
