from collections import OrderedDict
from PIL import Image
import math
import torch


def get_mil_instance_grid(image, num_wh, overlap_perc_wh, input_size=None):
    img_w, img_h = image.size
    instance_w = math.floor(img_w / ((1 - overlap_perc_wh[0]) * (num_wh[0] - 1) + 1))
    instance_h = math.floor(img_h / ((1 - overlap_perc_wh[1]) * (num_wh[1] - 1) + 1))

    step_w = (1 - overlap_perc_wh[0]) * instance_w
    step_h = (1 - overlap_perc_wh[1]) * instance_h

    instance_list = []
    for i in range(num_wh[1]):
        i1 = i * step_h
        i2 = i1 + instance_h
        for j in range(num_wh[0]):
            j1 = j * step_w
            j2 = j1 + instance_w
            if input_size is not None:
                instance = image.crop((j1, i1, j2, i2)).resize(
                    input_size, resample=Image.BICUBIC
                )
            else:
                instance = image.crop((j1, i1, j2, i2))

            instance_list.append(instance)

    return instance_list

