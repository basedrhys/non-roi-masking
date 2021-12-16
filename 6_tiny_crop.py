# Import necessary libraries
from fastai.basics import *
from fastai.data.all import *
from fastai.vision.all import *
from fastai.callback.all import *

from fastai.metrics import *
from fastai.callback.wandb import *

import numpy as np
from PIL import Image

from matplotlib import pyplot as plt

from tqdm.auto import tqdm

import os
import cv2
import shutil

import matplotlib.image as mpimg
from matplotlib import rcParams

from random import sample
import os

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str)
    return parser.parse_args()

# Load the raw dataset and gt masks
root_path = Path(args.data_path)
img_path = root_path/"imgs"
output_mask_path = root_path/"masks_smooth"

img_fnames = get_image_files(img_path)

os.makedirs(output_mask_path, exist_ok=True)

mask_dim = 448
crop_dim = 150
arr = np.zeros((mask_dim,mask_dim)).astype(np.uint8)
total_width = mask_dim-crop_dim
start = int(total_width / 2)
end = start + crop_dim
arr[start:end, start:end] = 255

for fname in img_fnames:
    im = Image.fromarray(arr)
    im.save(output_mask_path/img_name)