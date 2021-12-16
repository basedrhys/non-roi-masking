import math
from pathlib import Path
from os import listdir
from os.path import isfile, join
from random import choice, sample
from fastai.data.transforms import get_image_files
from fastai.vision.core import PILImage, PILMask
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# Import necessary libraries
from fastai.basics import *
from fastai.data.all import *
from fastai.vision.all import *
from fastai.callback.all import *

from fastai.metrics import *
from fastai.callback.wandb import *
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams

root_path = Path("/scratch/rc4499/chexpert_pn")
mask_path = root_path/"masks_raw"

output_path = root_path/"masks_smooth"
output_path.mkdir(exist_ok=True)

img_fnames = get_image_files(root_path/"imgs")
mask_fnames = get_image_files(mask_path)

def clean_mask(mask, kernel_size=(21, 21), blur_size=7, blur_iter=5):
    np_img = np.array(mask).astype(np.uint8)
    
    ret, np_img = cv2.threshold(np_img, 0.5, 255, cv2.THRESH_BINARY)
    orig = np.array(np_img)
    # https://stackoverflow.com/questions/37409811/smoothing-edges-of-a-binary-image/37458312
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    np_img = cv2.morphologyEx(np_img, cv2.MORPH_OPEN, kernel)
    # np_img = cv2.morphologyEx(np_img, cv2.MORPH_CLOSE, kernel)

    
    for i in range(blur_iter):
        # np_img = cv2.GaussianBlur(np_img, (7, 7), 0)
        np_img = cv2.medianBlur(np_img, blur_size)

    np_img = cv2.dilate(np_img, (5, 5), iterations=20)
    
    return np_img
    
    
for img_path in tqdm(img_fnames):
    img_name = img_path.name
    img_mask = mask_path/img_name
    
    mask_data = mpimg.imread(img_mask)
    
    smoothed = clean_mask(mask_data)
    
    im = Image.fromarray(smoothed)
    im.save(output_path/img_name)
#     # read images
#     img_A = mpimg.imread(img_path)
#     img_B = mpimg.imread(img_mask)

#     # display images
#     fig, ax = plt.subplots(1,4)
#     ax[0].imshow(img_A);
#     mask, orig, clean = clean_mask(img_B)
#     ax[1].imshow(mask, cmap="gray")
#     ax[2].imshow(orig, cmap="gray")
#     ax[3].imshow(clean, cmap="gray")
    
