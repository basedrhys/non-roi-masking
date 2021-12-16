from pathlib import Path

# from fastai.vision import *
from fastai.data.transforms import get_image_files
from fastai.vision.core import PILImage, PILMask

import numpy as np
from skimage.morphology.convex_hull import convex_hull_image
from PIL import Image

from matplotlib import pyplot as plt

from tqdm.auto import tqdm

import os
import cv2
import shutil

from matplotlib import rcParams

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--type", type=str)
    #parser.add_argument("--crop", default=False, type=lambda x:bool(strtobool(x)), nargs='?', const=True)
    return parser.parse_args()

def get_mask_path(dataset_type, fpath):
    return mask_path/fpath.name
    # if dataset_type == "pneumonia":
    #     return mask_path/fpath.name
    # if dataset_type == "hospital_systems":
    #     return mask_path/fpath.name
    # if dataset_type == "hospital_systems_clean":
    #     return 

def create_instances(img_fnms, dataset_type):
    result = []
    for fname in tqdm(img_fnms, desc="Creating instances"):
        mask_name = get_mask_path(dataset_type, fname)
        result.append((fname, mask_name))
        
    return result

from sklearn.model_selection import train_test_split

def create_splits(df, already_split = True, combine=False):
    """
    Create the training/val/test split
    """
    if combine:
        return {"combined": df}
    
    train_ratio = 0.80
    # validation_ratio = 0.10
    # test_ratio = 0.10
    df_train, df_val, df_test = [], [], []
    if already_split:
        for p in df:
            if "/train/" in str(p):
                df_train.append(p)
            elif "/val/" in str(p):
                df_val.append(p)
            elif "/test/" in str(p):
                df_test.append(p)
            else:
                raise Exception("Split not found")
    else:
        # train is now 80% of the entire data set
        # the _junk suffix means that we drop that variable completely
        df_train, df_val = train_test_split(df, test_size=1 - train_ratio, random_state=0)
        df_test = []
        # # # test is now 10% of the initial data set
        # # # validation is now 10% of the initial data set
        # df_val, df_test = train_test_split(df_val, test_size=test_ratio/(test_ratio + validation_ratio), random_state=42) 

    print(f"Split {len(df)} instances into train:{len(df_train)}, val:{len(df_val)}, test:{len(df_test)}")
    
    return {"train": df_train, "val": df_val, "test": df_test}

def label_func(fname, dataset_type):
    fname = str(fname)
    if "pneumonia" in dataset_type or "chexpert_pn" in dataset_type:
        if "/NORMAL/" in fname:
            return "NORMAL"
        elif "/PNEUMONIA/" in fname:
            return "PNEUMONIA"
        else:
            raise Exception("Invalid file path")
    elif "hospital_systems" in dataset_type:
        if "/chexpert/" in fname:
            return "chexpert"
        elif "/iu/" in fname:
            return "iu"
        elif "/nih/" in fname:
            return "nih"
        else:
            raise Exception("Invalid file path")
    else:
        raise Exception("Invalid dataset type")
        
from PIL import Image, ImageOps

def resize_img(im, desired_size=448):
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_im

if __name__ == "__main__":
    args = parse_args()
    
    # Load the raw dataset and gt masks
    root_path = Path(args.data_path)
    dataset_type = root_path.name
    print("Using square crop instead of predicted masks...")
    
    img_path = root_path/"imgs"
    mask_path = root_path/"masks_smooth"
    
    print("Getting image files")
    img_fnames = get_image_files(img_path)
    print(f"Found {len(img_fnames)}")
    
    print("Creating instances")
    instances = create_instances(img_fnames, dataset_type)

    mask_dim = 448
    crop_dim = 150
    square_crop = np.zeros((mask_dim,mask_dim)).astype(np.uint8)
    total_width = mask_dim-crop_dim
    start = int(total_width / 2)
    end = start + crop_dim
    square_crop[start:end, start:end] = 255
    
    if args.type:
        MASK_VERSIONS=[args.type]
    else:
        MASK_VERSIONS = ["crop", "none", "raw", "convex_hull"]
    
    splits = create_splits(instances, already_split=False)
    
    output_path = Path(f"/scratch/rc4499/masked/{dataset_type}")
    for mask_version in tqdm(MASK_VERSIONS, desc="Mask version"):
        print(f"Beginning processing for mask type {mask_version}")
        for split_name, split_files in tqdm(splits.items(), desc="Data Split"):

            output_dir = output_path/mask_version/split_name

            shutil.rmtree(output_dir, ignore_errors=True)

            for inst in tqdm(split_files, total=len(split_files)):
                img_fname = inst[0]
                mask_fname = inst[1]

                # load the original input image and display it to our screen
                # image = np.asarray(PILImage.create(img_fname))
                image = cv2.imread(str(img_fname))
                if image is None:
                    print(f"Couldn't load image {img_fname}")
                    continue

                resized_img = resize_img(image)
                
                #if args.crop:
                #    mask = square_crop
                #else:
                mask = np.asarray(PILMask.create(mask_fname))

                if mask_version == "none":
                    mask[:,:] = 255
                elif mask_version == "raw":
                    pass
                elif mask_version == "convex_hull":
                    hull = convex_hull_image(mask)
                    mask = hull.astype(np.uint8) * 255
                elif mask_version == "crop":
                    mask = square_crop
                else:
                    raise Exception(f"Invalid mask version {mask_version}")

                masked = cv2.bitwise_and(resized_img, resized_img, mask=mask)

                im = Image.fromarray(masked)
                os.makedirs(f"{output_dir}/{label_func(img_fname, dataset_type)}", exist_ok=True)
                im.save(f"{output_dir}/{label_func(img_fname, dataset_type)}/{img_fname.name}")
    
    print("Done.")
