# Import necessary libraries
from fastai.basics import *
from fastai.data.all import *
from fastai.vision.all import *
from fastai.callback.all import *

from fastai.metrics import *
from fastai.callback.wandb import *
from pathlib import Path
import wandb
import argparse
import torch
import numpy as np
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", required=True, type=str)
    parser.add_argument("--variation", required=True, type=str)
    parser.add_argument("--iter", required=True, type=int)
    parser.add_argument("--num_epochs", default=20, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--track", default=False, type=lambda x:bool(strtobool(x)), nargs='?', const=True)
    return parser.parse_args()

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    args = parse_args()
    
    random_seed(args.iter * 1000, True)
    
    path = Path(args.base_path)
    
    cbs = None
    
    if args.track:
        run = wandb.init(project="ml4h",
                name=f"{args.iter}-{path.name}-{args.variation}",
                job_type="final")
        run.config.update(args) 
        cbs = [WandbCallback(), SaveModelCallback()]

    print("Loading images...")
    img_folder = path/args.variation
    print(f"Found {len(get_image_files(img_folder))} total images")
    
    data = ImageDataLoaders.from_folder(path/args.variation, train="train", valid="val",
                                        bs=32, num_workers=2, 
        batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)],
        item_tfms=None)

    print("Loading ResNet50 model...")
    # Build the CNN model with the pretrained resnet50
    learn = cnn_learner(data, models.resnet50, metrics = [accuracy, 
            F1Score(), 
            RocAucBinary(),
            Precision(),
            Recall()], cbs=cbs)

    print("Begin training")
    learn.fit_one_cycle(args.num_epochs, args.lr);

    filename = f"{args.iter}-{path.name}-{args.variation}"
    save_path = learn.save(f"/home/rc4499/final/models_fnl/{filename}")

    model_artifact = wandb.Artifact(f"model-{path.name}-{args.variation}", type='model')
    model_artifact.add_file(save_path)
    wandb.run.log_artifact(model_artifact)

