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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--track", default=False, type=lambda x:bool(strtobool(x)), nargs='?', const=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    path = Path(args.data_path)
    
    cbs = None
    
    if args.track:
        run = wandb.init(project="ml4h",
                name=f"train_{args.task}_{path.name}",
                job_type="train")
        run.config.update(args) 
        cbs = [WandbCallback(), SaveModelCallback()]

    print("Loading images...")
    
    data = ImageDataLoaders.from_folder(path, train="train", valid="val",
                                        bs=32, num_workers=2, 
        batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)],
        item_tfms=None)
    # data = ImageDataLoaders.from_folder(path, valid_pct=0.2, bs=32, num_workers=2, 
           # batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)],
           # item_tfms=Resize(448, method="pad", pad_mode="zeros"))

    print("Loading ResNet50 model...")
    # Build the CNN model with the pretrained resnet50
    learn = cnn_learner(data, models.resnet50, metrics = [accuracy, 
						F1Score(), 
						RocAucBinary(),
						Precision(),
						Recall()], cbs=cbs)

    print("Begin training")
    learn.fit_one_cycle(args.num_epochs, args.lr);

    # filename = "hospital_system-none-1"
    # learn.save(filename)
    # learn.load(filename)

    # model_artifact = wandb.Artifact('model', type='model')
    # model_artifact.add_file("models/" + filename)
    # wandb.run.log_artifact(model_artifact)

