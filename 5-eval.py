# Import necessary libraries
from fastai.basics import *
from fastai.data.all import *
from fastai.vision.all import *
from fastai.callback.all import *

from fastai.metrics import *
from fastai.callback.wandb import *
import wandb

eval_results = []

ITER=[1, 2, 3]
DS_NAMES=["chexpert_pn_clean", "pneumonia_clean"]
VARIATIONS=["none", "raw", "convex_hull", "crop"]
eval_split = ["train", "valid"]

num_iter = len(DS_NAMES) * len(VARIATIONS) * len(ITER) * len(eval_split)

eval_data_root = Path("/scratch/rc4499/masked")

i = 0

for train_iter in ITER:
    for orig_ds in DS_NAMES:
        for orig_var in VARIATIONS:
            for new_ds in DS_NAMES:
                for new_var in VARIATIONS:
                    # Load the eval data
                    data_path = eval_data_root/new_ds/new_var
                    # print(data_path)

                    dls = ImageDataLoaders.from_folder(data_path, train="train", valid="val",
                            bs=32, num_workers=2, 
                            batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)],
                            item_tfms=None)

                    learn = cnn_learner(dls, models.resnet50, metrics = [accuracy, 
                        F1Score(), 
                        RocAucBinary(),
                        Precision(),
                        Recall()], cbs=None)

                    model_pth = f"/home/rc4499/final/models_fnl/{train_iter}-{orig_ds}-{orig_var}"
                    learn.load(model_pth)


                    for split in eval_split:
                        if split == "train":
                            my_dl = dls.train
                        elif split == "valid":
                            my_dl = dls.valid
                        else:
                            raise Exception()
                        
                        res = learn.validate(dl=my_dl)

                        params = [train_iter, orig_ds, orig_var, new_ds, new_var, split]
                        params.extend(res)
                        eval_results.append(params)
                        
                        i+=1
                        print(f"Iteration {i}/{num_iter}")
                        
                        
df = pd.DataFrame(eval_results, columns=[
 "iter", "model_ds", "model_var", "eval_ds", "eval_var", "split", "valid_loss", "accuracy", "F1Score", "RocAuc", "Precision", "Recall"
])
df.to_csv("eval_results.csv", index=False)
