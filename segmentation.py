# Import necessary libraries
from fastai.basics import *
from fastai.data.all import *
from fastai.vision.all import *
from fastai.callback.all import *

from fastai.metrics import *
from fastai.callback.wandb import *
from pathlib import Path

from tqdm import tqdm
from math import ceil

import wandb
# wandb.login()

path = Path("/scratch/rc4499/NLM-kaggle")
image_path = path/"CXR_png"
mask_path = path/"masks"

img_fnames = get_image_files(image_path)
mask_fnames = get_image_files(mask_path)

len(img_fnames), len(mask_fnames)

def get_msk(o):
    m_path = mask_path/o.name
    msk = np.array(PILMask.create(m_path))
    msk[msk == 255] = 1
    assert msk.max() == 1
    return PILMask.create(msk)

codes = ["bg", "fg"]

lungs = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                  get_items=get_image_files,
                  splitter=RandomSplitter(valid_pct=0.2, seed=42),
                  get_y=get_msk,
                  batch_tfms=[Normalize.from_stats(*imagenet_stats)],
                   item_tfms=Resize(448, method="pad", pad_mode="zeros"))

dls = lungs.dataloaders(image_path, bs=1)
dls.vocab = codes
dls.show_batch(figsize=(16,10))

filename = 'combined-2-v3-padzero'
learn = unet_learner(dls, resnet50, metrics=Dice, 
                     self_attention=True, 
                     act_cls=Mish, opt_func=ranger)
learn.load(filename)
print("Model loaded...")

root_path = Path("/scratch/rc4499/chexpert_pn")
img_path = root_path/"imgs"

mask_path = root_path/"masks_raw"
mask_path.mkdir(exist_ok=True)

test_imgs = get_image_files(img_path)
test_imgs.sort()
test_imgs

bs=1000
num_batches = ceil(len(test_imgs) / bs)
print(f"{num_batches} batches of size {bs}")

for batch_i in range(num_batches):
    start_idx = batch_i * bs
    end_idx = (batch_i + 1) * bs
    
    print(f"Running on batch {batch_i}/{num_batches} from {start_idx} to {end_idx}")
    img_batch = test_imgs[start_idx:end_idx]
    
    dl = learn.dls.test_dl(img_batch)
    dl.show_batch()

    preds = learn.get_preds(dl=dl)
    
    for i, path in tqdm(enumerate(img_batch)):
        pred = preds[0][i]
        pred_arg = pred.argmax(dim=0).numpy()
        if i == 0:
            assert pred_arg.max() == 1
        rescaled = (pred_arg * 255).astype(np.uint8)

        im = Image.fromarray(rescaled)
        im.save(mask_path/path.name)

print("Done")