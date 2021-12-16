#!/bin/bash

#SBATCH --job-name=gpu-job
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=n1s8-v100-1
#SBATCH --account=csci_ga_3033_083 
#SBATCH --requeue

DATASET_NAME="hospital_systems"

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

singularity exec $nv \
	    --bind /home,/scratch,/share/apps,/mnt \
	    --overlay overlay-50G-10M.ext3 \
            /share/apps/images/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "
source /ext3/env.sh
conda activate
conda activate ml4h
python segmentation.py
# scp -rp /scratch/rc4499/$DATASET_NAME/masks_raw greene-dtn:/scratch/rc4499/$DATASET_NAME/masks_raw
"
