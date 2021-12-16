#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --partition=n1s8-v100-1
#SBATCH --account=csci_ga_3033_083 

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

singularity exec $nv \
	    --bind /home,/scratch,/share/apps,/mnt \
	    --overlay ../overlay-50G-10M.ext3:ro \
            /share/apps/images/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "
source /ext3/env.sh
conda activate
conda activate ml4h
python eval.py
"
