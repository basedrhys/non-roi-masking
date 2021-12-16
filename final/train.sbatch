#!/bin/bash

#SBATCH --job-name="${NAME}_${VARIATION}"
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output="./out/${NAME}_${VARIATION}_%j.out"
#SBATCH --time=1:00:00
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
python train.py --base_path "/scratch/rc4499/masked/${NAME}_clean" --variation $VARIATION --iter $ITER --track
"
