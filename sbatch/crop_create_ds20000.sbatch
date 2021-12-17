#!/bin/bash 
#SBATCH --time=24:00:00
#SBATCH --output=./out/create_20000_%j.out
#SBATCH --job-name=20000_ds
#SBATCH --account=csci_ga_3033_083
#SBATCH --partition=interactive

singularity exec $nv \
            --bind /home,/scratch,/share/apps,/mnt \
            --overlay overlay-50G-10M.ext3:ro \
            /share/apps/images/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c "
source /ext3/env.sh
conda activate
conda activate ml4h
python 4_create_datasets.py --data_path /scratch/rc4499/hospital_systems_crop20000 --crop"
