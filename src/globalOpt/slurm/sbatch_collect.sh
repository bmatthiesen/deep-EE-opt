#!/bin/bash

#SBATCH -J "collect"
#SBATCH --account=p_mwrc
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=10583
#SBATCH --cpus-per-task=1

ml load h5py

srun python3 collect.py
