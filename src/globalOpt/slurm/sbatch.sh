#!/bin/bash

#SBATCH -J "wsee_test"
#SBATCH --array=0
#SBATCH --account=p_mwrc
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=2583
#SBATCH --partition=haswell
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/p_mwrc/log-DO/%x-wp_%a-job_%A.out
#SBATCH --license=scratch

export JOB_HPC_SAVEDIR="/scratch/p_mwrc/deep-opt/${SLURM_JOB_NAME}"

module use /projects/p_mwrc/privatemodules

ml load h5py
ml load imkl
ml load foss

srun python3 /home/bmatth/deep-opt/src/globalOpt/run_wsee.py /home/bmatth/deep-opt/data/channels-7.h5
