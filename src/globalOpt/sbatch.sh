#!/bin/bash

# Copyright (C) 2018 Bho Matthiesen
# 
# This program is used in the article:
# 
# Bho Matthiesen, Alessio Zappone, Eduard A. Jorswieck, and Merouane Debbah,
# "Deep Learning for Optimal Energy-Efficient Power Control in Wireless
# Interference Networks," submitted to IEEE Journal on Selected Areas in
# Communication.
# 
# License:
# This program is licensed under the GPLv2 license. If you in any way use this
# code for research that results in publications, please cite our original
# article listed above.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

#SBATCH -J "wsee_lambert"
#SBATCH --array=0-1999
#SBATCH --account=p_mwrc
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=2583
#SBATCH --partition=haswell
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/p_mwrc/deep-opt/log/%x-wp_%a-job_%A.out
#SBATCH --license=scratch

export JOB_HPC_SAVEDIR="/scratch/p_mwrc/deep-opt/${SLURM_JOB_NAME}"

module use /projects/p_mwrc/privatemodules

ml load h5py
ml load imkl
ml load foss

srun python3 /home/bmatth/deep-opt/src/globalOpt/run_wsee.py /home/bmatth/deep-opt/data/selchan.h5
