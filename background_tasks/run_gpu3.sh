#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
## resources
#SBATCH --partition gpu3
#SBATCH --cpus-per-task=1  # cores per task
#SBATCH --mem-per-cpu=12G  # memory per core (default is 1GB/core)
#SBATCH --time 1-23:00     # days-hours:minutes
#SBATCH --account=general-gpu  # investors will replace this (e.g. `rc-gpu`)
#SBATCH --gres gpu:1

## labels and outputs
#SBATCH --job-name=goggins_gpu_test
#SBATCH --output=results-%j.out  # %j is the unique jobID

## notifications
#SBATCH --mail-user=username@missouri.edu  # email address for notifications
#SBATCH --mail-type=END,FAIL  # which type of notifications to send
#-------------------------------------------------------------------------------

echo "### Starting at: $(date) ###"

## Module Commands
module load miniconda3
module load cudnn/cudnn-7.1.4-cuda-9.0.176

# Conda Env
source activate /storage/hpc/group/gogginsgroup/conda_env

# Science goes here:
picpath='/storage/hpc/group/gogginsgroup/AVA/images'
labelpath='/storage/hpc/group/gogginsgroup/AVA/labels'
modelname='Mar6_MINI32_AVA_vgg16.pt'

nohup python vgg16_quality_trainer.py $modelname &> vgg16_trainer.out &

echo "### Ending at: $(date) ###"