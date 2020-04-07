#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
## resources
#SBATCH --partition gpu3
#SBATCH --cpus-per-task=1  # cores per task
#SBATCH --mem-per-cpu=16G  # memory per core (default is 1GB/core)
#SBATCH --time 1-23:00     # days-hours:minutes
#SBATCH --account=general-gpu  # investors will replace this (e.g. `rc-gpu`)
#SBATCH --gres gpu:1

## labels and outputs
#SBATCH --job-name=goggins_gpu_test
#SBATCH --output=results-%j.out  # %j is the unique jobID

## notifications
#SBATCH --mail-user=mjc6r9@mail.missouri.edu  # email address for notifications
#SBATCH --mail-type=END,FAIL  # which type of notifications to send
#-------------------------------------------------------------------------------

echo "### Starting at: $(date) ###"

## Module Commands
module load cudnn/cudnn-7.1.4-cuda-9.0.176

# Science goes here:
avapicpath='/storage/hpc/group/augurlabs/images'
labelpath='/storage/hpc/group/augurlabs/ava/AVA.txt'
modelname='Apr7_AVA_30ep_MINI32_resnet'
dataset='AVA'
epochs='30'
batch='32'
architecture='resnet'

python model_builder.py $modelname $dataset $epochs $batch $architecture 

echo "### Ending at: $(date) ###"
