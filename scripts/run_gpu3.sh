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
#SBATCH --job-name=resnet_train
#SBATCH --output=results-%j.out  # %j is the unique jobID

## notifications
#SBATCH --mail-user=mjc6r9@mail.missouri.edu  # email address for notifications
#SBATCH --mail-type=END,FAIL  # which type of notifications to send
#-------------------------------------------------------------------------------

echo "### Starting at: $(date) ###"

## Module Commands
module load cudnn/cudnn-7.1.4-cuda-9.0.176

# Science goes here:
modelname='May7_ava_30ep_MINI16_resnet'
dataset='ava'
epochs='30'
batch='16'
architecture='resnet'
classification='content'

python ../background_tasks/model_builder.py $modelname $dataset $epochs $batch $architecture $classification

echo "### Ending at: $(date) ###"
