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

python ../background_tasks/fill_db.py

echo "### Ending at: $(date) ###"