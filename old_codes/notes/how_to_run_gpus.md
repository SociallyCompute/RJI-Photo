## How to Run the Training Model on GPUs (Lewis Cluster)
##### Make a Conda Environment
First you need to create a personal conda environment. With a shared environment this just happens once. Specific instructions can be followed here: http://docs.rnet.missouri.edu/Software/anaconda 
1. Create a .condarc file at ~/.condarc and add the following code:
    ``` 
    envs_dirs:
            - /storage/hpc/data/${USER}/miniconda/envs
        pkgs_dirs:
            - /storage/hpc/data/${USER}/miniconda/pkgs
    ```

2. Create a partition for cpu to run commands:
    `srun --partition Interactive --pty /bin/bash`

3. Load anaconda 3:
    `module load miniconda3`

4. Create the Conda Environment (replace my_environment with the name of the environment):
    `conda create -n my_environment python=3.7`

5. Activate your conda environment:
    `source activate my_environment`

#### Load Libraries
Secondly, load all the necessary libraries
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install sqlalchemy
conda install matplotlib
conda install pathlib2
```
Ensure pillow is installed too, there have been issues with versions working correctly with pytorch. Installing this way should install pillow based on what is compatible with pytorch.

#### Changing Variables
Before we run it, ensure all values are correct in the `run_gpu3.sh` are correct. Change the variables to fit what you need them to be (specifically check your model name/batch_size/epochs) In addition to the general variables check the SLURM variables. They are outlined below:
```
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
#SBATCH --mail-user=mjc6r9@mail.missouri.edu  # email address for notifications
#SBATCH --mail-type=END,FAIL  # which type of notifications to send
#-------------------------------------------------------------------------------
```

To switch to gpu4, change:
`#SBATCH --partition gpu3` to `#SBATCH --partition gpu4`
`#SBATCH --account=general-gpu` to `#SBATCH --account=engineering-gpu`
Add `#SBATCH --qos gpu4` after the last resources line (`#SBATCH --gres gpu:1`)

To change time program will run for, change:
`SBATCH --time 1-23:00` to whatever you need. It is in day-hour:min format

To change the email getting notifications, change:
`#SBATCH --mail-user=mjc6r9@mail.missouri.edu` to the email you want notifications sent

#### Run Sbatch
Finally, we are ready to run the SLURM file. It is titled `run_gpu3.sh` and is in the `background_tasks` folder. This is done by:
`sbatch run_gpu3.sh`

All files will be issued a job ID and a .out file will be created titled: `results-{jobID}.out` 

#### Monitor Job
To monitor the status of the job, use:
`sacct` - list all user jobs (Includes Completed, Pending, Running)
`squeue -p gpu3` - list all jobs queued for gpu3 (change gpu3 for gpu4 to get gpu4)
`scancel {jobID}` - cancel the job based on the job ID
