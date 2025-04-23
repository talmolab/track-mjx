#!/bin/bash

#SBATCH --job-name=rodent-lstm     ### Job Name
#SBATCH --partition=d3
#SBATCH --time=1-23:29:00     ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1             ### Node count required for the job
#SBATCH --ntasks-per-node=1   ### Number of tasks to be launched per Node
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100:4              ### General REServation of gpu:number of gpus
#SBATCH --mem=32G
# #SBATCH --array=10 ###0-63 ###  ### Array index 
# #SBATCH --exclude=n[01-13,69,72-79,91-119,170-175,214-227,238,246,258-262,265-266,270-271,276]
#SBATCH --output=/allen/aind/scratch/tim.kim/logs/slurm-%A.%a.out
#SBATCH --error=/allen/aind/scratch/tim.kim/logs/slurm-%A.%a.err

##turn on e-mail notification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=timkimd@uw.edu

module purge
module load anaconda/2024.10
source activate track_mjx
python -m track_mjx.train data_path="/allen/aind/scratch/tim.kim/track-mjx/data/art_tmjx/2020_12_22_1.h5" +hydra.job.config_name="rodent-full-clips"
