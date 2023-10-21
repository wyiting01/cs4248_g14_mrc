#!/bin/sh

#SBATCH --time=180
#SBATCH --job-name=g14_xlnet
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e0559908@comp.nus.edu.sg
#SBATCH --gpus=a100:1
#SBATCH --partition=medium
#SBATCH -o cs4248_g14.out

srun train.sh