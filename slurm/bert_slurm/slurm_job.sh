#!/bin/sh

#SBATCH --time=180
#SBATCH --job-name=g14_bert
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e0538450@comp.nus.edu.sg
#SBATCH --gpus=v100:1
#SBATCH --partition=medium
#SBATCH -o cs4248_g14.out

chmod +x test.sh
srun test.sh