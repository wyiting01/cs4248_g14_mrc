#!/bin/sh

#SBATCH --time=180
#SBATCH --mem=0
#SBATCH --job-name=g14_bilstm
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e0418141@comp.nus.edu.sg
#SBATCH --gpus=v100:1
#SBATCH --partition=medium
#SBATCH -o cs4248_g14.out

chmod +x bilstm.sh
srun bilstm.sh