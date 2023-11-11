#!/bin/sh

#SBATCH --time=180
#SBATCH --job-name=g14_xlnet
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e0559908@comp.nus.edu.sg
#SBATCH --gpus=v100:1
#SBATCH --mem=0
#SBATCH --partition=medium
#SBATCH -o cs4248_g14.out

chmod +x train.sh
chmod +x test.sh
srun train.sh
srun test.sh