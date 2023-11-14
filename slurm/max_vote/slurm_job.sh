#!/bin/sh

#SBATCH --time=4320
#SBATCH --mem=0
#SBATCH --job-name=max_vote
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e0543676@comp.nus.edu.sg
#SBATCH --gpus=a100mig:7
#SBATCH --partition=long
#SBATCH -o cs4248_max_vote.out

chmod +x max_vote.sh
srun max_vote.sh