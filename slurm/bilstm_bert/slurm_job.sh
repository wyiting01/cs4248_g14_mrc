#!/bin/sh

#SBATCH --time=4320
#SBATCH --job-name=g14_bilstm
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e0543676@comp.nus.edu.sg
#SBATCH --gpus=v100:1
#SBATCH --partition=long
#SBATCH -o cs4248_g14.out

chmod +x bilstm_train.sh
chmod +x bilstm_train_kf.sh
srun bilstm_train.sh
srun bilstm_train_kf.sh