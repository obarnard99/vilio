#!/bin/bash

#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p pascal

. /etc/profile.d/modules.sh
module purge
module load rhel7/default-gpu
module load cuda/10.0
module load cudnn/7.6_cuda-10.0

source ~/ernie-vil/bin/activate

nvidia-smi
module list
