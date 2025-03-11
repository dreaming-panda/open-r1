#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=512G
#SBATCH --job-name=Sparse7B
#SBATCH -o Sparse7B.o
#SBATCH -e Sparse7B.e
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chenzhuoming911@gmail.com
nvidia-smi
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r1-conv
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft_sparse.py  --config recipes/OpenR1-Qwen-7B/sft/config_sparse.yaml