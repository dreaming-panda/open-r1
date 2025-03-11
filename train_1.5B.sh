#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=512G
#SBATCH --job-name=Conv1.5B
#SBATCH -o Conv1.5B.o
#SBATCH -e Conv1.5B.e
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chenzhuoming911@gmail.com
nvidia-smi
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r1-conv
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft_conv.py  --config recipes/OpenR1-Qwen-7B/sft/config1.5B.yaml