#!/bin/bash
#SBATCH --time=30:00
#SBATCH --gpus-per-task=8
#SBATCH -e batch-%j.err
#SBATCH -o batch-%j.out
#SBATCH -c 48
#SBATCH --mem=512G
#SBATCH --job-name=test

nvidia-smi
source /home/jwzhao/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate /home/jwzhao/anaconda3/envs/czm

cd /home/jwzhao/scripts/open-r1

ls
wandb login --relogin 95ddbfff160be1b11674c6673ea55772c7ea20cc

accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py  --config recipes/OpenR1-Qwen-7B/sft/config1.5B.yaml
