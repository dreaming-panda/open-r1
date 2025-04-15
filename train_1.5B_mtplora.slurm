#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH -e batch-%j.err
#SBATCH -o batch-%j.out
#SBATCH -c 48
#SBATCH --mem=512G
#SBATCH --job-name=MTPLORA

nvidia-smi
source /home/jwzhao/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate /home/jwzhao/anaconda3/envs/czm

cd /home/jwzhao/scripts/open-r1

ls
wandb login --relogin 95ddbfff160be1b11674c6673ea55772c7ea20cc
export WANDB_PROJECT="MTP"

python src/open_r1/mtp3_save.py
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft_mtp3.py  --config recipes/OpenR1-Qwen-7B/sft/config1.5BMTPLORA_mt.yaml

ls /fsx-storygen/jwzhao/checkpoints/Qwen1.5B-MTP-Lora