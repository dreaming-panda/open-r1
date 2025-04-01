#!/bin/bash
#SBATCH --job-name=sft-zero3
#SBATCH --nodes=2                      # 节点数量
#SBATCH --ntasks-per-node=8           # 每节点使用的GPU数（每个任务 = 1 GPU）
#SBATCH --gres=gpu:8                  # 每节点 GPU 数
#SBATCH --cpus-per-task=6             # 每个任务的CPU核心数
#SBATCH --mem=0                       # 使用全部内存
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out       # 日志文件输出

# === 1. 环境设置 ===
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r1-train

# === 2. 分布式设置 ===
NODE_RANK=$SLURM_NODEID
WORLD_SIZE=$SLURM_NTASKS
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500  # 你也可以随机选一个空闲端口

# === 3. 启动命令 ===
accelerate launch \
  --config_file recipes/accelerate_configs/distributed.yaml \
  --num_processes $WORLD_SIZE \
  --machine_rank $NODE_RANK \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  src/open_r1/sft.py \
  --config recipes/OpenR1-Qwen-7B/sft/config_llama.yaml
