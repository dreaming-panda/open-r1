#!/bin/bash

export WANDB_PROJECT="sparsity"
python src/open_r1/make_mlp.py
huggingface-cli download ZMC2019/Qwen7B-MP --local-dir data/Qwen7B-MP
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py  --config recipes/OpenR1-Qwen-7B/sft/config7BMP.yaml