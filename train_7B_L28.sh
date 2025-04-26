#!/bin/bash
export WANDB_PROJECT="MATH"
huggingface-cli download ZMC2019/Qwen7B-Math-L28-Base2 --local-dir data/Qwen7B-Math-L28-Base2
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py  --config recipes/OpenR1-Qwen-7B/sft/config7BMath.yaml