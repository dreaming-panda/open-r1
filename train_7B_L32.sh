#!/bin/bash
export WANDB_PROJECT="MATH"
python src/open_r1/unroll.py --start 12 --end 16 --num_conv 1 --model ZMC2019/Qwen2.5-Math-7B-Instruct --output_dir Qwen7B-Math-L32
huggingface-cli download ZMC2019/Qwen7B-Math-L32 --local-dir data/Qwen7B-Math-L32
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py  --config recipes/OpenR1-Qwen-7B/sft/config7BL32.yaml