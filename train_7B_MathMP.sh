#!/bin/bash
export WANDB_PROJECT="MATH"
python src/open_r1/make_mlp.py --scaler 1.3 --model ZMC2019/Qwen2.5-Math-7B-Instruct --output_dir Qwen7B-Math-L28-MP
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py  --config recipes/OpenR1-Qwen-7B/sft/config7BMathMP.yaml