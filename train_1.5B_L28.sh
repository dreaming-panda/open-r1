#!/bin/bash
export CUDA_VISIBLE_DEVICES=6,7,8,0
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py  --config recipes/OpenR1-Qwen-7B/sft/config1.5BL28.yaml