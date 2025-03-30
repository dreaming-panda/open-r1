#!/bin/bash
export CUDA_VISIBLE_DEVICES=3,5,6,7
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft_relu.py  --config recipes/OpenR1-Qwen-7B/sft/config1.5B_sparse.yaml