#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft_adj_lr_freeze.py  --config recipes/OpenR1-Qwen-7B/sft/config1.5BL44_adjlr.yaml