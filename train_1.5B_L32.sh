#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5 accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py  --config recipes/OpenR1-Qwen-7B/sft/config1.5BL32.yaml