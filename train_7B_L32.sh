#!/bin/bash

accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft_adj_lr.py  --config recipes/OpenR1-Qwen-7B/sft/config7BL32.yaml