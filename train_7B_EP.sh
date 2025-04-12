#!/bin/bash

python src/open_r1/fat_save.py --multiplier 3
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py  --config recipes/OpenR1-Qwen-7B/sft/config7B.yaml