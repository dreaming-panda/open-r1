#!/bin/bash
python src/open_r1/unroll.py --start 12 --end 16 --num_conv 6 --output_dir Qwen1.5B-L52-Flat
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft_adj_lr.py  --config recipes/OpenR1-Qwen-7B/sft/config1.5BL52.yaml