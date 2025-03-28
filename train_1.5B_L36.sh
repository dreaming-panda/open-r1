#!/bin/bash
python src/open_r1/unroll.py --start 12 --end 16 --num_conv 2 --output_dir Qwen1.5B-L36-Flat
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft_adj_lr.py  --config recipes/OpenR1-Qwen-7B/sft/config1.5BL36.yaml