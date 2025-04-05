#!/bin/bash
python src/open_r1/unroll.py --start 12 --end 16 --num_conv 4 --output_dir Qwen7B-L44-Flat
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft_conv.py  --config recipes/OpenR1-Qwen-7B/sft/config7BL44.yaml