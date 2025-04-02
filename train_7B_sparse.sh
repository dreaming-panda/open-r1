#!/bin/bash

accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft_relu.py  --config recipes/OpenR1-Qwen-7B/sft/config7B_P25.yaml