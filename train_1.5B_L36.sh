#!/bin/bash

accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft_conv.py  --config recipes/OpenR1-Qwen-7B/sft/config1.5BL36.yaml