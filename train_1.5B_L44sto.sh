#!/bin/bash
export WANDB_PROJECT="Stochastic"
python src/open_r1/unroll.py --start 12 --end 16 --num_conv 4 --output_dir Qwen1.5B-L44-Flat
huggingface-cli download ZMC2019/Qwen1.5B-L44-Stochastic-tuned --local-dir data/Qwen1.5B-L44-Stochastic-tuned
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft_stochas.py  --config recipes/OpenR1-Qwen-7B/sft/config1.5BL44.yaml