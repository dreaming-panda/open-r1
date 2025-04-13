#!/bin/bash

export WANDB_PROJECT="sparsity"
#python src/open_r1/fat_save.py --multiplier 2
huggingface-cli download ZMC2019/Qwen7B-EP --revision 5466f148e29be31c8d0d379665b341cc2ea9c8aa --local-dir Qwen7B-EP
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py  --config recipes/OpenR1-Qwen-7B/sft/config7B.yaml