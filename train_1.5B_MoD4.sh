

python src/open_r1/mod_save.py --num_conv 4
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft_mod.py  --config recipes/OpenR1-Qwen-7B/sft/config1.5B_MoD4.yaml