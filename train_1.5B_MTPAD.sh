
export WANDB_PROJECT="MTP"
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft_mtp2.py  --config recipes/OpenR1-Qwen-7B/sft/config1.5BMTP2AD.yaml

