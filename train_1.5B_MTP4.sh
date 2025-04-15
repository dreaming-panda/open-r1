
export WANDB_PROJECT="MTP"
python src/open_r1/mtp_init.py
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft_mtp4.py  --config recipes/OpenR1-Qwen-7B/sft/config1.5BMTP4.yaml 