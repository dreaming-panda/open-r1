
export WANDB_PROJECT="MTP"
python src/open_r1/mtp3_save.py
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft_mtp3.py  --config recipes/OpenR1-Qwen-7B/sft/config1.5BMTPLORA.yaml

