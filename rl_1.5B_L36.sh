#!/bin/bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
trl vllm-serve --model ZMC2019/Qwen1.5B-L36-Flat-90K --tensor_parallel_size 4 --gpu_memory_utilization 0.9 &

sleep 120

export CUDA_VISIBLE_DEVICES=4,5,6,7
accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=4 src/open_r1/grpo.py \
    --config recipes/Qwen2.5-1.5B-Instruct/grpo/configL36.yaml
