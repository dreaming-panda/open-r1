#!/bin/bash

nvidia-smi

#python unroll.py --model 
MODEL=ZMC2019/Qwen1.5B-L28E6
#MODEL=ZMC2019/Qwen1.5B-Conv-L52
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=8,max_model_length=32768,gpu_memory_utilization=0.9,generation_parameters={max_new_tokens:30768,temperature:0.6,top_p:0.95}"

OUTPUT_DIR=data/evals/$MODEL

# AIME 2024
TASK=amc23
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

TASK=aime24
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
     --custom-tasks src/open_r1/evaluate.py \
     --use-chat-template \
     --output-dir $OUTPUT_DIR

TASK=gsm8k
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

TASK=math_500
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
     --custom-tasks src/open_r1/evaluate.py \
     --use-chat-template \
     --output-dir $OUTPUT_DIR
