from transformers import Qwen2Config, AutoTokenizer
from MTPQwen4 import Qwen2ForCausalLM
import torch
import copy
from torch import nn
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-1.5B-Instruct",help='model')
parser.add_argument('--output_dir', type=str, default="Qwen2.5-1.5B-MTP",help='output_dir')
args = parser.parse_args()

def init_weights(module):
    std = model.config.initializer_range  # 或者自定义 std
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

# Load model and configuration
config = Qwen2Config.from_pretrained(args.model)
config.start_conv_idx = 0
config.end_conv_idx = 0
config.num_conv = 0
model = Qwen2ForCausalLM.from_pretrained(args.model, config=config)

tokenizer = AutoTokenizer.from_pretrained(args.model)
model.model.mtp.apply(init_weights)
model.model.mtp.transformer.load_state_dict(model.model.layers[-1].state_dict())

# Save the modified model
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print(model)