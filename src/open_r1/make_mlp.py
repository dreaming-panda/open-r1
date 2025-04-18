from transformers import Qwen2Config, AutoTokenizer, Qwen2ForCausalLM

import torch
import copy
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-7B-Instruct",help='model')
parser.add_argument('--scaler', type=float, default=1.25,help='sclaer')
parser.add_argument('--output_dir', type=str, default="Qwen7B-MP-AMP",help='output_dir')
args = parser.parse_args()
# Load model and configuration
config = Qwen2Config.from_pretrained(args.model)


model = Qwen2ForCausalLM.from_pretrained(args.model, config=config)
tokenizer = AutoTokenizer.from_pretrained(args.model)

im_size = int(args.scaler * config.intermediate_size)
h_size = config.hidden_size
exp_size = im_size - config.intermediate_size
for layer in model.model.layers:
    exp_gate = torch.nn.Linear(h_size, im_size, bias=False)
    exp_up = torch.nn.Linear(h_size, im_size, bias=False)
    exp_down = torch.nn.Linear(im_size, h_size, bias=False)
    exp_gate.weight.data[:-exp_size].copy_(layer.mlp.gate_proj.weight.data)
    exp_gate.weight.data[-exp_size:].zero_()
    exp_up.weight.data[:-exp_size].copy_(layer.mlp.up_proj.weight.data)
    exp_up.weight.data[-exp_size:].copy_(layer.mlp.up_proj.weight.data[-exp_size:])
    
    exp_down.weight.data[:,:-exp_size].copy_(layer.mlp.down_proj.weight.data)
    exp_down.weight.data[:,-exp_size:].copy_(layer.mlp.down_proj.weight.data[:,-exp_size:])
    del layer.mlp.up_proj
    del layer.mlp.down_proj
    del layer.mlp.gate_proj
    
    layer.mlp.up_proj = exp_up
    layer.mlp.gate_proj = exp_gate
    layer.mlp.down_proj = exp_down

model.config.intermediate_size = im_size
model = model.to(torch.bfloat16)
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print(model)