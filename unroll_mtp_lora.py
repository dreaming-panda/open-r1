from transformers import Qwen2Config, AutoTokenizer
import torch
import copy
import argparse
import open_r1
from src.open_r1.MTPQwen3 import Qwen2ForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="ZMC2019/Qwen1.5B-MTP-Lora",help='model')
args = parser.parse_args()
# Load model and configuration
config = Qwen2Config.from_pretrained(args.model)
model = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.model)

start_conv_idx = config.start_conv_idx
end_conv_idx = config.end_conv_idx
num_conv = config.num_conv
# Extract original layers and create a deep copy to avoid tensor sharing
layers = copy.deepcopy(model.model.layers[:end_conv_idx])

# Unroll the convolutions
for _ in range(num_conv):
    layers.extend(copy.deepcopy(model.model.layers[start_conv_idx:end_conv_idx]))

# Append remaining layers
layers.extend(copy.deepcopy(model.model.layers[end_conv_idx:]))

# Replace model layers with the expanded ones
model.model.layers = torch.nn.ModuleList(layers)

# Update configuration
model.config.num_hidden_layers = config.num_hidden_layers + (end_conv_idx - start_conv_idx) * num_conv
for layer_idx in range(start_conv_idx, end_conv_idx):
    del model.model.layers[layer_idx].self_attn.adapter_q
    del model.model.layers[layer_idx].self_attn.adapter_v

for layer_idx in range(end_conv_idx, end_conv_idx + (num_conv - 1) * (end_conv_idx - start_conv_idx)):
    q0 = model.model.layers[layer_idx].self_attn.adapter_q[0].weight.data
    q1 = model.model.layers[layer_idx].self_attn.adapter_q[1].weight.data

# Save the modified model
model.save_pretrained("Qwen-1.5B-Instruct-Conv-Unrolled")
tokenizer.save_pretrained("Qwen-1.5B-Instruct-Conv-Unrolled")
print(model)
