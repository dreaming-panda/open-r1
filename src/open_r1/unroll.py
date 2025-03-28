from transformers import Qwen2ForCausalLM, Qwen2Config, AutoTokenizer
import torch
import copy
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-1.5B-Instruct",help='model')
parser.add_argument('--start', type=int, default=12,help='start_conv_idx')
parser.add_argument('--end', type=int, default=16,help='end_conv_idx')
parser.add_argument('--num_conv', type=int, default=4,help='num_conv')
parser.add_argument('--output_dir', type=str, default="Qwen1.5B-L60-Flat",help='output_dir')
args = parser.parse_args()
# Load model and configuration
config = Qwen2Config.from_pretrained(args.model)
model = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.model)

start_conv_idx = args.start
end_conv_idx = args.end
num_conv = args.num_conv

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

# Save the modified model
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print(model)
