from transformers import Qwen2ForCausalLM, Qwen2Config, AutoTokenizer
import torch
import copy
import argparse
import open_r1
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="ZMC2019/Qwen1.5B-Conv-L52-rolled",help='model')
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

# Save the modified model
model.save_pretrained("Qwen-1.5B-Instruct-Conv-Unrolled")
tokenizer.save_pretrained("Qwen-1.5B-Instruct-Conv-Unrolled")
print(model)
