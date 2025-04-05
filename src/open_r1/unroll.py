from transformers import Qwen2Config, AutoTokenizer, Qwen2ForCausalLM
# from MoDQwen2 import Qwen2ForCausalLM
import torch
import copy
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-1.5B-Instruct",help='model')
parser.add_argument('--start', type=int, default=12,help='start_conv_idx')
parser.add_argument('--end', type=int, default=16,help='end_conv_idx')
parser.add_argument('--num_conv', type=int, default=1,help='num_conv')
parser.add_argument('--output_dir', type=str, default="Qwen1.5B-L32-MoD",help='output_dir')
args = parser.parse_args()
# Load model and configuration
config = Qwen2Config.from_pretrained(args.model)
config.start_conv_idx = args.start
config.end_conv_idx = args.end
config.num_conv = args.num_conv

model = Qwen2ForCausalLM.from_pretrained(args.model, config=config)
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

# model.model.router[0].weight.data.normal_(mean=0.0, std=0.02)
# model.model.router[2].weight.data.normal_(mean=0.0, std=0.02)
# model.model.router[4].weight.data.normal_(mean=0.0, std=0.02)
# model.model.router[6].weight.data.normal_(mean=0.0, std=0.02)

# model.model.router[0].bias.data.zero_()
# model.model.router[2].bias.data.zero_()
# model.model.router[4].bias.data.zero_()

# Save the modified model
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print(model)