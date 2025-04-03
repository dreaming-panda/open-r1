from transformers import Qwen2Config, AutoTokenizer
from MoDQwen import Qwen2ForCausalLM
import torch
import copy
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-1.5B-Instruct",help='model')
parser.add_argument('--start', type=int, default=12,help='start_conv_idx')
parser.add_argument('--end', type=int, default=16,help='end_conv_idx')
parser.add_argument('--num_conv', type=int, default=6,help='num_conv')
parser.add_argument('--output_dir', type=str, default="Qwen1.5B-S12E16-MoD",help='output_dir')
args = parser.parse_args()
# Load model and configuration
config = Qwen2Config.from_pretrained(args.model)
config.start_conv_idx = args.start
config.end_conv_idx = args.end
config.num_conv = args.num_conv

model = Qwen2ForCausalLM.from_pretrained(args.model, config=config)
tokenizer = AutoTokenizer.from_pretrained(args.model)
model.model.router[0].weight.data.normal_(mean=0.0, std=0.02)
model.model.router[2].weight.data.normal_(mean=0.0, std=0.02)
model.model.router[4].weight.data.normal_(mean=0.0, std=0.02)
model.model.router[6].weight.data.normal_(mean=0.0, std=0.02)
# Save the modified model
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print(model)
