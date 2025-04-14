from MTPQwen3 import Qwen2ForCausalLM, Qwen2Config
from transformers import AutoTokenizer
import torch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-1.5B-Instruct",help='model')
parser.add_argument('--start', type=int, default=24,help='start_conv_idx')
parser.add_argument('--end', type=int, default=28,help='end_conv_idx')
parser.add_argument('--num_conv', type=int, default=2,help='num_conv')
parser.add_argument('--lora_dim', type=int, default=32,help='num_conv')
parser.add_argument('--output_dir', type=str, default="Qwen1.5B-LoraMTP",help='output_dir')
args = parser.parse_args()
name = args.model
config = Qwen2Config.from_pretrained(name)
config.lora_dim = args.lora_dim
config.num_conv = args.num_conv
config.start_conv_idx = args.start
config.end_conv_idx = args.end

llm = Qwen2ForCausalLM.from_pretrained(name, config=config)
llm = llm.to(torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.model)

for layer_idx in range(config.start_conv_idx, config.end_conv_idx):
    llm.model.layers[layer_idx].self_attn.adapter_q[0].weight.data.normal_(mean=0.0, std=0.02)
    llm.model.layers[layer_idx].self_attn.adapter_q[0].bias.data.zero_()
    llm.model.layers[layer_idx].self_attn.adapter_q[1].weight.data.zero_()
    llm.model.layers[layer_idx].self_attn.adapter_q[1].bias.data.zero_()
    llm.model.layers[layer_idx].self_attn.adapter_v[0].weight.data.normal_(mean=0.0, std=0.02)
    llm.model.layers[layer_idx].self_attn.adapter_v[0].bias.data.zero_()
    llm.model.layers[layer_idx].self_attn.adapter_v[1].weight.data.zero_()
    llm.model.layers[layer_idx].self_attn.adapter_v[1].bias.data.zero_()


llm.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print(llm)