from transformers import Qwen2Config, AutoTokenizer
from HPQwen import Qwen2ForCausalLM
import torch
import copy
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-7B-Instruct",help='model')
parser.add_argument('--output_dir', type=str, default="Qwen7B-HP-AMP",help='output_dir')
args = parser.parse_args()
# Load model and configuration
config = Qwen2Config.from_pretrained(args.model)


model = Qwen2ForCausalLM.from_pretrained(args.model, config=config)
tokenizer = AutoTokenizer.from_pretrained(args.model)
for layer in model.model.layers:
    layer.self_attn.amp_q_proj.load_state_dict(layer.self_attn.q_proj.state_dict())
    layer.self_attn.amp_k_proj.load_state_dict(layer.self_attn.k_proj.state_dict())
    layer.self_attn.amp_v_proj.load_state_dict(layer.self_attn.v_proj.state_dict())
    layer.self_attn.amp_scaler.data = torch.zeros(1, 1, config.num_attention_heads, 1) + 0.5

model = model.to(torch.bfloat16)
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print(model)