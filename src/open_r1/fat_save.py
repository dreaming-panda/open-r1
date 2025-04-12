from transformers import Qwen2Config, AutoTokenizer, Qwen2ForCausalLM
import torch
import argparse
import copy
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-7B-Instruct", help='model')
parser.add_argument('--output_dir', type=str, default="Qwen2.5-7B-Expand", help='output_dir')
args = parser.parse_args()

# Load model and tokenizer
config = Qwen2Config.from_pretrained(args.model)
model = Qwen2ForCausalLM.from_pretrained(args.model, config=config)
tokenizer = AutoTokenizer.from_pretrained(args.model)

# Double the attention head settings
old_config = copy.deepcopy(config)

config.num_attention_heads *= 2
config.num_key_value_heads *= 2
config.head_dim = 128
# Create a new model with the updated config
new_model = Qwen2ForCausalLM(config)
new_model = new_model.to(torch.bfloat16)

# Loop through transformer layers and update attention weights
for i in range(len(model.model.layers)):
    old_layer = model.model.layers[i]
    new_layer = new_model.model.layers[i]

    # --- QKV Projections ---
    import torch
    import torch.nn as nn

    def expand_linear(old_linear: nn.Linear, multiplier=2, dropout_prob=0.1, noise_std=0.01):
        out_features = old_linear.out_features * multiplier
        new_linear = nn.Linear(old_linear.in_features, out_features, bias=old_linear.bias is not None)

        with torch.no_grad():
        
            new_linear.weight[:old_linear.out_features] = old_linear.weight

            perturbed_weight = old_linear.weight.clone() + noise_std * torch.randn_like(old_linear.weight)
            new_linear.weight[old_linear.out_features:] = perturbed_weight

            if old_linear.bias is not None:
                original_bias = old_linear.bias.clone()
                mask = (torch.rand_like(original_bias) > dropout_prob).float()
                dropped_bias = original_bias * mask

                new_linear.bias[:old_linear.out_features] = original_bias
                new_linear.bias[old_linear.out_features:] = dropped_bias

        return new_linear


    new_layer.self_attn.q_proj = expand_linear(old_layer.self_attn.q_proj)
    new_layer.self_attn.k_proj = expand_linear(old_layer.self_attn.k_proj)
    new_layer.self_attn.v_proj = expand_linear(old_layer.self_attn.v_proj)

    # --- Output Projection ---
    def expand_output_linear(old_linear, multiplier=2):
        in_features = old_linear.in_features * multiplier
        new_linear = torch.nn.Linear(in_features, old_linear.out_features, bias=old_linear.bias is not None, dtype=torch.bfloat16)
        with torch.no_grad():
            new_linear.weight[:, :old_linear.in_features] = old_linear.weight
            new_linear.weight[:, old_linear.in_features:].zero_()
            if old_linear.bias is not None:
                new_linear.bias = old_linear.bias
        return new_linear

    new_layer.self_attn.o_proj = expand_output_linear(old_layer.self_attn.o_proj)

    # Copy rest of the layer
    new_layer.mlp.load_state_dict(old_layer.mlp.state_dict())
    new_layer.input_layernorm.load_state_dict(old_layer.input_layernorm.state_dict())
    new_layer.post_attention_layernorm.load_state_dict(old_layer.post_attention_layernorm.state_dict())

# Copy embeddings and final ln
new_model.lm_head.load_state_dict(model.lm_head.state_dict())
new_model.model.norm.load_state_dict(model.model.norm.state_dict())
new_model.model.embed_tokens.load_state_dict(model.model.embed_tokens.state_dict())
new_model.model.rotary_emb.load_state_dict(model.model.rotary_emb.state_dict())

# Save new model
new_model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print(new_model)
print(f"Expanded model saved to {args.output_dir}")
