from transformers import Qwen2Config, AutoTokenizer, Qwen2ForCausalLM
import torch
import argparse
import copy
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-7B-Instruct", help='model')
parser.add_argument('--output_dir', type=str, default="Qwen2.5-7B-Expand", help='output_dir')
parser.add_argument('--multiplier', type=int, default=2, help='multiplier')
args = parser.parse_args()

# Load model and tokenizer
config = Qwen2Config.from_pretrained(args.model)
model = Qwen2ForCausalLM.from_pretrained(args.model, config=config)

tokenizer = AutoTokenizer.from_pretrained(args.model)

# Double the attention head settings
old_config = copy.deepcopy(config)

config.num_attention_heads *= args.multiplier
config.num_key_value_heads *= args.multiplier
config.head_dim = 128
# Create a new model with the updated config
new_model = Qwen2ForCausalLM(config)
new_model = new_model.to(torch.bfloat16)
total_params = sum(p.numel() for p in new_model.parameters())
print(f"Total Parameters: {total_params / 1e9:.2f}B")

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
            # Expand weights
            for i in range(multiplier):
                new_linear.weight[i * old_linear.out_features:(i + 1) * old_linear.out_features] = \
                    old_linear.weight.clone()
            
            # Add noise only to the expanded part (not the first block)
            if multiplier > 1:
                expanded_weights = new_linear.weight[old_linear.out_features:]
                new_linear.weight[old_linear.out_features:] = expanded_weights + noise_std * torch.randn_like(expanded_weights)

            # Expand bias if exists
            if old_linear.bias is not None:
                original_bias = old_linear.bias.clone()
                new_bias = []

                for i in range(multiplier):
                    if i == 0:
                        # Keep original bias
                        new_bias.append(original_bias)
                    else:
                        # Apply dropout + noise to bias copy
                        mask = (torch.rand_like(original_bias) > dropout_prob).float()
                        bias_i = original_bias * mask
                        bias_i += noise_std * torch.randn_like(bias_i)
                        new_bias.append(bias_i)

                new_linear.bias[:] = torch.cat(new_bias, dim=0)

        return new_linear

    new_layer.self_attn.q_proj = expand_linear(old_layer.self_attn.q_proj, multiplier=args.multiplier)
    new_layer.self_attn.k_proj = expand_linear(old_layer.self_attn.k_proj, multiplier=args.multiplier)
    new_layer.self_attn.v_proj = expand_linear(old_layer.self_attn.v_proj, multiplier=args.multiplier)

    # --- Output Projection ---
    def expand_output_linear(old_linear, multiplier=2):
        in_features = old_linear.in_features * multiplier
        new_linear = torch.nn.Linear(in_features, old_linear.out_features, bias=old_linear.bias is not None, dtype=torch.bfloat16)
        with torch.no_grad():
            new_linear.weight[:, :old_linear.in_features] = old_linear.weight
            new_linear.weight[:, old_linear.in_features:].zero_()
        
        return new_linear

    new_layer.self_attn.o_proj = expand_output_linear(old_layer.self_attn.o_proj, multiplier=args.multiplier)

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
