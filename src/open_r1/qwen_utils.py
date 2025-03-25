import transformers
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

def get_optimizer_parameter_group_qwen(
    model,
    start_adj_lr_idx, 
    end_adj_lr_idx,
    adj_lr,
    weight_decay,
    learning_rate):

    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS, ["bias", "layernorm", "rmsnorm"])
    decay_parameters = set(decay_parameters)

    
    group_decay_adj = []
    group_nodecay_adj = []
    group_decay_base = []
    group_nodecay_base = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        
        use_adj = False
        if "model.layers." in name:
            parts = name.split(".")
            try:
                block_idx = int(parts[2])
                if start_adj_lr_idx <= block_idx < end_adj_lr_idx:
                    use_adj = True
            except:
                pass

        use_decay = name in decay_parameters

        if use_adj and use_decay:
            group_decay_adj.append(param)
        elif use_adj and not use_decay:
            group_nodecay_adj.append(param)
        elif not use_adj and use_decay:
            group_decay_base.append(param)
        else:
            group_nodecay_base.append(param)

    optimizer_dict = [
        {"params": group_decay_adj, "lr": adj_lr, "weight_decay": weight_decay},
        {"params": group_nodecay_adj, "lr": adj_lr, "weight_decay": 0.0},
        {"params": group_decay_base, "lr": learning_rate, "weight_decay": weight_decay},
        {"params": group_nodecay_base, "lr": learning_rate, "weight_decay": 0.0},
    ]
    
    return {"optimizer_dict": optimizer_dict, 'betas': (0.9, 0.999), 'eps': 1e-08}
