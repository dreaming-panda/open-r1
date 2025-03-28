import wandb
from SparseQwen import Qwen2ForCausalLM
import datasets
from transformers import AutoTokenizer
from tqdm import tqdm
import torch

model_name = "/home/zhuominc/conv/open-r1/DeepSeek-R1-7B"

# === WandB 初始化 ===
wandb.init(
    project="openr1-math-training",   # 项目名，可自定义
    name="ds-7b-sparse-finetune-layer20",     # 运行名
    config={
        "model_name": model_name,
        "lr": 1e-4,
        "accum_steps": 64,
        "max_length": 4096
    }
)

device = torch.device("cuda:2")

# Load model in bfloat16
llm = Qwen2ForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, _attn_implementation="eager")
llm.train()
llm.init_training(layer_idx=20)

trainable_params = sum(p.numel() for p in llm.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in llm.parameters())
print(f"Trainable parameters: {trainable_params} / {total_params}")
wandb.log({"trainable_params": trainable_params, "total_params": total_params})

# Move to GPU
llm.to(device)

# Dataset & tokenizer
data = datasets.load_dataset("open-r1/OpenR1-Math-220k", split="train")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Optimizer
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, llm.parameters()), lr=1e-4)

# Gradient Accumulation
accum_steps = 64
optimizer.zero_grad()

# Training loop
for idx in tqdm(range(len(data))):
    # Prepare inputs
    inputs = tokenizer.apply_chat_template(
        data[idx]["messages"],
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        max_length=2048,
        padding="max_length",
        truncation=True
    ).to(device)
    
    outputs = llm(**inputs)
    loss = outputs.attention_loss
    recall = outputs.recall
    (loss / accum_steps).backward()

    if (idx + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

        # === WandB logging ===
        wandb.log({
            "step": idx + 1,
            "loss": loss.item(),
            "recall": recall[0],
            "lr": optimizer.param_groups[0]['lr']
        })
    
# Final step if number of steps not divisible by accum_steps
if (idx + 1) % accum_steps != 0:
    optimizer.step()
    optimizer.zero_grad()
    wandb.log({
        "step": idx + 1,
        "loss": loss.item(),
        "lr": optimizer.param_groups[0]['lr']
    })
