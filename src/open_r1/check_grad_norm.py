import transformers
import datasets
from transformers import AutoTokenizer, Qwen2ForCausalLM
import torch
import copy
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from tqdm import tqdm
# 加载数据
data = datasets.load_dataset("open-r1/OpenR1-Math-220k", split="train")
model_name = "ZMC2019/Qwen1.5B-L36-FlatV2"
device = "cuda:0"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 模型加载，开启 gradient checkpointing
llm: Qwen2ForCausalLM = AutoLigerKernelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    torch_dtype=torch.float32,
)

llm.gradient_checkpointing_enable()
llm.train()

# 假设有 28 层
num_layers = llm.config.num_hidden_layers
layer_norm_sums = [0.0 for _ in range(num_layers)]
layer_norm_counts = [0 for _ in range(num_layers)]
total_loss = 0.0

# 遍历前100条数据
for idx in tqdm(range(25)):
    # 准备输入
    inputs = tokenizer.apply_chat_template(
        data[idx]["messages"],
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)

    inputs["labels"] = copy.deepcopy(inputs["input_ids"])

    # 清空梯度
    llm.zero_grad()

    # forward & backward
    outputs = llm(**inputs)
    loss = outputs.loss
    total_loss += loss.item()
    loss.backward()

    # 统计每层梯度范数
    for i in range(num_layers):
        layer = llm.model.layers[i]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            layer.parameters(), max_norm=1.0
        ).item()

        layer_norm_sums[i] += grad_norm
        layer_norm_counts[i] += 1


avg_loss = total_loss / 100
print(f"\n[Loss]: {avg_loss:.4f}")

print("\n[Grad Norm]")
for i in range(num_layers):
    avg_norm = layer_norm_sums[i] / layer_norm_counts[i]
    print(f"Layer {i:02d}: Avg Grad Norm = {avg_norm:.4f}")
