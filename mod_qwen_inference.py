import torch
from src.open_r1.MoDQwen import Qwen2ForCausalLM
from transformers import AutoTokenizer

model_name = "ZMC2019/Qwen1.5B-S12E16-MoD4"
device = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = Qwen2ForCausalLM.from_pretrained(model_name, device_map=device)

messages = [
    {"role": "user", "content": r"Define\[p = \sum_{k = 1}^\infty \frac{1}{k^2} \quad \text{and} \quad q = \sum_{k = 1}^\infty \frac{1}{k^3}.\]Find a way to write\[\sum_{j = 1}^\infty \sum_{k = 1}^\infty \frac{1}{(j + k)^3}\]in terms of $p$ and $q.$"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors='pt').to(device)
output_ids = llm.generate(**inputs, max_new_tokens=4096, do_sample=True, temperature=0.6, top_p=0.95)


output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(output_text)
