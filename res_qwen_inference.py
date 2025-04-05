import torch
from src.open_r1.ResQwen import Qwen2ForCausalLM
from transformers import AutoTokenizer

model_name = "ZMC2019/Qwen1.5B-L44-Flat-Zero"
device = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = Qwen2ForCausalLM.from_pretrained(model_name, device_map=device)

messages = [
    {"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors='pt').to(device)
output_ids = llm.generate(**inputs, max_new_tokens=4096, do_sample=True, temperature=0.6, top_p=0.95)


output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(output_text)
