from .ConvQwen import Qwen2ForCausalLM, Qwen2Config
from transformers import AutoTokenizer
import torch
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = Qwen2Config.from_pretrained(model_name)
config.start_conv_idx=12
config.end_conv_idx=16
config.num_conv=2
config._attn_implementation = "flash_attention_2"
model = Qwen2ForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16, 
    device_map="cuda:0", config=config)

question = r"Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$"

inputs = tokenizer(question, return_tensors="pt").to("cuda:0")

outputs = model.generate(**inputs, max_new_tokens=512)

output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)
