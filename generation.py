from transformers import Qwen2ForCausalLM, AutoTokenizer, TextStreamer
import torch

# 加载模型和分词器
model_path = "/home/zhuominc/conv/open-r1/Qwen2.5-7B-QO-Expand"
model = Qwen2ForCausalLM.from_pretrained(model_path, device_map="cuda:0", torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


user_input = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": user_input}],
    tokenize=False,
    add_generation_prompt=True
)

# 编码输入并生成回复
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 生成（也可用 do_sample=True 等参数）
_ = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    streamer=streamer
)
