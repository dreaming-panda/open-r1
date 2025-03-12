#from open_r1.ConvQwen import Qwen2ForCausalLM, Qwen2Config
from transformers import AutoTokenizer, Qwen2ForCausalLM, TextStreamer
import torch
from flash_attn import flash_attn_varlen_func
model_name = "/home/zhuominc/OpenEval/open-r1/src/open_r1/Qwen-1.5B-Instruct-Conv-Unrolled"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen2ForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16, 
    device_map="cuda:0", _attn_implementation="flash_attention_2"
)

MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

question = r"""Define
\[p = \sum_{k = 1}^\infty \frac{1}{k^2} \quad \text{and} \quad q = \sum_{k = 1}^\infty \frac{1}{k^3}.\]Find a way to write
\[\sum_{j = 1}^\infty \sum_{k = 1}^\infty \frac{1}{(j + k)^3}\]in terms of $p$ and $q.$"""
question = MATH_QUERY_TEMPLATE.format(Question=question)

conversation = [
    {"role": "user", "content": question}
]

question = tokenizer.apply_chat_template(conversation=conversation, add_generation_prompt=True, tokenize=False)

inputs = tokenizer(question, return_tensors="pt").to("cuda:0")

# Initialize the streamer
streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=False)

# Generate output using streamer
model.generate(**inputs, max_new_tokens=30768, streamer=streamer, top_p=0.95, temperature=0.6)
