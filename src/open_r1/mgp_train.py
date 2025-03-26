from SparseQwen import Qwen2ForCausalLM

model_name = "/home/zhuominc/conv/open-r1/DeepSeek-R1-7B"

llm = Qwen2ForCausalLM.from_pretrained(model_name)
