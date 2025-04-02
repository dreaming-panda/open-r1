from ResQwen import Qwen2ForCausalLM, Qwen2Config

config = Qwen2Config.from_pretrained("Qwen1.5B-L44-Flat")
config.start_conv_idx=16
config.end_conv_idx=32
config.num_conv=0

    
model = Qwen2ForCausalLM.from_pretrained("Qwen1.5B-L44-Flat", config=config)
model.zero_init()