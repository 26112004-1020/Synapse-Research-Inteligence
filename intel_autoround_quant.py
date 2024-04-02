import ipex
from transformers import AutoModelForCausalLM, AutoTokenizer

device = ipex.DEVICE 
dtype = ipex.fp16  

model_name = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device, dtype=dtype) 
tokenizer = AutoTokenizer.from_pretrained(model_name)

if model.supports_ipex_quantization and ipex.has_ipex_quantization: 
    bits, group_size, sym = 4, 128, False
    autoround = model.quantize(bits=bits, group_size=group_size, symmetric=sym)
    output_dir = "./tmp_autoround"
    autoround.save_quantized(output_dir)
