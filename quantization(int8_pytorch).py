import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.quantization.quantize_fx import prepare_fx, convert_fx

qconfig_mapping = get_default_qconfig_mapping(qengine='x86')

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
model.eval()

def generate_input(tokenizer, sequence_length=10, input_dim=768):
 input_ids = torch.randint(0, tokenizer.vocab_size, (1, sequence_length), dtype=torch.long)
 x = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
 x = tokenizer(x, return_tensors="pt")
 x = x["input_ids"].to(memory_format=torch.channels_last)
 return x

x = generate_input(tokenizer)

prepared_model = prepare_fx(model, qconfig_mapping, example_inputs=x)

def calibrate(model, data_loader):
 model.to("cpu")
 model.eval()
 with torch.no_grad():
   for data, _ in data_loader:
     model(data)

# Implement your calibration function here

quantized_model = convert_fx(prepared_model)

def inference(quantized_model, input_tensor):
 quantized_model.to("cpu")
 with torch.no_grad():
   model_output = quantized_model(input_tensor)
 return model_output

inference_output = inference(quantized_model, x.clone())
print(inference_output)
