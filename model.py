from transformers import AutoModelForCausalLM, AutoTokenizer
quantized_model_dir = "AVMLegend/Quantised-Phi-2"

model = AutoModelForCausalLM.from_pretrained(quantized_model_dir, device_map="cpu", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir)

prompt = "Explain logistic regression"
inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_new_tokens=200)
text = tokenizer.batch_decode(outputs)[0]

last_stop_index = text.rfind(".")
if last_stop_index != -1:
  text = text[:last_stop_index + 1]
print(text)
