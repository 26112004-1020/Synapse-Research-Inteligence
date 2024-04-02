!pip -q install auto-gptq
!pip -q install optimum
!pip -q install bitsandbytes
!pip -q install einops
!pip install transformers sacrebleu torch  

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from time import time


def generate_text_and_perplexity(model, tokenizer, prompt):
  """Function to generate text and estimate perplexity"""
  start_mem = torch.cuda.memory_allocated() / 1024**3 
  input_ids = tokenizer.encode(prompt, return_tensors="pt")
  output = model.generate(input_ids, max_length=100, num_return_sequences=1)
  generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
  end_mem = torch.cuda.memory_allocated() / 1024**3 
  memory_used = end_mem - start_mem
  with torch.no_grad():
    logits = model(input_ids).logits
    perplexity = torch.exp(torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), output.view(-1))).item()
  return generated_text, perplexity, memory_used


def calculate_bleu_score(references, predictions):
  """Function to calculate BLEU score (requires sacrebleu library)"""
  try:
    from sacrebleu import corpus_bleu
    return corpus_bleu(references, predictions)
  except ImportError:
    print("sacrebleu library not installed. BLEU score calculation skipped.")
    return None


def measure_latency_and_generate_text(pipeline, prompt):
  """Function to measure latency and generate text using pipeline"""
  start_time = time.time()
  output = pipeline(prompt, max_new_tokens=128, do_sample=True, top_p=0.9, temperature=0.9)
  end_time = time.time()
  latency_per_token_in_ms = ((end_time - start_time) / len(pipeline.tokenizer(prompt)["input_ids"])) * 1000
  return output[0]["generated_text"][len(prompt):], round(latency_per_token_in_ms, 2)


def print_analysis_results(original_results, quantized_results):
  """Function to print analysis results"""
  print("Original Model Results:")
  print("Generated Text:", original_results['generated_text'])
  print("Perplexity:", original_results['perplexity'])
  print("Memory Used (GB):", original_results['memory_used'])
  print("BLEU Score:", original_results['bleu_score'])
  if 'latency' in original_results:  
    print("Latency:", original_results['latency'])
  print()
  print("Quantized Model Results:")
  print("Generated Text:", quantized_results['generated_text'])
  print("Perplexity:", quantized_results['perplexity'])
  print("Memory Used (GB):", quantized_results['memory_used'])
  print("BLEU Score:", quantized_results['bleu_score'])
  if 'latency' in quantized_results:  
    print("Latency:", quantized_results['latency'])

general_prompt = "Explain support vector machines."

original_model = AutoModelForCausalLM.from_pretrained('microsoft/phi-2')
original_tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2')

quantized_model = AutoModelForCausalLM.from_pretrained('AVMLegend/Quantised-Phi-2')
quantized_tokenizer = AutoTokenizer.from_pretrained('AVMLegend/Quantised-Phi-2')

if original_model and original_tokenizer:
  original_generated_text, original_perplexity, original_memory_used = generate_text_and_perplexity(original_model, original_tokenizer, general_prompt)
else:
    print("Original model/tokenizer not loaded. Skipping original model evaluation.")
    original_generated_text = None
    original_perplexity = None
    original_memory_used = None

if quantized_model and quantized_tokenizer:
  quantized_generated_text, quantized_perplexity, quantized_memory_used = generate_text_and_perplexity(quantized_model, quantized_tokenizer, general_prompt)
else:
  print("Quantized model/tokenizer not loaded. Skipping quantized model evaluation.")
  quantized_generated_text = None
  quantized_perplexity = None
  quantized_memory_used = None

original_bleu_score = calculate_bleu_score([[general_prompt.split()]], [original_generated_text.split()] if original_generated_text else [])

quantized_bleu_score = calculate_bleu_score([[general_prompt.split()]], [quantized_generated_text.split()] if quantized_generated_text else [])

original_results = {
  'generated_text': original_generated_text,
  'perplexity': original_perplexity,
  'memory_used': original_memory_used,
  'bleu_score': original_bleu_score
}

quantized_results = {
  'generated_text': quantized_generated_text,
  'perplexity': quantized_perplexity,
  'memory_used': quantized_memory_used,
  'bleu_score': quantized_bleu_score
}

original_latency = None
quantized_latency = None

print_analysis_results(original_results, quantized_results)

original_latency = measure_latency_and_generate_text(original_pipeline, general_prompt)
quantized_latency = measure_latency_and_generate_text(quantized_pipeline, general_prompt)

if original_latency:
  print("Original Model Latency:", original_latency)
if quantized_latency:
  print("Quantized Model Latency:", quantized_latency)
