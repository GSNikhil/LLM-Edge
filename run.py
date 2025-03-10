import evaluate_llm as eval
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch

model_name = "./Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

start_time = time.perf_counter()
eval.measure_test_accuracy(model, tokenizer, "./gsm8k_saved", device, display=True)
end_time = time.perf_counter()

execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.6f} seconds")
