import evaluate_llm as eval
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "./Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

eval.measure_test_accuracy(model, tokenizer, "./gsm8k_saved", 'cuda', display=True)