from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Datasets
gsm8k = load_dataset("gsm8k", "main")  # Load the main test set
gsm8k.save_to_disk("./gsm8k_saved")
print("Dataset downloaded to ./gsm8k_saved")

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save model and tokenizer to the current directory
model.save_pretrained("./Qwen2.5-0.5B")
tokenizer.save_pretrained("./Qwen2.5-0.5B")

print("Model and tokenizer saved to ./Qwen2.5-0.5B")