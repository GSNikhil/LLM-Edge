from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def get_model(model_name = "Qwen/Qwen2-Math-1.5B-Instruct", save_model=False):
    #############################
    # Download Model or Load Model
    #############################

    # model_name = "Qwen/Qwen2-Math-1.5B-Instruct"
    # model_name = "wzzju/Qwen2.5-1.5B-GRPO-GSM8K"
    model_pth = f"./{model_name.split('/')[-1]}"

    if os.path.isdir(model_pth):
        print("Using Pre-Downloaded Model and Tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_pth, local_files_only=True, padding_side="left")
        base_model = AutoModelForCausalLM.from_pretrained(model_pth)
    else:
        print("Downloading Model and Tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        base_model = AutoModelForCausalLM.from_pretrained(model_name)

        # Save model and tokenizer to the current directory
        print(f"Saving Model to {model_pth}")
        if save_model:
            base_model.save_pretrained(f"./{model_name.split('/')[-1]}")
            tokenizer.save_pretrained(f"./{model_name.split('/')[-1]}")

    print("Model and Tokenizer Loaded")
    return base_model, tokenizer

def get_dataset(dataset_name = "gsm8k"):
    #################################
    # Load or Download GSM8k Dataset
    #################################
    if os.path.isdir(f"./{dataset_name}"):
        print("Using Pre-Downloaded Dataset")
        dataset = load_from_disk("./gsm8k")
    else:
        print("Downloading Dataset")
        dataset = load_dataset("gsm8k", "main")
        print(f"Saving Dataset to ./{dataset_name}")
        dataset.save_to_disk("./gsm8k")
        
    print("Dataset Loaded")
    return dataset