import os 
import utils.visulaiser as visulaiser
from datasets import load_dataset, load_from_disk

from torch import nn
from tqdm import tqdm
import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
import re
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim import AdamW
import torch.nn as nn
from torchvision.transforms import v2
from rouge_score import rouge_scorer
# Logging
from datetime import datetime
import time

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#############################
# Download Model or Load Model
#############################

# model_name = "wzzju/Qwen2.5-1.5B-GRPO-GSM8K"
model_name = "Qwen/Qwen2-Math-1.5B-Instruct"
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
    base_model.save_pretrained(f"./{model_name.split('/')[-1]}")
    tokenizer.save_pretrained(f"./{model_name.split('/')[-1]}")

print("Done")

#################################
# Load or Download GSM8k Dataset
#################################

dataset_name = "gsm8k"

if os.path.isdir(f"./{dataset_name}"):
    print("Using Pre-Downloaded Dataset")
    dataset = load_from_disk("./gsm8k")
else:
    print("Downloading Dataset")
    dataset = load_dataset("gsm8k", "main")
    print(f"Saving Dataset to ./{dataset_name}")
    dataset.save_to_disk("./gsm8k")
    
print("Dataset Loaded")

if os.path.isdir(f"./{dataset_name}_tokenized"):
    tokenized_data = load_from_disk(f"./{dataset_name}_tokenized")
else:
    def extract_final_answer(answer):
        """
        Extracts only the numerical value after '####' in the answer field.
        """
        match = re.search(r"####\s*([\d\.]+)", answer)  # Match number after ####
        return float(match.group(1)) if match else 0  # Return extracted number
    
    # Process training and test sets
    for split in ["train", "test"]:
        dataset[split] = dataset[split].map(lambda example: {
            "original_answer": example['answer'],
            "question": example["question"],
            # "answer": tokenizer(extract_final_answer(example["answer"]),
            #                     padding='max_length',
            #                     truncation=True,
            #                     max_length=16,
            #                     return_tensors='pt').to(device),
            "answer": extract_final_answer(example["answer"]),
        })

    def format_example(example):
        # print(example)
        return f"You are a math expert. Now answer this question - " + example["question"] + " Your answer should only contain the final answer as a number. Print final answer here: "
        # return f"Question: YOU ARE A EXPERT AT MATH. NOW ANSWER THIS QUESTION - {example['question']}. REPLY JUST THE FINAL ANSWER AS A NUMBER. Answer: "

    # Tokenize data
    def preprocess_function(examples):
        texts = format_example(examples)
        tokens = tokenizer(texts, 
                        padding="max_length", 
                        truncation=True, 
                        max_length=128, 
                        return_tensors="pt")
        return tokens

    tokenized_data = dataset.map(preprocess_function, batched=False)
    # Save processed dataset
    tokenized_data.save_to_disk("./gsm8k_tokenized")

# Print an example to verify
# print(tokenized_data["train"][0])

# Split into train and test sets
# train_data = tokenized_data["train"]
test_data = tokenized_data["test"]

# small_train_dataset = train_data.shuffle(seed=42).select(range(1000)) # Loading only 1000
small_eval_dataset = test_data.shuffle(seed=42).select(range(200))

# train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=1)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=1)

def generate_answer(model, tokenizer, sample, device):
    batch = {}
    for k, v in sample.items():
        if k != "question" and k != "answer" and k != 'original_answer':
            batch[k] = torch.tensor(v).to(device)
    
    output = model.generate(**batch, max_new_tokens=16, do_sample=False,
                            # do_sample = True,
                            # temperature = 0.3,
                            )
    output = tokenizer.decode(output[0][len(batch['input_ids'][0]):], skip_special_tokens=True) 

    return output

def measure_test_accuracy(model, tokenizer, dataloader, device, display=False):
    # Make the model eval
    model.eval()
    model = model.to(device)

    total = len(dataloader)
    num_training_steps = total
    progress_bar = tqdm(range(num_training_steps))

    # Evaluate - Basic
    accuracy_log = []
    accuracy = 0
    

    # ROUGE Scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

    # Open File for Logging
    os.makedirs("./logs", exist_ok=True)
    log_file = open(f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w")
    log_file.write("Sample\tMatch\tRouge1\tRouge2\tRougeL\n")

    with torch.no_grad():
        for i, sample in enumerate(dataloader):

            output = generate_answer(model, tokenizer, sample, device)

            match = re.search(r"\s*([\d\.]+)", output)  # Match number after ####
            generated_answer = float(match.group(1)) if match else 0  # Return extracted number
            
            accuracy = (generated_answer == sample['answer'].item())
            accuracy_log.append(accuracy)

            # Compute ROUGE scores
            scores = scorer.score(sample['original_answer'][0], output)

            rouge1_scores.append(scores["rouge1"].fmeasure)
            rouge2_scores.append(scores["rouge2"].fmeasure)
            rougeL_scores.append(scores["rougeL"].fmeasure)

            if display:
                print(f"Example {i+1}:\n")
                print(f"Input: {sample['question']}\n")
                print(f"Generated Answer: {output}\n")
                print(f"Target Output: {sample['answer'].item()}\n")
                print(f"Output Answer: {generated_answer}")
                print("-" * 50)

            log_file.write(f"{i}\t{accuracy:.2f}\t{scores['rouge1'].fmeasure:.4f}\t{scores['rouge2'].fmeasure:.4f}\t{scores['rougeL'].fmeasure:.4f}\n")
            if i % 100 == 0:
                print(f"{i}\t{accuracy:.2f}\t{scores['rouge1'].fmeasure:.4f}\t{scores['rouge2'].fmeasure:.4f}\t{scores['rougeL'].fmeasure:.4f}\n")
            
            progress_bar.update(1)

    accuracy = np.sum(accuracy_log) / total * 100
    
    # Calculate Average ROUGE Scores
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

    print(f"Model Accuracy on GSM8K: {accuracy:.2f}%")
    print(f"Average ROUGE-1: {avg_rouge1:.4f}")
    print(f"Average ROUGE-2: {avg_rouge2:.4f}")
    print(f"Average ROUGE-L: {avg_rougeL:.4f}")

    log_file.write(f"Avg\t{accuracy:.2f}\t{avg_rouge1:.4f}\t{avg_rouge2:.4f}\t{avg_rougeL:.4f}\n")

start_time = time.perf_counter()
measure_test_accuracy(base_model, tokenizer, eval_dataloader, device)
end_time = time.perf_counter()

execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.6f} seconds")