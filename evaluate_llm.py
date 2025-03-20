from rouge_score import rouge_scorer
import torch
import os
import re
from tqdm.auto import tqdm
import numpy as np

# Logging
from datetime import datetime

def generate_answer(model, tokenizer, sample, device):
    batch = {}
    for k, v in sample.items():
        if k != "question" and k != "answer" and k != 'original_answer':
            batch[k] = torch.tensor(v).to(device)
    
    output = model.generate(**batch, max_new_tokens=16, do_sample=False)
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

            match = re.search(r"\s*([\d]+)", output)  # Match number after ####
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