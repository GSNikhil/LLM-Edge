from datasets import load_dataset, load_from_disk
from rouge_score import rouge_scorer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Logging
from datetime import datetime

def generate_answer(model, tokenizer, question, device):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)  # Adjust max_length if needed
    
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

def measure_test_accuracy(model, tokenizer, dataset_path, device):
    # Make the model eval
    model.eval()

    # Load Dataset
    dataset = load_from_disk(dataset_path)
    test_data = dataset['test']
    print("Len of Test Data: ", len(test_data))

    # Evaluate - Basic
    correct = 0
    total = len(test_data)

    # ROUGE Scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

    # Open File for Logging
    os.makedirs("./logs", exist_ok=True)
    log_file = open(f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w")
    log_file.write("Sample\tMatch\tRouge1\tRouge2\tRougeL\n")

    match = 0
    with torch.no_grad():
        for i, sample in enumerate(test_data):
            question = sample["question"]
            ground_truth = sample["answer"]
            
            model_answer = generate_answer(model, tokenizer, question, device)
            
            if ground_truth.strip() in model_answer:  # Basic match check
                match = 1
                correct += 1

            # Compute ROUGE scores
            scores = scorer.score(ground_truth, model_answer)

            rouge1_scores.append(scores["rouge1"].fmeasure)
            rouge2_scores.append(scores["rouge2"].fmeasure)
            rougeL_scores.append(scores["rougeL"].fmeasure)

            log_file.write(f"{i}\t{match:.2f}\t{scores['rouge1'].fmeasure:.4f}\t{scores['rouge2'].fmeasure:.4f}\t{scores['rougeL'].fmeasure:.4f}\n")
            match = 0
            if i == 20:
                print(i)
                break

    accuracy = correct / total * 100
    
    # Calculate Average ROUGE Scores
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

    print(f"Model Accuracy on GSM8K: {accuracy:.2f}%")
    print(f"Average ROUGE-1: {avg_rouge1:.4f}")
    print(f"Average ROUGE-2: {avg_rouge2:.4f}")
    print(f"Average ROUGE-L: {avg_rougeL:.4f}")

    log_file.write(f"Avg\t{accuracy:.2f}\t{avg_rouge1:.4f}\t{avg_rouge2:.4f}\t{avg_rougeL:.4f}\n")
    return accuracy