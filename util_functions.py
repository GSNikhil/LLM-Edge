import tqdm
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch import nn
from datasets import load_dataset, load_from_disk
from download_datasets_models import get_dataset

def get_model_size(model: nn.Module, data_width=16, group_size=-1):

    if group_size != -1:
        data_width += (16 + 4) / group_size

    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    

    Byte = 8
    KiB = 1024 * Byte
    MiB = 1024 * KiB
    GiB = 1024 * MiB

    return (num_elements * data_width) / MiB

# Get Model Perplexity
def evaluate_perplexity(model, tokenizer):
    dataset = get_dataset()

    testenc = dataset['test']
    testenc = tokenizer("\n\n".join(testenc['question']), return_tensors='pt')

    testenc = testenc.input_ids.to(model.device)
    nsamples = 10
    model = model.eval()

    nlls = []
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * 2048):((i + 1) * 2048)].to(model.device)
        with torch.no_grad():
            lm_logits = model(batch, temperature=0.3).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * 2048):((i + 1) * 2048)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * 2048
        nlls.append(neg_log_likelihood)

    return torch.exp(torch.stack(nlls).sum() / (nsamples * 2048))
