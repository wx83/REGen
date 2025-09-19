## Retrival based: Precision, Recall and F1

# load finetuned model
import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, BartTokenizer
import random
from dataset import InterviewDataset
from pathlib import Path
import wandb
import pandas as pd
import torch.nn.functional as F
import loader
import re
from statistics import mode
def fair_string_compare_ignore_case(s1, s2):
    s1_clean = re.sub(r'\s+', '', s1).lower()
    s2_clean = re.sub(r'\s+', '', s2).lower()
    return s1_clean == s2_clean

def encode_article_context(tokenizer, raw_text, max_length) -> dict:
    mask_token = tokenizer.mask_token
    mask_token_id = tokenizer.mask_token_id

    parts = raw_text.split(mask_token)
    prev_text = parts[0].strip()
    next_text = parts[1].strip()


    prev_tokens = tokenizer(prev_text, add_special_tokens = False).input_ids if prev_text else [] # input: <sos><w1><eos><mask><sos><w2><eos> 
    next_tokens = tokenizer(next_text, add_special_tokens = False).input_ids if next_text else []

    combined_tokens = [tokenizer.bos_token_id] + prev_tokens + [mask_token_id] + next_tokens + [tokenizer.eos_token_id] # ensure mask id is included

    if len(combined_tokens) > max_length:
        # if no mask token, then I should truncate the last token
        combined_tokens = combined_tokens[:max_length-1]
        if mask_token_id not in combined_tokens:
            combined_tokens[-1] = mask_token_id  
        # add the end of sentenece token
        combined_tokens.extend([tokenizer.eos_token_id])
        attention_mask = [1] * max_length
    else:
        attention_mask = [1] * len(combined_tokens)
        pad_length = max_length - len(combined_tokens)
        combined_tokens += [tokenizer.pad_token_id] * pad_length
        attention_mask += [0] * pad_length

    return {
        "input_ids": torch.tensor(combined_tokens),
        "attention_mask": torch.tensor(attention_mask)
    }


def retrieve_interview_segment(query_context, ground_truth_interview, interview_segments, bart, embedder, bart_tokenizer, device, MAX_LENGTH):

    inputs = encode_article_context(bart_tokenizer, query_context[0], max_length=MAX_LENGTH)
    input_ids = inputs["input_ids"].unsqueeze(0).to(device)  # Add batch dimension
    attention_mask = inputs["attention_mask"].unsqueeze(0).to(device)  # Add batch dimension

    bart.eval()
    embedder.eval()
    with torch.no_grad():
        bart_outputs = bart(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        query_embedding = bart_outputs.decoder_hidden_states[-1][:, -1, :] 
        gt_embeddings = embedder.encode(interview_segments, convert_to_tensor=True).to(device)  # Shape: (num_candidates, hidden_dim)
        
    query_embedding = F.normalize(query_embedding, p=2, dim=-1)
    gt_embeddings = F.normalize(gt_embeddings, p=2, dim=-1)
    cosine_sim = F.cosine_similarity(query_embedding.unsqueeze(1), gt_embeddings.unsqueeze(0), dim=-1)
    cosine_sim = cosine_sim.squeeze(0)
    best_idx = torch.argmax(cosine_sim).item()
    best_candidate = interview_segments[best_idx]
    best_score = cosine_sim[best_idx].item()
    
    sorted_indices = torch.argsort(cosine_sim, descending=True)

    # Compute Recall@1, @5, @10
    recall_scores = {}
    for k in (1, 5, 10):
        topk = sorted_indices[:k].tolist()
        # 1 if GT is in the top-k, else 0
        recall_scores[k] = int(
            any(
                fair_string_compare_ignore_case(interview_segments[idx], ground_truth_interview)
                for idx in topk
            )
        )

    # You can still pull out best/2nd/3rd if you like:
    best_idx             = sorted_indices[0].item()
    best_candidate       = interview_segments[best_idx]
    best_score           = cosine_sim[best_idx].item()
    second_best_idx      = sorted_indices[1].item()
    second_best_idx_score= cosine_sim[second_best_idx].item()
    third_best_idx       = sorted_indices[2].item()
    third_best_idx_score = cosine_sim[third_best_idx].item()

    # And your drop‐idx logic remains the same…
    drops    = torch.clamp(cosine_sim[:-1] - cosine_sim[1:], min=0.0)
    drop_idx = torch.argmax(drops).item()

    return {
        "best_candidate": best_candidate,
        "best_score": best_score,
        "second_best_score": second_best_idx_score,
        "third_best_score": third_best_idx_score,
        "drop_idx": drop_idx,
        "recall": recall_scores,
    }



def calcualte_average_interview_length(input_dir, video_name_list):

    total_num_interview = 0

    for vd in video_name_list:
        input_path = input_dir / vd[0] / vd / f"interview.txt"
        interview_list = loader.load_txt(input_path)
        num_interview = len(interview_list)
        total_num_interview += num_interview
    average_num_interview = total_num_interview / len(video_name_list)
    return average_num_interview



if __name__ == "__main__":
    pass