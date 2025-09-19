import pathlib
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
# Should run univtg 
import re
import loader
import numpy as np
import pandas as pd
import torch
import librosa
import gpt_tf_queue
from gpt_tf_queue import binary_search_threshold, extract_txt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib
from openai import OpenAI
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, BartTokenizer
import random
from pathlib import Path
import pandas as pd
import torch.nn.functional as F
from run_on_video import clip

#!/usr/bin/env python
import argparse
from pathlib import Path
import wandb
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd

# ─── Argument Parsing ────────────────────────────────────────────────────────────
from pathlib import Path
import argparse
import sys
import logging
import json
from loader import load_json, save_json
import logging
import torch.backends.cudnn as cudnn
from main.config import TestOptions, setup_model
from utils.basic_utils import l2_normalize_np_array
import matplotlib.pyplot as plt
from run_on_video import vid2clip, txt2clip
from run_on_video import clip

import torch

client = OpenAI(
    # This is the default and can be omitted
    api_key=Your_API_KEY,
)



def load_model():
    opt = TestOptions().parse(args)
    print("opt = ", opt)
    opt.gpu_id = 0
    print("opt = ", opt.gpu_id)
    # pdb.set_trace()
    cudnn.benchmark = True
    cudnn.deterministic = False

    if opt.lr_warmup > 0:
        total_steps = opt.n_epoch
        warmup_steps = opt.lr_warmup if opt.lr_warmup > 1 else int(opt.lr_warmup * total_steps)
        opt.lr_warmup = [warmup_steps, total_steps]

    model, criterion, _, _ = setup_model(opt) # set up gpu for model
    return model

vtg_model = load_model()
vtg_model.eval()



def generate_wav_length(text,  narrator_count, out_speech_dir, video_name, DEVICE="cuda"):

    out_path = out_speech_dir / video_name[0] / video_name / f"narrator_{int(narrator_count)}_speech.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():

        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
        )
        response.stream_to_file(out_path)

    else:
        print(f"Audio file already exists: {out_path}")
    audio, sample_rate = librosa.load(str(out_path), sr=None)
    duration = librosa.get_duration(y=audio, sr=sample_rate)

    return duration

def inference(input_dir, OUT_DIR, screenplay_name, state, FPS, embedder, bart_tokenizer, checkpoint_dir, video_name_path, video_input_dir, interview_base, MIN_DURATION=3, TOLERANCE=1, DEVICE="cuda"):
    
    clip_model = clip.load("ViT-B/32", device=DEVICE, jit=False)[0]
    checkpoint = torch.load(checkpoint_dir)

    embedder_dict = checkpoint['embedder_state_dict']
    embedder.load_state_dict(embedder_dict)
    embedder = embedder.to(DEVICE)
    finetuned_bart_state_dict = checkpoint["bart_state_dict"]
    bart.load_state_dict(finetuned_bart_state_dict)
    bart.to(DEVICE)
    bart.eval()  
    print("bart loaded")


    logging.info("Model loaded")

    video_name_list = video_name_path.read_text().splitlines()

    for video_name in video_name_list:
        # Build the path to the screenplay summary CSV.
        print(f"Processing video: {video_name}")
        time_interval_path = OUT_DIR / video_name[0] / video_name / "time_interval_with_interview_contents.csv"
        time_interval_path.parent.mkdir(parents=True, exist_ok=True)
        if time_interval_path.exists():
            print(f"Time interval file already exists: {time_interval_path}")
            continue
        csv_path = input_dir / video_name[0] / video_name / screenplay_name
        if not csv_path.exists():
            print(f"CSV file not found for video: {video_name} at {csv_path}")
            continue

        # Read the CSV file.
        df = pd.read_csv(csv_path)
        time_interval = []
        last_last = None
        last = None
        part = None
        best_time_intervals = None
        narration_id = None
        narration_txt = None
        interview_contents = None
        # Process each row in the DataFrame.
        for idx, row in df.iterrows():
            tag = row["Tag"].strip()
            if tag == "Narration":
                part = "main"
                interview_contents = None

                text_duration = row["Duration"]
                narration_id = row["Narrator_Count"]
                narration_txt = row["Text"]

                best_duration, best_time_intervals = call_teasergenpt(row["Text"], video_name, video_input_dir, FPS, text_duration, last, last_last, MIN_DURATION, TOLERANCE, clip_model, vtg_model, DEVICE)

                print(f"best_duration = {best_duration}, best_time_intervals = {best_time_intervals}, type = {tag}, part = {part}")
            else:

                part = None
                narration_id = None
                narration_txt = None
                prev_narrator = find_prev_narrator(idx, df)
                next_narrator = find_next_narrator(idx, df)
                
                if state == "call_textual_retriver": 
                    interview_input_txt_path = interview_base / video_name[0] / video_name / f"interview.txt"
                    interview_input_json_path = interview_base / video_name[0] / video_name / f"interview.json"
                  
                    start_time, end_time, part, interview_contents = call_textual_retriver(last, last_last, prev_narrator, next_narrator, interview_input_txt_path, interview_input_json_path, bart, embedder, bart_tokenizer, DEVICE)
                

                if state == "call_gpt_retriver": 
                    interview_input_txt_path = interview_base / video_name[0] / video_name / f"interview.txt"
                    interview_input_json_path = interview_base / video_name[0] / video_name / f"interview.json"

                    print(f"gpt_out = {row['Text']}")
                    gpt_out = row['Text']
                    if pd.isna(gpt_out):

                        print(f"row {idx} has no Text (NaN)")
                        continue
                    start_time, end_time, part, interview_contents = call_gpt_retriver(last, last_last, row["Text"], interview_input_txt_path, interview_input_json_path, embedder, DEVICE)
                # might contain float
                best_time_intervals = [[round(start_time), round(end_time)]] # a list of time intervals
                best_duration = end_time - start_time
                print(f"best_duration = {best_duration}, best_time_intervals = {best_time_intervals}, type = {tag}, part = {part}")
            
            for interval in best_time_intervals:
                start_time = interval[0]
                end_time = interval[1]

                time_interval.append({"start_time": round(start_time), "end_time": round(end_time), "part": part, "tag": tag, "narration_id": narration_id, "narration_txt": narration_txt, "interview_contents": interview_contents})

            last_last = last
            last = best_time_intervals 
        time_interval_df = pd.DataFrame(time_interval)

        time_interval_df.to_csv(time_interval_path, index=False)


def find_prev_narrator(idx, df) -> str:
    for i in range(idx - 1, -1, -1):
        if df.iloc[i]["Tag"] == "Narration":
            return df.iloc[i]["Text"].strip()
    return ""

def find_next_narrator(idx, df) -> str:
    for i in range(idx + 1, len(df)):
        if df.iloc[i]["Tag"] == "Narration":
            return df.iloc[i]["Text"].strip()
    return ""


def call_teasergenpt(text, video_name, input_dir, FPS, text_duration, last, last_last, min_duration, tolerance, clip_model, vtg_model, device):
    print(f"text = {text}, text_duration = {text_duration}")
    text = str(text)
    best_duration, best_time_intervals = binary_search_threshold( text, video_name, input_dir, FPS, text_duration, last, last_last, min_duration,tolerance, clip_model,vtg_model, device,  min_threshold=0, max_threshold=2, precision=0.001)
    return best_duration, best_time_intervals

def encode_article_context(tokenizer, raw_text, max_length) -> dict:
    """
    Encodes the article context by splitting it on the mask token,
    tokenizing each part, and then manually inserting the mask token ID in between.
    Assumes raw_text contains exactly one mask token.
    """
    mask_token = tokenizer.mask_token
    mask_token_id = tokenizer.mask_token_id

    # Check for exactly one mask token.
    if raw_text.count(mask_token) != 1:
        raise AssertionError(f"Expected exactly 1 mask token in article_context_raw, but found {raw_text.count(mask_token)} in: {raw_text}")

    parts = raw_text.split(mask_token)
    prev_text = parts[0].strip()
    next_text = parts[1].strip()

    prev_tokens = tokenizer(prev_text, add_special_tokens = False).input_ids if prev_text else [] # input: <sos><w1><eos><mask><sos><w2><eos> 
    next_tokens = tokenizer(next_text, add_special_tokens = False).input_ids if next_text else []

    combined_tokens = [tokenizer.bos_token_id] + prev_tokens + [mask_token_id] + next_tokens + [tokenizer.eos_token_id]

    if len(combined_tokens) > max_length:

        combined_tokens = combined_tokens[:max_length-1]
        if mask_token_id not in combined_tokens:
            combined_tokens[-1] = mask_token_id  
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


def call_textual_retriver(last, last_last, prev_narrator, next_narrator, interview_input_txt_path, interview_input_json_path, bart, embedder, bart_tokenizer, device, MAX_LENGTH):
    interview_segments = loader.load_txt(interview_input_txt_path)
    mask_token = bart_tokenizer.mask_token  # e.g., "<mask>"
    query_context = f"{prev_narrator} {mask_token} {next_narrator}"
    inputs = encode_article_context(bart_tokenizer, query_context, max_length=MAX_LENGTH)
    input_ids = inputs["input_ids"].unsqueeze(0).to(device)  # Add batch dimension
    attention_mask = inputs["attention_mask"].unsqueeze(0).to(device)  # Add batch dimension
    bart.eval()
    embedder.eval()
    with torch.no_grad():
        bart_outputs = bart(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Pool the encoder hidden states (e.g., take the last hidden state of the decoder)
        query_embedding = bart_outputs.decoder_hidden_states[-1][:, -1, :]  # Shape: (1, hidden_dim)
    
        # Encode the candidate interview segments using the embedder.
        gt_embeddings = embedder.encode(interview_segments, convert_to_tensor=True).to(device)  # Shape: (num_candidates, hidden_dim)
        

    query_embedding = F.normalize(query_embedding, p=2, dim=-1)
    gt_embeddings = F.normalize(gt_embeddings, p=2, dim=-1)
    cosine_sim = F.cosine_similarity(query_embedding.unsqueeze(1), gt_embeddings.unsqueeze(0), dim=-1)
    cosine_sim = cosine_sim.squeeze(0)
    best_idx = torch.argmax(cosine_sim).item()


    sorted_indices = torch.argsort(cosine_sim, descending=True)

    if len(sorted_indices) < 3:
        second_best_idx = sorted_indices[0].item()
        third_best_idx = sorted_indices[0].item()
    else:
        second_best_idx = sorted_indices[1].item()
        third_best_idx = sorted_indices[2].item()
        
    interview_json_contents = loader.load_json(interview_input_json_path)
    
    start_time = interview_json_contents[best_idx]["Start Time"]
    end_time = interview_json_contents[best_idx]["End Time"]
    part = interview_json_contents[best_idx]["part"]
    interview_contents = interview_json_contents[best_idx]["Text"]
    if last is not None:
        if [round(start_time), round(end_time)] == last[0]:
            start_time = interview_json_contents[second_best_idx]["Start Time"]
            end_time = interview_json_contents[second_best_idx]["End Time"]
            part = interview_json_contents[second_best_idx]["part"]
            interview_contents = interview_json_contents[second_best_idx]["Text"]
    if last_last is not None:
        if [round(start_time), round(end_time)] == last_last[0]:
            start_time = interview_json_contents[third_best_idx]["Start Time"]
            end_time = interview_json_contents[third_best_idx]["End Time"]
            part = interview_json_contents[third_best_idx]["part"]
            interview_contents = interview_json_contents[third_best_idx]["Text"]
    return start_time, end_time, part, interview_contents
    # find start and end point 

def call_gpt_retriver(last, last_last, gpt_out, interview_input_txt_path, interview_input_json_path, embedder, device): # find the closest embedding
    interview_segments = loader.load_txt(interview_input_txt_path)
    with torch.no_grad():
        gt_embeddings = embedder.encode(interview_segments, convert_to_tensor=True).to(device)
        gpt_out_embedding = embedder.encode([gpt_out], convert_to_tensor=True).to(device)  # Shape: (1, hidden_dim)
    query_embedding = F.normalize(gpt_out_embedding, p=2, dim=-1)
    gt_embeddings = F.normalize(gt_embeddings, p=2, dim=-1)
    cosine_sim = F.cosine_similarity(query_embedding.unsqueeze(1), gt_embeddings.unsqueeze(0), dim=-1)

    cosine_sim = cosine_sim.squeeze(0)  # Now shape: (num_candidates,)

    best_idx = torch.argmax(cosine_sim).item()
    sorted_indices = torch.argsort(cosine_sim, descending=True)
    if len(sorted_indices) < 3:
        second_best_idx = sorted_indices[0].item()
        third_best_idx = sorted_indices[0].item()
    else:
        second_best_idx = sorted_indices[1].item()
        third_best_idx = sorted_indices[2].item()

    interview_json_contents = loader.load_json(interview_input_json_path)
    
    start_time = interview_json_contents[best_idx]["Start Time"]
    end_time = interview_json_contents[best_idx]["End Time"]
    part = interview_json_contents[best_idx]["part"]
    interview_contents = interview_json_contents[best_idx]["Text"]
    if last is not None:
        if [round(start_time), round(end_time)] == last[0]:
            start_time = interview_json_contents[second_best_idx]["Start Time"]
            end_time = interview_json_contents[second_best_idx]["End Time"]
            part = interview_json_contents[second_best_idx]["part"]
            interview_contents = interview_json_contents[second_best_idx]["Text"]
    if last_last is not None:
        if [round(start_time), round(end_time)] == last_last[0]:
            start_time = interview_json_contents[third_best_idx]["Start Time"]
            end_time = interview_json_contents[third_best_idx]["End Time"]
            part = interview_json_contents[third_best_idx]["part"]
            interview_contents = interview_json_contents[third_best_idx]["Text"]
    return start_time, end_time, part, interview_contents

def construct_demo():
    pass



if __name__ == "__main__":
    pass