import subprocess
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import soundfile as sf
import librosa
import pathlib
import numpy as np
import pandas as pd
from pathlib import Path
import json
import os
import logging
import time
import torch
import argparse
import subprocess


import json
from loader import load_json, save_json
import logging
import torch.backends.cudnn as cudnn
from main.config import TestOptions, setup_model
from utils.basic_utils import l2_normalize_np_array
import matplotlib.pyplot as plt
from run_on_video import vid2clip, txt2clip
from run_on_video import clip
# from openai import OpenAI
import torch





def convert_to_hms(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def load_data(txt, vid_path, FPS, clip_model, device):

    txt = extract_txt(txt, clip_model)
    vid = extract_vid(vid_path, FPS, clip_model)

    vid = vid.astype(np.float32)
    txt = txt.astype(np.float32)

    vid = torch.from_numpy(l2_normalize_np_array(vid))
    txt = torch.from_numpy(l2_normalize_np_array(txt))
    clip_len = 1 / FPS

    ctx_l = vid.shape[0]

    timestamp =  ( (torch.arange(0, ctx_l) + clip_len / 2) / ctx_l).unsqueeze(1).repeat(1, 2)

    if True:
        tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
        tef_ed = tef_st + 1.0 / ctx_l
        tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
        vid = torch.cat([vid, tef], dim=1)  # (Lv, Dv+2)

    src_vid = vid.unsqueeze(0).to(device)
    src_txt = txt.unsqueeze(0).to(device)
    src_vid_mask = torch.ones(src_vid.shape[0], src_vid.shape[1]).to(device)
    src_txt_mask = torch.ones(src_txt.shape[0], src_txt.shape[1]).to(device)

    return src_vid, src_txt, src_vid_mask, src_txt_mask, timestamp, ctx_l

def forward(model, txt, vid_path, FPS, clip_model, device):

    clip_len = 1 / FPS
    src_vid, src_txt, src_vid_mask, src_txt_mask, timestamp, ctx_l = load_data(txt, vid_path, FPS, clip_model, device
    )
    src_vid = src_vid.to(device)
    src_txt = src_txt.to(device)
    src_vid_mask = src_vid_mask.to(device)
    src_txt_mask = src_txt_mask.to(device)

    model.eval()
    with torch.no_grad():
        output = model(src_vid=src_vid, src_txt=src_txt, src_vid_mask=src_vid_mask, src_txt_mask=src_txt_mask)

    pred_logits = output['pred_logits'][0].cpu().T # f --> 1,150
    pred_spans = output['pred_spans'][0].cpu() # b
    pred_saliency = output['saliency_scores'].cpu() # s

    saliency_score = pred_saliency.numpy()

    logits = pred_logits.numpy()

    highlight_score = saliency_score + logits
    return highlight_score
    
def extract_vid(vid_path, FPS, clip_model, save_dir):
    clip_len = 1/FPS
    video_name = vid_path.parts[-2]
    clip_name = vid_path.parts[-1].split('.')[0] # only need clip_3
    vid_path = str(vid_path)
    save_path = save_dir / video_name / clip_name
    save_path.mkdir(parents=True, exist_ok=True)
    check_exist = save_path / "vid.npz"
    if not check_exist.exists():
        vid_features = vid2clip(clip_model, vid_path, str(save_path), clip_len=clip_len)

    else:
        vid_features = np.load(check_exist)['features']
    return vid_features

def extract_txt(txt, clip_model, save_dir):
    save_dir = save_dir
    txt_features = txt2clip(clip_model, txt, save_dir) 
    return txt_features

def load_mp4_files(folder_path):
    folder = Path(folder_path)
    
    mp4_files = list(folder.glob('*.mp4'))
    return mp4_files

def get_saliency_curve(text, video_name, video_clip_folder, FPS, clip_model, vtg_model, device):
    """
    Given the text and video name, extract the saliency curve for the video
    """
    mp4_folder_path = video_clip_folder / video_name[0] / video_name
    mp4_files = load_mp4_files(mp4_folder_path)
    saliency_score_concat = []
    for mp4 in mp4_files: 
        saliency_score = forward(vtg_model, text, mp4, FPS, clip_model, device)

        saliency_score = saliency_score.squeeze() 
        saliency_score_concat.append(saliency_score)


    saliency_score_concat = np.concatenate(saliency_score_concat) 

    return saliency_score_concat



def find_consecutive_intervals(indices):
    if len(indices) == 0:
        return []

    intervals = []
    start = indices[0]
    
    for i in range(1, len(indices)):

        if indices[i] != indices[i - 1] + 1:
            intervals.append([start, indices[i - 1]])
            start = indices[i] 
    intervals.append([start, indices[-1]])
    
    return intervals

def filter_intervals_by_duration(intervals, min_duration):
    return [interval for interval in intervals if (interval[1] - interval[0] + 1) >= min_duration]


def clip_interval_extraction(text, clip_sal_score, threshold, FPS, last, last_last, min_dur, tolerance, clip_model, device):

    print(f"clip_sal_score = {clip_sal_score.shape}")

    clip_sal_score = clip_sal_score.flatten() 
    global_track = np.ones(clip_sal_score.shape[0])  # remove those should not be selected
    if last is not None:
        for ts in last: # a list of time interval
            start, end = ts
            start = start + tolerance
            end = end - tolerance # should be include
            global_track[int(start):int(end+1)] = 0
    if last_last is not None:
        for ts in last_last:
            start, end = ts
            start = start + tolerance
            end = end - tolerance
            global_track[int(start):int(end+1)] = 0

    available_indices = np.where(global_track == 1)[0]


    thresholded_indices = available_indices[clip_sal_score[available_indices] > threshold]

    thresholded_indices = thresholded_indices.tolist()
    consective_intervals = find_consecutive_intervals(thresholded_indices)
    filtered_interval = filter_intervals_by_duration(consective_intervals, min_dur)
    total_len = sum((interval[1] - interval[0] + 1) for interval in filtered_interval)
    return total_len, filtered_interval





def binary_search_threshold(text, video_name, input_dir, FPS, text_duration, last, last_last, min_duration,tolerance, clip_model, vtgmodel, device, min_threshold=-1, max_threshold=1.0, precision=0.001):

    low, high = min_threshold, max_threshold
    best_threshold = None
    best_duration = float('inf')
    best_time_intervals = None

    clip_sal_score = get_saliency_curve(text, video_name, input_dir, FPS, clip_model, vtgmodel, device)

    while high - low > precision:
        mid = (low + high) / 2
        print(f"mid = {mid}")

        total_duration, _= clip_interval_extraction(text, clip_sal_score, mid, FPS, last, last_last, min_duration, tolerance, clip_model, device)

        if total_duration > text_duration:
            if total_duration < best_duration:
                
                best_duration = total_duration
                best_threshold = mid
            low = mid  
        else:
            high = mid
    best_duration, best_time_intervals= clip_interval_extraction(text, clip_sal_score, best_threshold, FPS, last, last_last, min_duration, tolerance, clip_model, device)

    return best_duration, best_time_intervals
    


if __name__ == "__main__":
    pass