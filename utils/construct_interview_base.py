import json
from pathlib import Path
import pandas as pd
import loader  # Assuming loader has load_txt, save_json, save_txt functions
import math
import numpy as np
import torch
import random
torch.manual_seed(42)
def process_video(video_name: str, input_dir: Path, output_dir: Path, narrator_df: pd.DataFrame):

    print(f"Processing {video_name}...")

    # Find the narrator id for the video
    narrator_row = narrator_df.loc[narrator_df['video_name'] == video_name]
    if narrator_row.empty:
        print(f"Warning: No narrator found for video '{video_name}'. Skipping.")
        return
    pred_narrator_id = narrator_row.iloc[0]['narrator_id']
    print(f"pred_narrator_id = {pred_narrator_id}")
    script_csv_path = input_dir / video_name[0] / video_name / "script_timeline_combined.csv"
    try:
        narr_df = pd.read_csv(script_csv_path)
    except Exception as e:
        print(f"Error reading {script_csv_path}: {e}")
        return

    interview_segments = []  # List of dicts for JSON output
    interview_texts = []     # List of interview texts for TXT output

    # Loop through rows and collect interview segments.
    for idx, row in narr_df.iterrows():
        if row['Speaker'] != pred_narrator_id:
            segment = {
                "Speaker": row['Speaker'],
                "Text": row['Text'],
                "Start Time": row['Start Time'],
                "End Time": row['End Time']
            }
            interview_segments.append(segment)
            interview_texts.append(row['Text'])

    video_output_dir = output_dir / video_name[0] / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = video_output_dir / "interview.json"
    output_txt_path = video_output_dir / "interview.txt"

    loader.save_json(output_json_path, interview_segments)
    loader.save_txt(output_txt_path, interview_texts)
    print(f"Saved interview segments to {output_json_path}")



def construct_visual_embedding(current_text_clip_frame_embedding):
    num_visual = current_text_clip_frame_embedding.shape[0]
    print(f"num_visual = {num_visual}")
    if num_visual >= 3:
        indices = torch.randperm(num_visual)[:3]
        selected_visual = current_text_clip_frame_embedding[indices]
    else:

        indices = torch.randint(0, num_visual, (3,))
        selected_visual = current_text_clip_frame_embedding[indices]
    
    visual_flat = selected_visual.reshape(1, -1) 
    return visual_flat


def process_video_full(video_name: str, input_dir_main: Path, output_dir: Path, visual_embedding_dir_main: Path): # only main contents
    """
    Processes a single video to extract interview segments and save them.

    Args:
        video_name (str): The name of the video.
        input_dir_main (Path): Directory containing the main input files.
        input_dir_intro (Path): Directory containing the introductory input files.
        output_dir (Path): Directory where output files will be saved.
    """
    print(f"Processing {video_name}...")
    interview_segments = []  
    interview_texts = []    
    interview_visuals = []   

    narrator_csv_main = input_dir_main / "narrator_id_by_length.csv"
    narrator_df_main = pd.read_csv(narrator_csv_main)

    narrator_row = narrator_df_main.loc[narrator_df_main['video_name'] == video_name]
    if narrator_row.empty:
        print(f"Warning: No narrator found for video '{video_name}'. Skipping.")
        return
    pred_narrator_id = narrator_row.iloc[0]['narrator_id']
    print(f"Predicted narrator ID for main: {pred_narrator_id}")

    script_csv_path_main = input_dir_main / video_name[0] / video_name / "script_timeline_combined.csv"
    visual_embedding_path_main = visual_embedding_dir_main /  f"{video_name}_clip.npy"
    visual_embedding_path_main_np = np.load(visual_embedding_path_main)

    try:
        narr_df_main = pd.read_csv(script_csv_path_main)
    except Exception as e:
        print(f"Error reading {script_csv_path_main}: {e}")
        return


    for idx, row in narr_df_main.iterrows():
        if row['Speaker'] != pred_narrator_id:
            segment = {
                "Speaker": row['Speaker'],
                "Text": row['Text'],
                "Start Time": row['Start Time'],
                "End Time": row['End Time'],
                "part": "main"
            }
            interview_segments.append(segment)
            interview_texts.append(row['Text'])
            current_text_start = row["Start Time"]
            current_text_end = row["End Time"]

            start_index = math.floor(current_text_start)
            end_index = math.ceil(current_text_end)
            if start_index < 1:
                start_index = 1
            if end_index > visual_embedding_path_main_np.shape[0]:
                end_index = visual_embedding_path_main_np.shape[0]
            print(f"start_index = {start_index}, end_index = {end_index}")
            print(f"visual_embedding_path_main_np.shape = {visual_embedding_path_main_np.shape}")
            current_visual_embedding = visual_embedding_path_main_np[start_index-1:end_index] # should include start_index postion
            print(f"current_visual_embedding shape = {current_visual_embedding.shape}")
            compressed_visual_embedding = construct_visual_embedding(current_visual_embedding)
            assert compressed_visual_embedding.shape[0] == 1
            interview_visuals.append(compressed_visual_embedding) # average pooling

    print(f"Total interview segments (intro + main): {len(interview_segments)}")

    # Build output paths and ensure the output directory exists.
    video_output_dir = output_dir / video_name[0] / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = video_output_dir / "interview.json"
    output_txt_path = video_output_dir / "interview.txt"
    output_embedding_path = video_output_dir / "interview_visuals_random.npy"
    # need to stack 
    interview_visuals_stacked = np.vstack(interview_visuals)


    
    assert len(interview_segments) == interview_visuals_stacked.shape[0]
    
    np.save(output_embedding_path, interview_visuals_stacked)

    loader.save_json(output_json_path, interview_segments)
    loader.save_txt(output_txt_path, interview_texts)  

    print(f"Saved interview segments to {output_json_path}")

def process_video_full_lecture(video_name: str, input_dir_main: Path, output_dir: Path, visual_embedding_dir_main: Path): # only main contents

    print(f"Processing {video_name}...")
    interview_segments = []  # List of dicts for JSON output
    interview_texts = []     # List of interview texts for TXT output
    interview_visuals = []   # List of visual embeddings

    narrator_csv_main = input_dir_main / "narrator_id_by_length.csv"
    narrator_df_main = pd.read_csv(narrator_csv_main)

    narrator_row = narrator_df_main.loc[narrator_df_main['video_name'] == video_name]
    if narrator_row.empty:
        print(f"Warning: No narrator found for video '{video_name}'. Skipping.")
        return
    pred_narrator_id = narrator_row.iloc[0]['narrator_id']
    print(f"Predicted narrator ID for main: {pred_narrator_id}")

    script_csv_path_main = input_dir_main / video_name[0] / video_name / "script_timeline_combined.csv"
    visual_embedding_path_main = visual_embedding_dir_main /  f"{video_name}_clip.npy"
    visual_embedding_path_main_np = np.load(visual_embedding_path_main)

    try:
        narr_df_main = pd.read_csv(script_csv_path_main)
    except Exception as e:
        print(f"Error reading {script_csv_path_main}: {e}")
        return


    for idx, row in narr_df_main.iterrows():
        if row['Speaker'] == pred_narrator_id:
            segment = {
                "Speaker": row['Speaker'],
                "Text": row['Text'],
                "Start Time": row['Start Time'],
                "End Time": row['End Time'],
                "part": "main"
            }
            interview_segments.append(segment)
            interview_texts.append(row['Text'])
            current_text_start = row["Start Time"]
            current_text_end = row["End Time"]

            start_index = math.floor(current_text_start)
            end_index = math.ceil(current_text_end)
            if start_index < 1:
                start_index = 1
            if end_index > visual_embedding_path_main_np.shape[0]:
                end_index = visual_embedding_path_main_np.shape[0]
            print(f"start_index = {start_index}, end_index = {end_index}")
            print(f"visual_embedding_path_main_np.shape = {visual_embedding_path_main_np.shape}")
            current_visual_embedding = visual_embedding_path_main_np[start_index-1:end_index] 
            compressed_visual_embedding = construct_visual_embedding(current_visual_embedding)
            assert compressed_visual_embedding.shape[0] == 1
            interview_visuals.append(compressed_visual_embedding) # average pooling

    print(f"Total interview segments (intro + main): {len(interview_segments)}")


    video_output_dir = output_dir / video_name[0] / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = video_output_dir / "interview.json"
    output_txt_path = video_output_dir / "interview.txt"
    output_embedding_path = video_output_dir / "interview_visuals_random.npy"

    interview_visuals_stacked = np.vstack(interview_visuals)
    
    assert len(interview_segments) == interview_visuals_stacked.shape[0]
    
    np.save(output_embedding_path, interview_visuals_stacked)

    loader.save_json(output_json_path, interview_segments)
    loader.save_txt(output_txt_path, interview_texts)  

    print(f"Saved interview segments to {output_json_path}")


if __name__ == "__main__":
    pass