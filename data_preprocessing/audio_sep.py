import whisperx
import os
import torch
import pandas as pd
import pathlib
from pathlib import Path
import helper
from helper import load_txt, save_json, save_txt, load_json
from pydub import AudioSegment
device = "cuda"
batch_size = 8 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
model_config = "large-v2"
cuda_index = 0

use_auth_token = None  # your huggingface token here
def group_by_speaker(transcript):
    speakers = {}
    for entry in transcript:
        if 'speaker' in entry:
            speaker = entry['speaker']
            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append((entry['start'], entry['end'], entry['text']))
        else:
            print("No speaker found in entry:", entry)
            continue # cannot identify speaker, exclude from output
    return speakers

def diarize_audio(video_name_path, input_wav_dir, output_dir, part, bitrate='192k'):
    video_name_list = load_txt(video_name_path)
    error_file = []
    df = pd.DataFrame(columns=['video_name', 'narrator_id'])
    for vd in video_name_list:
        output_path_csv = output_dir / vd[0] / vd / "diarize_segments.csv"
        output_path_csv.parent.mkdir(parents=True, exist_ok=True)

        output_path_json = output_dir / vd[0] / vd / "diarize_result.json"
        output_path_json.parent.mkdir(parents=True, exist_ok=True)

        if output_path_csv.exists() and output_path_json.exists():
            print(f"Already processed video: {vd}")
            results = load_json(output_path_json)
            if results == {}:
                print(f"Empty results for video: {vd}")
                error_file.append(vd)
                continue
            diarize_segments = pd.read_csv(output_path_csv)
            narrator_id = get_narr_id(results, diarize_segments)
            df = pd.concat([df, pd.DataFrame({'video_name': [vd], 'narrator_id': [narrator_id]})])
            continue
        # try:
            # need process
        audio_file = input_wav_dir / vd[0] / vd / f"dialog_{part}.wav"
        model = whisperx.load_model(model_config, device, device_index=cuda_index, compute_type=compute_type) # device_index: the gpu that i will be using 
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size)

        # # # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=use_auth_token, device=device)


        diarize_segments = diarize_model(audio)
        diarize_model(audio)

        result = whisperx.assign_word_speakers(diarize_segments, result)


        diarize_segments.to_csv(output_path_csv)


        output = result['segments']
        if output == []:
            print(f"Empty output for video: {vd}")
            error_file.append(vd)
            continue

        grouped_output = group_by_speaker(output) # json is group by speaker
        save_json(output_path_json, grouped_output)


def get_narr_id(output, diarize_segments):
    
    narrator_id = None
    row_num = 0
    first_row_speaker = diarize_segments.loc[row_num, 'speaker']
    while first_row_speaker not in output.keys():  # all speaker id
        row_num += 1
        first_row_speaker = diarize_segments.loc[row_num, 'speaker']
    narrator_id = first_row_speaker
    return narrator_id



def find_exact_matching_text(video_name_path,input_dir, output_dir):


    video_name_list = load_txt(video_name_path)
    for vd in video_name_list:
        print("Processing video:", vd)

        data_frame_path = input_dir / vd[0] / vd / "diarize_segments.csv"
        transcripts_path = input_dir / vd[0] / vd / "diarize_result.json"
        if not data_frame_path.exists() and not transcripts_path.exists():
            continue
        data = load_json(transcripts_path)
        rows = []

        for speaker, segments in data.items():
            for segment in segments:
                start_time, end_time, text = segment
                text = text.strip().replace('"', '')
                rows.append([speaker, start_time, end_time, text])

        # Create DataFrame
        df = pd.DataFrame(rows, columns=["Speaker", "Start Time", "End Time", "Text"])

        # Sort the DataFrame by end Time
        df_sorted = df.sort_values(by=["End Time"]) # no overlap
        output_path = output_dir / vd[0] / vd / "script_timeline.csv"     
        output_path.parent.mkdir(parents=True, exist_ok=True)
        all_text = df_sorted["Text"].tolist()
        all_text = " ".join(all_text)
        # also save all the text
        text_path = output_dir / vd[0] / vd / "dialogue_main.txt"
        with open(text_path, 'w') as file:
            file.write(all_text)

        df_sorted.to_csv(output_path, index=False)

def narrator_selection(video_name_path, input_dir, output_dir):
    # load all narrtor txt and fild the longest as the model output
    df = pd.DataFrame(columns=['video_name', 'narrator_id'])
    video_name_list = load_txt(video_name_path)
    for vd in video_name_list:
        # load all possible txt file in this folder
        print(f"processing ... {vd}")
        subfolder_path = input_dir / vd[0] / vd
        txt_files = [file for file in subfolder_path.glob('*.txt')] # remove the default narrator
        # load all txt_files and find what is the longest
        longest_txt_id = 0
        long_txt_len = 0
        
        for txt_file in txt_files:
            with open(txt_file, 'r') as file:
                content = file.read() # do not need to remove lines
            if len(content) > long_txt_len:
                long_txt_len = len(content)
                # no root path and no extension
                txt_file = Path(txt_file)

                longest_txt_id = txt_file.stem
                print(f"longest_txt_id = {longest_txt_id}")
        df = pd.concat([df, pd.DataFrame({'video_name': vd, 'narrator_id': [str(longest_txt_id)]})])
    df.to_csv(output_dir / "narrator_id_by_length.csv", index=False)

def remove_file(video_name_path, input_dir):
    # first remove all txt files
    video_name_list = load_txt(video_name_path)
    for vd in video_name_list:
        try:
            print(f"processing ... {vd}")
            subfolder_path = input_dir / vd[0] / vd
            txt_files = [file for file in subfolder_path.glob('*.txt')]
            for txt_file in txt_files:
                txt_file.unlink()
        except:
            print(f"error processing ... {vd}")

def text_script(video_name_path, input_dir, output_dir):
    video_name_list = load_txt(video_name_path)
    for vd in video_name_list:
        print(f"processing ... {vd}")
        input_json = input_dir / vd[0] / vd / "diarize_result.json"
        input_json_content = load_json(input_json)
        for speaker, content_list in input_json_content.items():
            content_str = ""
            for content in content_list:
                content_str += str(content[2])
            output_path = output_dir / vd[0] / vd / f"{speaker}.txt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'a') as file:
                file.write(content_str)

def combine_consecutive_rows(input_dir, output_dir, file_name):
    # Read the CSV file into a DataFrame
    input_csv = input_dir / file_name[0] / file_name / f"script_timeline.csv"
    output_csv = output_dir / file_name[0] / file_name / f"script_timeline_combined.csv"
    df = pd.read_csv(input_csv)
    
    # Check if the DataFrame is empty
    if df.empty:
        print("The CSV file is empty!")
        return

    # Initialize variables with the first row's values
    current_speaker = df.iloc[0]['Speaker']
    current_start = df.iloc[0]['Start Time']
    current_end = df.iloc[0]['End Time']
    current_text = df.iloc[0]['Text']

    # List to store combined rows
    combined_rows = []

    for idx in range(1, len(df)):
        row = df.iloc[idx]

        if row['Speaker'] == current_speaker:
            current_end = row['End Time']  # Update end time to current row's end time
            current_text = current_text + " " + row['Text']  # Append text with a space
        else:

            combined_rows.append({
                'Speaker': current_speaker,
                'Start Time': current_start,
                'End Time': current_end,
                'Text': current_text
            })

            current_speaker = row['Speaker']
            current_start = row['Start Time']
            current_end = row['End Time']
            current_text = row['Text']


    combined_rows.append({
        'Speaker': current_speaker,
        'Start Time': current_start,
        'End Time': current_end,
        'Text': current_text
    })

    combined_df = pd.DataFrame(combined_rows)
    combined_df.to_csv(output_csv, index=False)
    print(f"Combined CSV saved to {output_csv}")

def combine_chunk(start_index, end_index, df):
    combined_text = " ".join(df.iloc[start_index:end_index]['Text'].tolist())
    combined_row = {
        'Speaker': df.iloc[start_index]['Speaker'],
        'Start Time': df.iloc[start_index]['Start Time'],
        'End Time': df.iloc[end_index-1]['End Time'],
        'Text': combined_text
    }
    return combined_row



def combine_consecutive_lecture(input_dir, output_dir, annotation_path, file_name):
    input_csv = input_dir / file_name[0] / file_name / "script_timeline.csv"
    output_csv = output_dir / file_name[0] / file_name / "script_timeline_combined.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    slide_changes = [float(line) for line in lines]
    df = pd.read_csv(input_csv)
    slide_changes = sorted(set(slide_changes))
    boundaries = slide_changes.copy()
    if not boundaries or boundaries[0] > 0.0:
        boundaries.insert(0, 0.0)
    boundaries.append(float('inf'))

    labels = list(range(1, len(boundaries)))

    df["Slide"] = pd.cut(df["Start Time"],
                         bins=boundaries, # [(),(),(),]
                         labels=labels,
                         right=False)

    # 6) drop any rows that fell outside all bins (if any)
    df = df.dropna(subset=["Slide"])

    # 7) group by Slide and merge
    grouped = (
        df.groupby("Slide", as_index=False)
          .agg({
              "Speaker":    "first",           # assume same speaker per slide
              "Start Time": "min",             # earliest start in this slide
              "End Time":   "max",             # latest end in this slide
              "Text":       lambda texts:      # concatenate all texts
                            " ".join(texts.str.strip())
          })
    )

    grouped = grouped[grouped['Speaker'].notna()]
    grouped.to_csv(output_csv, index=False)
    print(f"Grouped CSV saved to {output_csv}")


def combine_consecutive_rows_scale(input_dir, output_dir, file_name_list_path, state, annotation_dir):
    file_name_list = load_txt(file_name_list_path)
    if not file_name_list:
        print("No files to process.")
    for file_name in file_name_list:
        print(f"Processing {file_name}...")
        if state == "lecture":
            annotation_path = annotation_dir / file_name / "segments.txt"
            combine_consecutive_lecture(input_dir, output_dir, annotation_path, file_name)
        else:
            combine_consecutive_rows(input_dir, output_dir, file_name)
    print("All files processed successfully.")

if __name__ == "__main__":
    pass