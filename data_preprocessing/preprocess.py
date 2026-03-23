"""
Divide into chunks for GPT4 to process, save as a summary
"""
import nltk
import json
from nltk.tokenize import sent_tokenize
import numpy
import os
import pandas as pd
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
# Ensure you have the Punkt sentence tokenizer models downloaded
nltk.download('punkt_tab')


YOUR_API_KEY = os.getenv("OPENAI_API_KEY")
def save_json(filename, data):
    """Save data as a JSON file."""
    with open(filename, "w", encoding="utf8") as f:
        json.dump(data, f)
def save_txt(filename, data):
    """Save a list to a TXT file."""
    with open(filename, "w", encoding="utf8") as f:
        for item in data:
            f.write(f"{item}\n")


def load_txt(filename):
    """Load a TXT file as a list."""
    with open(filename, encoding="utf8") as f:
        return [line.strip() for line in f]



def split_into_chunks(text, n=10):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Calculate the number of sentences per chunk
    k, m = divmod(len(sentences), n)
    
    # Generate the chunks
    chunks = []
    for i in range(n):
        # Calculate start and end indices for each chunk
        start_index = i * k + min(i, m)
        end_index = start_index + k + (1 if i < m else 0)
        chunk = ' '.join(sentences[start_index:end_index])
        chunks.append(chunk)
    
    return chunks


   
def get_chunk_summary(input_dir, output_dir, video_name):
    """
    input: original transcirpt
    # load_path, narr_path, output_csv_path, video_title
    """

    summary_text = [] 

    text = load_main_script(input_dir, video_name)
    chunks = split_into_chunks(text) # return a list of paragraph
      
    for index, content in enumerate(chunks):
      prompt = "Summarize the paragraph in one sentence:" + str(content)
      print(f"current prompt = {prompt}")
      system_prompt = "You are a narrator for this story."
      summary = call_gpt(prompt, system_prompt, "gpt-4o")
      summary_text.append(summary) 
    long_string = ' '.join(summary_text)
    
    output_path = output_dir / video_name[0] / video_name / "summary.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf8") as f:
        f.write(long_string)
    return long_string



"""
Call ChatGPT 4 to do summarization: should be one sentence summarization
"""

# lock = threading.lock()
@retry(wait= wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(100)) # solve wait limit
def call_gpt(prompt, system_prompt, model_name = "gpt-3.5-turbo"):
  client = OpenAI(
      # This is the default and can be omitted
      api_key=os.getenv("OPENAI_API_KEY"),
  )

  chat_completion = client.chat.completions.create(
      messages=[
          {
              "role": "system",
              "content": system_prompt,
          }, 
          {
              "role": "user",
              "content": prompt,
          }
      ],
      model= model_name,
      seed = 42,
      max_tokens = 500,
      temperature=0.0, # probability distribution
  )

  # return the contents
  return chat_completion.choices[0].message.content


def load_main_script(input_dir, video_name):
    """
    Load the main script for the video
    """
    main_csv_file = input_dir / video_name[0] / video_name / "script_timeline.csv"
    main_df = pd.read_csv(main_csv_file)
    long_text = " ".join(main_df['Text'])
    return long_text


def generate_summary_scale(input_dir, output_dir, video_name_list_path):
    """
    Generate summary for each video in the list
    """
    video_name_list = load_txt(video_name_list_path)
    if not video_name_list:
        print("No files to process.")
    for video_name in video_name_list:
        print(f"Processing {video_name}...")
        # Call the combine_consecutive_rows function for each file
        get_chunk_summary(input_dir, output_dir, video_name)
    print("All files processed successfully.")


if __name__ == "__main__":
    screenplay_input_dir_main = Path("/home/weihanx/videogpt/deepx_data6/screen_play_main_test")
    train_val_main = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    generate_summary_scale(screenplay_input_dir_main, screenplay_input_dir_main, train_val_main)
    