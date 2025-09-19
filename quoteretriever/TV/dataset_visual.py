import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler, random_split
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import pandas as pd
import math
from loader import load_txt, save_txt
import random
import pathlib
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from transformers import BartForConditionalGeneration, BartTokenizer
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from transformers import BartTokenizer
from torch.utils.data import Dataset

def load_txt(file_path: str) -> list:
    # Replace with your actual file-reading logic.
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]

class InterviewDataset(Dataset):
    def __init__(self, input_dir, video_name_list, visual_folder, narrator_id_csv_path, max_length, tokenizer, special_token_position, pooling="all"):

        self.tokenizer = tokenizer
        self.visual_folder = visual_folder
        self.max_length = max_length
        self.video_name_lists = video_name_list
        self.narrator_df = pd.read_csv(narrator_id_csv_path)
        self.special_token_position = special_token_position
        self.pooling = pooling # mean or max
        self.data = []
        mask_token = self.tokenizer.mask_token  # e.g., "<mask>"

        for file_name in self.video_name_lists:
            matching = self.narrator_df[self.narrator_df["video_name"] == file_name]
            if matching.empty:
                print(f"Warning: No narrator found for video '{file_name}'. Skipping.")
                continue

            narrator_id = matching["narrator_id"].values[0]
            csv_file = Path(input_dir) / file_name[0] / file_name / "script_timeline_combined.csv"
            df = pd.read_csv(csv_file)
            clip_embedding = self.visual_folder / f"{file_name}_clip.npy"
            clip_embedding_np = np.load(clip_embedding)
            for idx, row in df.iterrows():
                if row["Speaker"] == narrator_id:
                    continue

                current_text = row["Text"].strip()
                current_text_start = row["Start Time"]
                current_text_end = row["End Time"]

                start_index = math.floor(current_text_start)
                end_index = math.ceil(current_text_end) 
                if start_index <= 1:
                    start_index = 1
                if end_index > clip_embedding_np.shape[0]:
                    end_index = clip_embedding_np.shape[0]
                current_text_clip_frame_embedding = clip_embedding_np[start_index-1:end_index]

                prev_text = self._find_prev_narrator(idx, df, narrator_id)
                next_text = self._find_next_narrator(idx, df, narrator_id)
                if prev_text and next_text:
                    article_context_raw = f"{prev_text} {mask_token} {next_text}"
                    full_sentence = f"{prev_text} {current_text} {next_text}"
                elif prev_text:
                    article_context_raw = f"{prev_text} {mask_token}"
                    full_sentence = f"{prev_text} {current_text}"
                elif next_text:
                    article_context_raw = f"{mask_token} {next_text}"
                    full_sentence = f"{current_text} {next_text}"
                else:
                    article_context_raw = mask_token
                    full_sentence = current_text
                if current_text_clip_frame_embedding.shape[0] == 0:
                    print(f"Warning: No visual embedding found for video '{file_name}' at start index {start_index}, end index {end_index} Skipping.")
                visual_clip_embedding = None
                if self.pooling == "mean":
                    visual_clip_embedding = np.mean(current_text_clip_frame_embedding, axis=0) # along time axis dimension
                if self.pooling == "all":
                    visual_clip_embedding = self._construct_visual_embedding(current_text_clip_frame_embedding) # 1 dimension, three visual embedding flattened
                assert visual_clip_embedding is not None, f"visual_clip_embedding is None for video '{file_name}' at start index {start_index}, end index {end_index}"
                sample = {
                    "article_context_raw": article_context_raw,
                    "full_sentence": full_sentence,
                    "interview_segements": current_text,
                    "visual_clip_embedding": visual_clip_embedding,
                    "video_name": file_name,
                }
                self.data.append(sample)
    
    def _construct_visual_embedding(self, current_text_clip_frame_embedding):
        num_visual = current_text_clip_frame_embedding.shape[0]
        if num_visual >= 3:
            indices = torch.randperm(num_visual)[:3]
            selected_visual = current_text_clip_frame_embedding[indices]
        else:

            indices = torch.randint(0, num_visual, (3,))
            selected_visual = current_text_clip_frame_embedding[indices]
        
        visual_flat = selected_visual.reshape(1, -1)
        # print(f"visual_flat shape = {visual_flat.shape}") # 1, 768*3
        return visual_flat
    
    def _find_prev_narrator(self, idx: int, df: pd.DataFrame, narrator_id: str) -> str:
        for i in range(idx - 1, -1, -1):
            if df.iloc[i]["Speaker"] == narrator_id:
                return df.iloc[i]["Text"].strip()
        return ""

    def _find_next_narrator(self, idx: int, df: pd.DataFrame, narrator_id: str) -> str:
        for i in range(idx + 1, len(df)):
            if df.iloc[i]["Speaker"] == narrator_id:
                return df.iloc[i]["Text"].strip()
        return ""

    def _encode_article_context(self, raw_text: str) -> dict:

        mask_token = self.tokenizer.mask_token
        mask_token_id = self.tokenizer.mask_token_id

        parts = raw_text.split(mask_token)
        prev_text = parts[0].strip()
        next_text = parts[1].strip()

        prev_tokens = self.tokenizer(prev_text, add_special_tokens = False).input_ids if prev_text else [] # input: <sos><w1><eos><mask><sos><w2><eos> 
        next_tokens = self.tokenizer(next_text, add_special_tokens = False).input_ids if next_text else []

        combined_tokens = [self.tokenizer.bos_token_id] + prev_tokens + [mask_token_id] + next_tokens + [self.tokenizer.eos_token_id]


        if len(combined_tokens) > self.max_length:

            combined_tokens = combined_tokens[:self.max_length-1]
            if mask_token_id not in combined_tokens:
                combined_tokens[-1] = mask_token_id 
            combined_tokens.extend([self.tokenizer.eos_token_id])
            attention_mask = [1] * self.max_length
        else:
            attention_mask = [1] * len(combined_tokens)
            pad_length = self.max_length - len(combined_tokens)
            combined_tokens += [self.tokenizer.pad_token_id] * pad_length
            attention_mask += [0] * pad_length

        return {
            "input_ids": torch.tensor(combined_tokens),
            "attention_mask": torch.tensor(attention_mask)
        }

    def _encode_interview_end(self, text: str) -> dict:

        """Tokenizes the full sentence and pads/truncates to max_length."""
        tokens = self.tokenizer(text, add_special_tokens = False).input_ids

        tokens_update = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id] + [self.tokenizer.additional_special_tokens_ids[0]]
        
        if len(tokens_update) > self.max_length:
            tokens = tokens[:self.max_length-3] 
            tokens_update = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id] + [self.tokenizer.additional_special_tokens_ids[0]]
            attention_mask = [1] * self.max_length
        else:
            attention_mask = [1] * len(tokens_update)
            pad_length = self.max_length - len(tokens_update)
            tokens_update += [self.tokenizer.pad_token_id] * pad_length
            attention_mask += [0] * pad_length

        return {
            "input_ids": torch.tensor(tokens_update),
            "attention_mask": torch.tensor(attention_mask)
        }
    def _encode_interview_front(self, text: str) -> dict:

        """Tokenizes the full sentence and pads/truncates to max_length."""
        tokens = self.tokenizer(text, add_special_tokens = False).input_ids

        tokens_update = [self.tokenizer.additional_special_tokens_ids[0]] + [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id] 
        
        if len(tokens_update) > self.max_length:
            tokens = tokens[:self.max_length-3]  # -2 for <sos> and <eos> <sum>
            tokens_update = [self.tokenizer.additional_special_tokens_ids[0]] + [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
            attention_mask = [1] * self.max_length
        else:
            attention_mask = [1] * len(tokens_update)
            pad_length = self.max_length - len(tokens_update)
            tokens_update += [self.tokenizer.pad_token_id] * pad_length
            attention_mask += [0] * pad_length

        return {
            "input_ids": torch.tensor(tokens_update),
            "attention_mask": torch.tensor(attention_mask)
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]
        context_encoded = self._encode_article_context(sample["article_context_raw"]) 
        if self.special_token_position == "front":
            interview_encoded = self._encode_interview_front(sample["interview_segements"])
        else:
            interview_encoded = self._encode_interview_end(sample["interview_segements"])
        return {
            "article_context": context_encoded, # w1, w2, w3, [mask], w4, w5, w6
            "interview_segements": interview_encoded, # output interview
            "video_name": sample["video_name"],
            "article_context_raw": sample["article_context_raw"],
            "interview_raw": sample["interview_segements"],
            "visual_clip_embedding": sample["visual_clip_embedding"]
        }

class GroupedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=32):
        self.dataset    = dataset
        self.batch_size = batch_size

        self.grouped_indices = {}
        for idx, sample in enumerate(dataset):
            mid = sample["video_name"]
            self.grouped_indices.setdefault(mid, []).append(idx)

        self.other_pool = {}
        all_items = set(range(len(dataset)))
        for mid, idxs in self.grouped_indices.items():
            self.other_pool[mid] = list(all_items - set(idxs))

    def __iter__(self):
        movie_ids = list(self.grouped_indices.keys())
        random.shuffle(movie_ids)

        for mid in movie_ids:
            own_idxs = self.grouped_indices[mid]
            random.shuffle(own_idxs)

            num_hard = int(len(own_idxs) * 0.3)
            if num_hard == 0:
                continue  

            pool = self.other_pool[mid]

            for start in range(0, len(own_idxs), num_hard):
                hard_batch = own_idxs[start : start + num_hard]
                needed    = self.batch_size - len(hard_batch)

                if needed > 0 and pool:
                    if len(pool) >= needed:
                        other_batch = random.sample(pool, needed)
                    else:
                        other_batch = random.choices(pool, k=needed)
                    batch = hard_batch + other_batch
                else:
                    batch = hard_batch[: self.batch_size]

                yield batch

    def __len__(self):
        total = 0
        for idxs in self.grouped_indices.values():
            nh = len(idxs) // 2
            if nh > 0:
                total += (len(idxs) + nh - 1) // nh
        return total

if __name__ == "__main__":
    pass