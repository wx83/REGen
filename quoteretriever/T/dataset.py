import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler, random_split
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import pandas as pd
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
from pathlib import Path
from transformers import BartTokenizer
from torch.utils.data import Dataset

def load_txt(file_path: str) -> list:

    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]

class InterviewDataset(Dataset):
    def __init__(self, input_dir, video_name_list, narrator_id_csv_path, max_length, tokenizer, special_token_position):
        self.tokenizer = tokenizer

        self.max_length = max_length
        self.video_name_lists = video_name_list
        self.narrator_df = pd.read_csv(narrator_id_csv_path)
        self.special_token_position = special_token_position
        self.data = []
        mask_token = self.tokenizer.mask_token 
        for file_name in self.video_name_lists:
            matching = self.narrator_df[self.narrator_df["video_name"] == file_name]


            narrator_id = matching["narrator_id"].values[0]
            csv_file = Path(input_dir) / file_name[0] / file_name / "script_timeline_combined.csv"
            df = pd.read_csv(csv_file)

            for idx, row in df.iterrows():
                if row["Speaker"] == narrator_id:
                    continue

                current_text = row["Text"].strip()
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

                sample = {
                    "article_context_raw": article_context_raw,
                    "full_sentence": full_sentence,
                    "interview_segements": current_text,
                    "video_name": file_name,
                }
                self.data.append(sample)

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
        # <sos> w1 w2 <eos> <sum>
        """Tokenizes the full sentence and pads/truncates to max_length."""
        tokens = self.tokenizer(text, add_special_tokens = False).input_ids
        tokens_update = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id] + [self.tokenizer.additional_special_tokens_ids[0]]
        
        if len(tokens_update) > self.max_length:
            tokens = tokens[:self.max_length-3]  # -2 for <sos> and <eos> <sum>
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
        # <sos> w1 w2 <eos> <sum>
        """Tokenizes the full sentence and pads/truncates to max_length."""
        tokens = self.tokenizer(text, add_special_tokens = False).input_ids
        # add special token <sum>
        # add <sum> before the end of sentence token
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
            "interview_raw": sample["interview_segements"]
        }

class GroupedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=32):
        self.dataset = dataset
        self.batch_size = batch_size

        self.grouped_indices = {}
        for idx in range(len(dataset)): 
            movie_id = dataset[idx]["video_name"]
            if movie_id not in self.grouped_indices:
                self.grouped_indices[movie_id] = []
            self.grouped_indices[movie_id].append(idx)

        for indices in self.grouped_indices.values():
            random.shuffle(indices)

        # Prepare batches from each group.
        self.batches = []
        all_movie_ids = list(self.grouped_indices.keys())
        for movie_id, indices in self.grouped_indices.items():
            for i in range(0, len(indices), batch_size):
            
                batch = indices[i:i + batch_size]

                if len(batch) < batch_size:
                    other_movie_ids = [mid for mid in all_movie_ids if mid != movie_id]
                    if other_movie_ids:
                        select_movie_id = random.sample(other_movie_ids, 100) # in case of no other movie
                        pool = []
                        for mid in select_movie_id:
                            pool.extend(self.grouped_indices[mid])
                        needed = batch_size - len(batch)
                        extra = random.sample(pool, needed)
                        batch.extend(extra)
                self.batches.append(batch)

        # Shuffle the overall list of batches.
        random.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


# Create the grouped batch sampler using movie_id.
if __name__ == "__main__":
    pass