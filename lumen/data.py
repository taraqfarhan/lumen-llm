"""
    Chapter 2: Working with Text Data
    Input Text -> Tokenized Text -> Token IDs
    -> Token Embeddings + Positional Embeddings -> Input Embeddings
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_len, stride):
        self.input_ids = []
        self.target_ids = []

        # tokenize the entire text
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        # use a sliding window to chunk the text into overlapping sequences of max_length
        # the target is the input shifted one token to the right (next-token prediction)
        for i in range(0, len(token_ids) - max_len, stride):
            input_chunk = token_ids[i : i + max_len]
            target_chunk = token_ids[i + 1 : i + max_len + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # total num of rows in the dataset
    def __len__(self):
        return len(self.input_ids)

    # return a single row from the dataset
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def create_dataloader(text, batch_size=4, max_length=256,
                      stride=128, shuffle=True, drop_last=True, num_workers=0):

    # initializing the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # creating the dataset
    dataset = GPTDataset(text, tokenizer, max_length, stride)

    # drop_last=True drops the last batch if it is shorter than the
    # specified batch_size to prevent loss spikes during training
    # num_workers is the num of CPU processes to use for preprocessing
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            drop_last=drop_last, num_workers=num_workers)

    return dataloader


def load_verdict():
    with open(ASSETS_DIR / "the-verdict.txt", "r", encoding="utf-8") as f:
        return f.read()
