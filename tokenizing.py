"""
    Input Text -> Tokenized Text -> Token IDs 
    -> Token Embeddings + Positional Embeddings -> Input Embeddings
"""

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_len, stride):
        self.input_ids = []
        self.target_ids = []
        
        # tokenize the entire text
        token_ids = tokenizer.encode(text)
        
        # use a sliding window to chunk the text into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_len, stride):
            input_chunk = token_ids[i : i + max_len]
            target_chunk = token_ids[i + 1 : i + max_len + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    # total num of rows in the dataset
    def __len__(self):
        return len(self.input_ids)
    
    # return a single row from the dataseet
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
                            drop_last=drop_last, num_workers=0)
    
    return dataloader
    

with open('assets/the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read() 

    
VOCAB_SIZE = 50257 
# aligns with OpenAI’s GPT-2 standard.
# vocabulary size of 50,257 is the exact signature of the GPT-2 tokenizer
# 50,000 merged BPE tokens + 256 raw byte tokens + 1 <|endoftext|> token

OUTPUT_DIM = 256
MAX_LENGTH = 4    # maximum length of the input size
CONTEXT_LENGTH = 4     # represents the supported input size of the LLM
BATCH_SIZE = 8

dataloader = create_dataloader(raw_text, batch_size=BATCH_SIZE, 
                max_length=MAX_LENGTH, stride=MAX_LENGTH, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

# torch.nn.Embedding(num_embeddings, embedding_dim)
# torch.nn.Embedding is a lookup table that maps discrete token IDs into continuous vectors
# num embeddings: size of the dictionary/vocabulary
# embedding_dim: size of each embedding vector

# embedding_layer.weight holds the raw 2D weight matrix 
# that serves as the lookup table for the embedding layer

token_embedding_layer = torch.nn.Embedding(VOCAB_SIZE, OUTPUT_DIM)
token_embeddings = token_embedding_layer(inputs)

pos_embedding_layer = torch.nn.Embedding(CONTEXT_LENGTH, OUTPUT_DIM)
pos_embeddings = pos_embedding_layer(torch.arange(CONTEXT_LENGTH))

input_embeddings = token_embeddings + pos_embeddings
