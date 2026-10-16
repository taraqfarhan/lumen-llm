"""
    Chapter 2 demo: token IDs -> token embeddings + positional embeddings
    run from the repo root: python -m scripts.ch02_embeddings
"""

import torch

from lumen.data import create_dataloader, load_verdict

VOCAB_SIZE = 50257
# aligns with OpenAI’s GPT-2 standard.
# vocabulary size of 50,257 is the exact signature of the GPT-2 tokenizer
# 50,000 merged BPE tokens + 256 raw byte tokens + 1 <|endoftext|> token

OUTPUT_DIM = 256
MAX_LENGTH = 4    # maximum length of the input size
CONTEXT_LENGTH = 4     # represents the supported input size of the LLM
BATCH_SIZE = 8


if __name__ == "__main__":
    raw_text = load_verdict()

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

    print("inputs:\n", inputs)
    print("targets:\n", targets)
    print("token embeddings:", token_embeddings.shape)
    print("pos embeddings:", pos_embeddings.shape)
    print("input embeddings:", input_embeddings.shape)
