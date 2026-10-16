"""
    Chapter 3 demo: every attention variant on the book's 6-token example
    run from the repo root: python -m scripts.ch03_attention
"""

import torch

from lumen.attention import (
    simple_self_attention, SelfAttention_v1, SelfAttention_v2,
    CausalAttention, MultiHeadAttentionWrapper, MultiHeadAttention,
)

# "Your journey starts with one step" as 3-dim embeddings
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],   # Your
     [0.55, 0.87, 0.66],   # journey
     [0.57, 0.85, 0.64],   # starts
     [0.22, 0.58, 0.33],   # with
     [0.77, 0.25, 0.10],   # one
     [0.05, 0.80, 0.55]]   # step
)


if __name__ == "__main__":
    torch.manual_seed(123)
    d_in, d_out = inputs.shape[1], 2

    print("simplified self-attention:\n", simple_self_attention(inputs))
    print("self-attention v1:\n", SelfAttention_v1(d_in, d_out)(inputs))
    print("self-attention v2:\n", SelfAttention_v2(d_in, d_out)(inputs))

    # batch of 2 identical inputs to simulate a dataloader batch
    batch = torch.stack((inputs, inputs), dim=0)
    context_length = batch.shape[1]

    ca = CausalAttention(d_in, d_out, context_length, dropout=0.0)
    print("causal attention:", ca(batch).shape)

    mha_wrapper = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout=0.0, num_heads=2)
    print("multi-head wrapper:", mha_wrapper(batch).shape)

    mha = MultiHeadAttention(d_in, d_out, context_length, dropout=0.0, num_heads=2)
    print("multi-head attention:", mha(batch).shape)
