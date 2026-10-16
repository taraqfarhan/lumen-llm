"""
    Chapter 4 demo: build the 124M GPT model and generate (gibberish) text with it
    run from the repo root: python -m scripts.ch04_gpt_model
"""

import torch
import tiktoken

from lumen.config import GPT_CONFIG_124M
from lumen.model import GPTModel
from lumen.generate import generate_text_simple, text_to_token_ids, token_ids_to_text


if __name__ == "__main__":
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")

    batch = torch.stack([
        text_to_token_ids("Every effort moves you", tokenizer).squeeze(0),
        text_to_token_ids("Every day holds a", tokenizer).squeeze(0),
    ])

    model = GPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print("input batch:\n", batch)
    print("output shape:", logits.shape)    # (2, 4, 50257)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"total parameters: {total_params:,}")
    # GPT-2 reuses the token embedding matrix as the output head (weight tying),
    # which is why it is called "124M" even though this model has ~163M params
    tied_params = total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"parameters with weight tying: {tied_params:,}")

    size_mb = total_params * 4 / (1024 * 1024)    # float32 = 4 bytes
    print(f"model size: {size_mb:.2f} MB")

    # untrained model -> random-looking output
    model.eval()    # disable dropout
    encoded = text_to_token_ids("Hello, I am", tokenizer)
    out = generate_text_simple(model, encoded, max_new_tokens=6,
                               context_size=GPT_CONFIG_124M["context_length"])
    print("generated:", token_ids_to_text(out, tokenizer))
