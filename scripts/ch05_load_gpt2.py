"""
    Chapter 5 demo: load OpenAI's pretrained GPT-2 weights into our model and generate text
    run from the repo root: python -m scripts.ch05_load_gpt2 [--model "gpt2-small (124M)"]
"""

import argparse

import torch
import tiktoken

from lumen.config import GPT_CONFIG_124M, GPT2_MODEL_CONFIGS
from lumen.generate import generate, text_to_token_ids, token_ids_to_text
from lumen.gpt2_weights import download_gpt2, load_weights_into_gpt
from lumen.model import GPTModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small (124M)", choices=GPT2_MODEL_CONFIGS)
    parser.add_argument("--prompt", default="Every effort moves you")
    args = parser.parse_args()

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    # OpenAI's GPT-2 was trained with a 1024 context and bias in the qkv projections
    config = {**GPT_CONFIG_124M, **GPT2_MODEL_CONFIGS[args.model],
              "context_length": 1024, "qkv_bias": True}

    model = GPTModel(config)
    load_weights_into_gpt(model, download_gpt2(args.model))
    model.to(device).eval()

    token_ids = generate(
        model, text_to_token_ids(args.prompt, tokenizer).to(device),
        max_new_tokens=25, context_size=config["context_length"],
        top_k=50, temperature=1.5,
    )
    print(token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
