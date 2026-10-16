"""
    model configurations
"""

GPT_CONFIG_124M = {
    "vocab_size": 50257,       # GPT-2 BPE vocabulary
    "context_length": 1024,    # max number of input tokens
    "emb_dim": 768,            # embedding dimension
    "n_heads": 12,             # attention heads
    "n_layers": 12,            # transformer blocks
    "drop_rate": 0.1,          # dropout rate
    "qkv_bias": False,         # bias in the query/key/value projections
}

# sizes of the released OpenAI GPT-2 models
GPT2_MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
