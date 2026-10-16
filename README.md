# lumen-llm

A GPT-style large language model built from scratch in PyTorch, following
*Build a Large Language Model (From Scratch)* by Sebastian Raschka.

## Progress

| Chapter | Topic | Status | Code |
|---|---|---|---|
| 2 | Working with text data | ✅ | `lumen/data.py` |
| 3 | Coding attention mechanisms | ✅ | `lumen/attention.py` |
| 4 | Implementing a GPT model | ✅ | `lumen/model.py`, `lumen/generate.py` |
| 5 | Pretraining on unlabeled data | ✅ | `lumen/train.py`, `lumen/gpt2_weights.py` |
| 6 | Finetuning for classification | ⬜ | |
| 7 | Finetuning to follow instructions | ⬜ | |

## Layout

```
lumen/
  data.py          # GPT-2 BPE tokenizer, sliding-window dataset, dataloader
  attention.py     # self-attention -> causal attention -> multi-head attention
  config.py        # GPT_CONFIG_124M and GPT-2 model sizes
  model.py         # LayerNorm, GELU, FeedForward, TransformerBlock, GPTModel
  generate.py      # greedy, temperature and top-k decoding
  train.py         # loss, training loop, evaluation, loss plot
  gpt2_weights.py  # load OpenAI's pretrained GPT-2 weights
scripts/           # one runnable demo per chapter
assets/            # training text
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

## Running

Run the scripts as modules from the repo root:

```bash
python -m scripts.ch02_embeddings   # token + positional embeddings
python -m scripts.ch03_attention    # all attention variants
python -m scripts.ch04_gpt_model    # build the 124M model, count params, generate
python -m scripts.ch05_pretrain     # train on the-verdict.txt (--epochs N)
python -m scripts.ch05_load_gpt2    # load OpenAI GPT-2 weights (--model, --prompt)
```

Checkpoints, loss plots and downloaded weights go to `checkpoints/`, which is gitignored.
GPT-2 weights are downloaded from the Hugging Face hub in safetensors format,
instead of the TensorFlow checkpoint the book uses.
