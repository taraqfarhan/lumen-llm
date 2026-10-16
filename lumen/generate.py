"""
    text generation (chapter 4 greedy decoding + chapter 5 temperature / top-k sampling)
"""

import torch


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)    # add batch dimension


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())    # remove batch dimension


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # greedy decoding: always pick the most likely next token
    # idx: (batch, n_tokens) token IDs of the current context
    for _ in range(max_new_tokens):
        # crop the context if it exceeds what the model supports
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # only the last position predicts the next token: (batch, vocab_size)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)    # (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # top-k: keep only the k most likely tokens, mask out the rest
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val[:, None],
                torch.tensor(float("-inf"), device=logits.device),
                logits,
            )

        # temperature > 1 flattens the distribution (more random), < 1 sharpens it
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # stop early once the end-of-sequence token is produced
        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx
