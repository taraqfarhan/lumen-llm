"""
    Chapter 5: loading OpenAI's pretrained GPT-2 weights into our GPTModel

    the book downloads the original TensorFlow checkpoint; here we use the same weights
    from the Hugging Face hub in safetensors format (no TensorFlow dependency needed)
"""

from pathlib import Path

import requests
import torch
from safetensors.torch import load_file

HF_REPOS = {
    "gpt2-small (124M)": "openai-community/gpt2",
    "gpt2-medium (355M)": "openai-community/gpt2-medium",
    "gpt2-large (774M)": "openai-community/gpt2-large",
    "gpt2-xl (1558M)": "openai-community/gpt2-xl",
}


def download_gpt2(model_name, models_dir="checkpoints/gpt2"):
    repo = HF_REPOS[model_name]
    out_dir = Path(models_dir) / repo.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model.safetensors"

    if not path.exists():
        url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors"
        print(f"downloading {url}")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            tmp = path.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            tmp.rename(path)

    return load_file(path)


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.clone().detach())


def load_weights_into_gpt(gpt, params):
    # some checkpoints prefix every key with "transformer."
    params = {k.removeprefix("transformer."): v for k, v in params.items()}

    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe.weight"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte.weight"])

    for b, block in enumerate(gpt.trf_blocks):
        p = f"h.{b}."

        # GPT-2 uses Conv1D layers which store weights as (in, out),
        # nn.Linear stores (out, in) -> transpose every linear weight
        # the query, key and value weights are fused into a single c_attn matrix
        q_w, k_w, v_w = torch.chunk(params[p + "attn.c_attn.weight"], 3, dim=-1)
        block.att.W_query.weight = assign(block.att.W_query.weight, q_w.T)
        block.att.W_key.weight = assign(block.att.W_key.weight, k_w.T)
        block.att.W_value.weight = assign(block.att.W_value.weight, v_w.T)

        q_b, k_b, v_b = torch.chunk(params[p + "attn.c_attn.bias"], 3, dim=-1)
        block.att.W_query.bias = assign(block.att.W_query.bias, q_b)
        block.att.W_key.bias = assign(block.att.W_key.bias, k_b)
        block.att.W_value.bias = assign(block.att.W_value.bias, v_b)

        block.att.out_proj.weight = assign(block.att.out_proj.weight,
                                           params[p + "attn.c_proj.weight"].T)
        block.att.out_proj.bias = assign(block.att.out_proj.bias,
                                         params[p + "attn.c_proj.bias"])

        block.ff.layers[0].weight = assign(block.ff.layers[0].weight,
                                           params[p + "mlp.c_fc.weight"].T)
        block.ff.layers[0].bias = assign(block.ff.layers[0].bias,
                                         params[p + "mlp.c_fc.bias"])
        block.ff.layers[2].weight = assign(block.ff.layers[2].weight,
                                           params[p + "mlp.c_proj.weight"].T)
        block.ff.layers[2].bias = assign(block.ff.layers[2].bias,
                                         params[p + "mlp.c_proj.bias"])

        block.norm1.scale = assign(block.norm1.scale, params[p + "ln_1.weight"])
        block.norm1.shift = assign(block.norm1.shift, params[p + "ln_1.bias"])
        block.norm2.scale = assign(block.norm2.scale, params[p + "ln_2.weight"])
        block.norm2.shift = assign(block.norm2.shift, params[p + "ln_2.bias"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["ln_f.weight"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["ln_f.bias"])
    # weight tying: GPT-2 reuses the token embedding matrix as the output head
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte.weight"])
