"""
    Chapter 5 demo: pretrain the GPT model on "The Verdict"
    run from the repo root: python -m scripts.ch05_pretrain [--epochs 10]
"""

import argparse
from pathlib import Path

import torch
import tiktoken

from lumen.config import GPT_CONFIG_124M
from lumen.data import create_dataloader, load_verdict
from lumen.generate import generate, text_to_token_ids, token_ids_to_text
from lumen.model import GPTModel
from lumen.train import calc_loss_loader, plot_losses, train_model_simple

# shorter context than GPT-2's 1024 so training is feasible on a laptop
CONFIG = {**GPT_CONFIG_124M, "context_length": 256}
TRAIN_RATIO = 0.90
CHECKPOINT_DIR = Path("checkpoints")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    text = load_verdict()
    split_idx = int(TRAIN_RATIO * len(text))
    train_loader = create_dataloader(
        text[:split_idx], batch_size=2, max_length=CONFIG["context_length"],
        stride=CONFIG["context_length"], drop_last=True, shuffle=True)
    val_loader = create_dataloader(
        text[split_idx:], batch_size=2, max_length=CONFIG["context_length"],
        stride=CONFIG["context_length"], drop_last=False, shuffle=False)

    model = GPTModel(CONFIG).to(device)

    # an untrained model should be close to ln(50257) ~ 10.8 (uniform guessing)
    with torch.no_grad():
        print(f"initial train loss: {calc_loss_loader(train_loader, model, device):.3f}")
        print(f"initial val loss:   {calc_loss_loader(val_loader, model, device):.3f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer,
    )

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    epochs_seen = torch.linspace(0, args.epochs, len(train_losses))
    plot_losses(epochs_seen, tokens_seen, train_losses, val_losses,
                path=CHECKPOINT_DIR / "loss-plot.pdf")

    # sample with temperature + top-k for more varied text than greedy decoding
    model.eval()
    token_ids = generate(
        model, text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=15, context_size=CONFIG["context_length"],
        top_k=25, temperature=1.4,
    )
    print("sampled:", token_ids_to_text(token_ids, tokenizer))

    # save model + optimizer state so training can be resumed later
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": CONFIG,
    }, CHECKPOINT_DIR / "model_and_optimizer.pth")
    print(f"saved checkpoint to {CHECKPOINT_DIR / 'model_and_optimizer.pth'}")


if __name__ == "__main__":
    main()
