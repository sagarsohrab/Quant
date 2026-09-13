"""Train the MiniGPT on a small text corpus to see an LLM learn from scratch.

This is a character-level model (each char is a token), the original GPT-1
setup. It uses a tiny corpus so training works on CPU in seconds — enough
to *feel* the loss fall and watch coherent text emerge.

Run:
    uv run --extra llm python -m qj.llm.train_demo
"""

from __future__ import annotations

import math
import random

import torch

from qj.llm.minigpt import MiniGPT

CORPUS = (
    "The small bird sat upon the branch of a tall tree\n"
    "and sang a sweet and simple song to the morning\n"
    "The sun rose warm across the rolling quiet hills\n"
    "and light spilled down into every sleeping valley\n"
    "The river wandered past old stones and silver roots\n"
    "curving slowly onward toward a far and patient sea\n"
)

DEVICE = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"


def _char_vocab(data: str):
    chars = sorted(set(data))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    return stoi, itos


def train(
    data: str,
    epochs: int = 200,
    block: int = 64,
    batch: int = 8,
    seed: int = 0,
) -> tuple[MiniGPT, dict]:
    torch.manual_seed(seed)
    stoi, itos = _char_vocab(data)
    ids = torch.tensor([stoi[c] for c in data], dtype=torch.long)
    x, y = ids[:-1], ids[1:]

    vocab = len(stoi)
    model = MiniGPT(
        vocab_size=vocab,
        d_model=96,
        n_heads=3,
        n_layers=2,
        d_ff=192,
        max_len=block,
        dropout=0.0,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    # shrink batch/block to fit a small corpus (need at least a couple blocks)
    B = batch if batch * block <= len(x) else max(1, len(x) // block)
    n_batches = max(1, len(x) // (B * block))
    rng = random.Random(seed)

    print(f"device={DEVICE} tokens={len(ids)} vocab={vocab} batch={B} block={block} "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    print("-" * 62)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        span = B * block
        hi = len(x) - span
        if hi < 1:  # corpus smaller than one batch; just use the full span
            s = 0
        else:
            s = rng.randrange(0, hi)
        idx = x[s : s + span].view(B, block).to(DEVICE)
        tgt = y[s : s + span].view(B, block).to(DEVICE)
        logits = model(idx)
        loss = loss_fn(logits.view(-1, vocab), tgt.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if epoch in {1} or epoch % 20 == 0:
            mean = sum(losses) / len(losses)
            print(f"epoch {epoch:3d}  loss={mean:.4f}  ppl={math.exp(mean):6.2f}  "
                  f"gen='{generate(model, itos, block, 40)}…'")
    return model, {"stoi": stoi, "itos": itos}


@torch.no_grad()
def generate(model: MiniGPT, itos: dict, block: int, length: int) -> str:
    model.eval()
    # seed from a random slice of context-sized arbitrary tokens (use '<' warmup char)
    keys = list(itos.keys())
    seed_ids = [keys[0]] * (block // 2)  # start from the first vocab char, repeated
    idx = torch.tensor([seed_ids], dtype=torch.long).to(DEVICE)
    out = model.generate(idx, max_new_tokens=length, temperature=0.8, top_k=15)
    tail = out[0, block // 2 :].tolist()
    return "".join(itos[int(i)] for i in tail)


if __name__ == "__main__":
    train(CORPUS, epochs=120)
