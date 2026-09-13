"""A minimal transformer, built from scratch.

This is an *educational* implementation (PyTorch, no HuggingFace). It strips
LLMs to their essence so you can see exactly how a modern language model
works end-to-end: token embedding -> positional encoding -> multi-head
self-attention -> feed-forward blocks -> output head -> token sampling.

The design mirrors the original "Attention Is All You Need" transformer with
GPT-style causal (masked) self-attention and token-level next-token
prediction — the same core architecture powering ChatGPT, Llama, etc.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Causal scaled dot-product attention across multiple heads."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: (batch, seq, d_model)
        B, T, C = x.shape
        # (B, T, heads, d_k) -> (B, heads, T, d_k)
        Q = self.w_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (B,h,T,T)
        # Causal mask: each token attends only to itself and prior tokens
        causal = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        scores = scores.masked_fill(causal == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ V                        # (B,h,T,d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.w_o(out)


class FeedForward(nn.Module):
    """"MLP block: projects up to a wider space, applies non-linearity, projects back."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One decoder block: self-attention + feed-forward, with residuals & layernorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        # Pre-norm residual ("GPT-style") layout
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class PositionalEncoding(nn.Module):
    """Learned (GPT) or sinusoidal (Vaswani) position embeddings.

    Transformers have no inherent notion of order, so positions must be
    injected explicitly. GPT uses learned embeddings; we default to that.
    """

    def __init__(self, d_model: int, max_len: int = 1024, learned: bool = True) -> None:
        super().__init__()
        if learned:
            self.embed = nn.Embedding(max_len, d_model)
        else:
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            denom = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * denom)
            pe[:, 1::2] = torch.cos(pos * denom)
            self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        B, T, C = x.shape
        if self.embed is not None:
            positions = torch.arange(0, T, device=x.device)
            return x + self.embed(positions)[None, :, :]
        return x + self.pe[:, :T, :]


class MiniGPT(nn.Module):
    """A compact decoder-only transformer (GPT-style) for next-token prediction.

    Config params:
        vocab_size  : number of tokens/classes
        d_model     : embedding + hidden dim
        n_heads     : attention heads
        n_layers    : number of transformer blocks
        d_ff        : feed-forward hidden dim
        max_len     : max context length
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = PositionalEncoding(d_model, max_len, learned=True)
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):  # idx: (batch, seq)
        B, T = idx.shape
        tok = self.token_embed(idx)         # (B, T, d_model)
        x = self.pos_embed(tok)             # add positions
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)               # (B, T, vocab)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int = 50, temperature: float = 1.0, top_k: int | None = None):
        """Autoregressive sampling: feed back the generated token, repeat."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]  # truncate to context
            logits = self(idx_cond)[:, -1, :] / temperature            # next-token logits
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
