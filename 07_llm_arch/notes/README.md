# Layer 7: LLM & Full-Stack Architecture — the deeper tech

You want "everything from tech to LLM architecture". This doc maps the
*whole* stack and points you at the exact code in this repo that
illustrates each idea, so theory and implementation always connect.

## 1. The full stack, bottom-up

```
┌──────────────────────────────────────────────────────────────┐
│  APPLICATIONS        forecasting, backtesting, live trading   │
├──────────────────────────────────────────────────────────────┤
│  MODEL TREND          transformers (MiniGPT in qj/llm/)       │
├──────────────────────────────────────────────────────────────┤
│  ML PIPELINE          features → train → eval → deploy        │
│                       (qj/features, qj/models, qj/mlops)      │
├──────────────────────────────────────────────────────────────┤
│  DATA / INFRA         Binance REST+WS, Parquet, streams       │
│                       (qj/data, qj/live)                      │
├──────────────────────────────────────────────────────────────┤
│  COMPUTE / OS         CPU/GPU (your MPS GPU), memory, files   │
└──────────────────────────────────────────────────────────────┘
```

Every machine-learning system, quant or LLM, sits on this same skeleton.

## 2. How an LLM actually works (MiniGPT = the blueprint)

A GPT-style model has three conceptual parts. `qj/llm/minigpt.py`
implements all of them from scratch:

### a) Embeddings (token → vector)
Words/characters are ids; embeddings map each id to a learned vector so
*similar* tokens land near each other. `token_embed = nn.Embedding(...)`.

### b) Positional encoding (order matters)
Transformers see a *bag* of tokens — no innate sense of order. We inject
position either as learned embeddings (GPT) or sinusoids (original paper).
`PositionalEncoding`.

### c) Self-attention (the real magic)
Each token decides how much to *attend* to every other token using
query/key/value triples:
```
attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V
```
Multi-head attention runs this in parallel sub-spaces so the model can
track several relationships at once. Query "how much do I care", key "how
relevant am I", value "what I carry". Causal mask keeps it predicting only
*forward* — a language model never sees the future. `MultiHeadAttention`.

### d) The residual block + layers
Layers stack attention + feed-forward (`MLP`) with residual connections
(add the input back, so gradients flow) and layer-normalization for stable
training. More layers + wider dims + more data = bigger, smarter models.
`TransformerBlock`.

### e) Training: next-token prediction
A language model learns by predicting the next token from the previous
ones (self-supervised — no labels needed). The output head projects hidden
states back to vocabulary-size logits; softmax → probabilities; cross
entropy loss. `model.head`, `CrossEntropyLoss`.

Run it yourself: `uv run --extra llm python -m qj.llm.train_demo`
Watch loss/ppl fall and coherent text appear.

## 3. Scaling ideas (the leap to real LLMs)

MiniGPT is ~160k params. Real LLMs scale the *same* design several ways:

- **Bigger**: more layers/dims/data (GPT-2 1.5B → GPT-4 multi-hundred-B)
- **Tokenizers**: subword/BPE so a large vocab of tokens cover any text
- **MLP/attn tricks**: feed-forward ratio (~4×d_model), GELU, RMSNorm,
  RoPE positional-encoding, KV-cache to speed inference generation
- **Alignment**: RLHF / preference tuning on top of the pretrained base
- **Efficiency**: quantization, pruning, distillation (cf. layer 06)

## 4. The "everything from tech to LLM" bridge

If you want to go *further* down the stack than this repo shows, the path
into systems engineering is:

- **How models run on hardware**: GPUs/VRAM, memory bandwidth, batch
  size, inference vs training throughput; quantization (fp16 → int8).
- **Data systems**: Parquet/columnar formats, streaming (WS), storage
  engines, distributed compute (the data layer is your entry point).
- **Serving/infra**: queues, containers, model registries, monitoring,
  graceful retries (the mlops/live layers sketch this).
- **Tools**: PyTorch autodiff, NVCC/MLIR, llama.cpp (CPU), vLLM (GPU).

## 5. Your roadmap from here

1. Play with `train_demo.py` — change layers/hidden dim, watch overfitting.
2. Run `qj live` — see the data→model loop *while it happens*.
3. In 07_llm_arch, implement a tokenizer (BPE) next, then KV-cache.
4. Then benchmark a small real model (e.g. Llama 3.2 1B via llama.cpp)
   locally and compare to your MiniGPT's economics.
