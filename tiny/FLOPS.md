# FLOPs computation

How `flops_per_token` in each `config/model/*.yaml` is derived. The value is the
**forward-pass FLOPs per token** for the `Transformer` in [`model.py`](model.py),
computed exactly for our architecture (not the generic textbook formula).

## Setup / conventions

For the 3-digit addition task the sequence is always `lhs + rhs = answer`, laid out as
`3 + 1 + 3 + 1 + 4 = 12` tokens, so `seq_len (S) = 12` and `vocab_size (V) = 12`.

Symbols:

| symbol | meaning                       |
|--------|-------------------------------|
| `S`    | seq_len = 12                  |
| `V`    | vocab_size = 12               |
| `D`    | `d_model`                     |
| `F`    | `d_ffw`                       |
| `L`    | `n_layers`                    |
| `Nq`   | `n_heads` (query heads)       |
| `Nkv`  | `n_kv` (key/value heads, GQA) |
| `H`    | `d_head = D // Nq` (= 8 here) |

Counting rules (1 FLOP = one scalar op):

- **Matmuls** use the factor **2** (multiply + accumulate).
- Every elementwise op (`add`, `mul`, `sub`, `div`, `exp`, `sin`, `cos`, `rsqrt`, …)
  counts as **1 FLOP**. This is what makes the normalization/activation terms below
  non-zero — the generic formula drops them.
- Embedding lookup is billed the standard `2·S·V·D` matmul-equivalent (it is really a
  gather ≈ 0 FLOPs; kept for the usual param↔FLOP correspondence).

## Per-component FLOPs (whole sequence)

### Matmul terms

| component                     | FLOPs                       |
|-------------------------------|-----------------------------|
| Embeddings                    | `2·S·V·D`                   |
| **Attention, per layer**      |                             |
| &nbsp;&nbsp;Q/K/V proj (GQA)  | `2·S·D·H·(Nq + 2·Nkv)`      |
| &nbsp;&nbsp;Q@K logits        | `2·S·S·(H·Nq)`              |
| &nbsp;&nbsp;softmax @ V       | `2·S·S·(H·Nq)`              |
| &nbsp;&nbsp;output proj       | `2·S·(H·Nq)·D`              |
| **MLP (SwiGLU), per layer**   | `6·S·D·F`  (up + gate + down) |
| Final logits                  | `2·S·V·D`                   |

> SwiGLU is **3** matmuls, not the 2 of a vanilla MLP → `6·S·D·F`, not `4·S·D·F`.
> GQA means Q has `Nq` heads while K/V have `Nkv` (in every current config `Nkv == Nq`).

### Norm / elementwise terms (what the generic formula ignores)

RMSNorm on a vector of length `d` costs `4d` (square `d`, sum `d`, normalize-mul `d`,
scale-mul `d`).

| component (per layer unless noted) | FLOPs                            |
|------------------------------------|----------------------------------|
| RMSNorm pre-attention              | `4·S·D`                          |
| RMSNorm on Q / on K                | `4·S·Nq·H` + `4·S·Nkv·H`         |
| RoPE on Q and K                    | `3·S·(Nq+Nkv)·H` + `3·S·H`       |
| attention scaling `/√H`            | `S·S·Nq`                         |
| softmax (max+sub+exp+sum+div = 5)  | `5·Nq·S·S`                       |
| RMSNorm pre-MLP                    | `4·S·D`                          |
| SwiGLU swish + gate (5 + 1)        | `6·S·F`                          |
| 2 residual adds                    | `2·S·D`                          |
| RMSNorm final (once, not per layer)| `4·S·D`                          |

## Total

```text
attn_layer = 2·S·D·H·(Nq + 2·Nkv) + 2·S·S·(H·Nq) + 2·S·S·(H·Nq) + 2·S·(H·Nq)·D   # matmul
           + 4·S·D + 4·S·Nq·H + 4·S·Nkv·H + 3·S·(Nq+Nkv)·H + 3·S·H + S·S·Nq + 5·Nq·S·S   # norm/elem

mlp_layer  = 6·S·D·F + 4·S·D + 6·S·F                                             # matmul + norm/elem

residual   = 2·S·D

F_fwd      = 2·S·V·D                         # embeddings
           + L · (attn_layer + mlp_layer + residual)
           + 4·S·D                           # final RMSNorm
           + 2·S·V·D                         # logits

flops_per_token = F_fwd // S                 # <-- stored in the config
```

Because `S` is fixed, per-token is just `F_fwd / S`; storing per-token keeps the
config independent of how it's later multiplied.

### Used by `total_steps`

```yaml
total_steps: ${eval:'${total_flops} // (${model.flops_per_token} * 3 * (${data.max_digits} + 1) * ${batch_size})'}
```

- `* 3` — training FLOPs ≈ 3 × forward (forward + backward).
- `* (max_digits + 1)` — trained tokens per example (the answer span; the mask has
  `max_digits + 1` = 4 ones).
- `* batch_size` — examples per step.

### Budget for a full pass over the dataset

The 3-digit dataset is every `(lhs, rhs)` pair: `10^(2·max_digits) = 1,000,000` sequences.
The training FLOPs to see all of them once (`batch_size` cancels out):

```text
budget = flops_per_token * 3 * (max_digits + 1) * n_sequences
```

For the **smallest model (19k)** over the full **1M** sequences:

```text
budget = 43,200 * 3 * 4 * 1,000,000 = 5.184e11 FLOPs  (≈ 518 GFLOP)
```

Note this convention only bills the `max_digits + 1 = 4` answer tokens per example. If
you instead bill all 12 tokens the model actually runs, it is
`flops_per_token * 12 * 3 * 1e6 = 1.555e12` FLOPs (≈ 1.56 TFLOP).

## Comparison with the `6ND` approximation

The common rule of thumb is `C ≈ 6·N·D` training FLOPs (`N` = params, `D` = tokens),
i.e. `6N` per token. Our per-token training cost is `3 × flops_per_token`, so

```text
ratio = 3·flops_per_token / (6·N) = flops_per_token / (2·N)
```

`N` below is the actual parameter count from `model.init` (includes embeddings, all
RMSNorm scales, and biases).

| Params (N) | n_layers | d_model | ffw_size | n_heads | k/q size | flops/token | 6N | Ratio (Ours / 6ND) |
|-----------:|---------:|--------:|---------:|--------:|---------:|------------:|------------:|:------------------:|
| 19,796     | 2  | 24  | 96   | 3  | 8 | 43,200    | 118,776    | 1.091 |
| 34,572     | 2  | 32  | 128  | 4  | 8 | 73,968    | 207,432    | 1.070 |
| 51,452     | 3  | 32  | 128  | 4  | 8 | 110,120   | 308,712    | 1.070 |
| 105,876    | 4  | 40  | 160  | 5  | 8 | 223,776   | 635,256    | 1.057 |
| 189,212    | 5  | 48  | 192  | 6  | 8 | 396,456   | 1,135,272  | 1.048 |
| 307,604    | 6  | 56  | 224  | 7  | 8 | 640,448   | 1,845,624  | 1.041 |
| 533,708    | 8  | 64  | 256  | 8  | 8 | 1,105,856 | 3,202,248  | 1.036 |
| 934,556    | 9  | 80  | 320  | 10 | 8 | 1,923,176 | 5,607,336  | 1.029 |
| 1,640,444  | 11 | 96  | 384  | 12 | 8 | 3,360,168 | 9,842,664  | 1.024 |
| 3,172,172  | 12 | 128 | 512  | 16 | 8 | 6,459,680 | 19,033,032 | 1.018 |

The ratio is > 1 and shrinks with scale: at small sizes the `O(S²)` attention terms,
embeddings, and per-token normalization are a larger fraction of the total, exactly the
trend in the reference table (which approaches ~1.0 for multi-billion-param models).
