# scaling-book

Exercises from [How to Scale Your Model](https://jax-ml.github.io/scaling-book/), plus a scaling-law study on tiny transformers.

- `scaling_book/` – exercise notebooks (ch. 9–10)
- `tiny/` – transformers (5K–3M params) written from scratch in JAX/Flax, trained on 6-digit multiplication
- `reports/chinchilla/` – Chinchilla scaling laws derived from those runs ([pdf](reports/chinchilla/chinchilla.pdf))
- `reports/moe/` – a fused Pallas kernel for the MoE MLP ([pdf](reports/moe/moe.pdf))
- `scripts/` – sweep launchers and figure generation

```sh
uv run python -m tiny.train -cn approach_1/4000 -m   # run a sweep
bash scripts/make_plots.sh                           # regenerate all figures
```
