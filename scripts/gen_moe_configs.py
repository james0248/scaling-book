"""
Generate MoE model configs matching the dense family.

For each config/model/*.yaml (dense Transformer), write config/model/moe_*.yaml
using MixtureOfExpertsTransformer with n_experts=4, k_experts=1 and the same
architectural knobs (n_layers, d_model, d_ffw, n_heads, n_kv, vocab_size).

`params` stored in the yaml is the *active* param count (what analysis uses as N).
`total_params` records the full model size (~4x dense on the MLP portion).
`flops_per_token` is the active forward-pass FLOPs/token including router overhead.
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import yaml

from tiny.model import MixtureOfExpertsTransformer

N_EXPERTS = 4
K_EXPERTS = 1
MAX_DIGITS = 6
SEQ_LEN = 4 * MAX_DIGITS + 2  # matches training seq layout

MODEL_DIR = Path(__file__).resolve().parent.parent / "tiny" / "config" / "model"


def build_moe(n_layers, d_model, d_ffw, n_heads, n_kv, vocab_size):
    return MixtureOfExpertsTransformer(
        n_layers=n_layers,
        d_model=d_model,
        d_ffw=d_ffw,
        n_heads=n_heads,
        n_kv=n_kv,
        vocab_size=vocab_size,
        n_experts=N_EXPERTS,
        k_experts=K_EXPERTS,
    )


EXPERT_KEYS = ("up_kernel", "up_bias", "gate_kernel", "gate_bias", "down_kernel", "down_bias")


def count_params(params):
    total = 0
    active = 0
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        name = "/".join(str(k) for k in path)
        size = int(leaf.size)
        total += size
        if any(k in name for k in EXPERT_KEYS):
            # Expert MLP weights have leading dim = n_experts; only k are active per token
            per_expert = size // N_EXPERTS
            active += per_expert * K_EXPERTS
        else:
            active += size
    return active, total


def flops_per_token(n_layers, d_model, d_ffw, n_heads, n_kv, vocab_size):
    """
    Forward FLOPs/token for the MoE model (active, k=1 of n=4).

    Same formula as dense (see reports/chinchilla/FLOPS.md) plus per-layer router overhead:
      router linear:  2·S·D·n_experts
      router softmax: 5·S·k_experts        (top_k selection cost ignored)
      weight mul:     2·S·D·k_experts      (weights[..., None] * expert_out sum)
    """
    S, V, D, F, L = SEQ_LEN, vocab_size, d_model, d_ffw, n_layers
    Nq, Nkv = n_heads, n_kv
    H = D // Nq

    embed = 2 * S * V * D
    final_norm = 4 * S * D
    logits = 2 * S * V * D

    attn_matmul = 2 * S * D * H * (Nq + 2 * Nkv) + 2 * S * S * (H * Nq) * 2 + 2 * S * (H * Nq) * D
    attn_norm = (
        4 * S * D
        + 4 * S * Nq * H
        + 4 * S * Nkv * H
        + 3 * S * (Nq + Nkv) * H
        + 3 * S * H
        + S * S * Nq
        + 5 * Nq * S * S
    )
    attn_layer = attn_matmul + attn_norm

    mlp_matmul = 6 * S * D * F
    mlp_norm = 4 * S * D + 6 * S * F
    mlp_layer_dense = mlp_matmul + mlp_norm

    router = 2 * S * D * N_EXPERTS + 5 * S * K_EXPERTS + 2 * S * D * K_EXPERTS

    residual = 2 * S * D

    F_fwd = (
        embed
        + L * (attn_layer + mlp_layer_dense + router + residual)
        + final_norm
        + logits
    )
    return F_fwd // S


def main():
    dense_files = sorted(MODEL_DIR.glob("*.yaml"))
    dense_files = [f for f in dense_files if not f.name.startswith("moe_")]
    print(f"Generating MoE configs for {len(dense_files)} dense models")

    rng = jax.random.key(0)
    for f in dense_files:
        with f.open() as fh:
            src = yaml.safe_load(fh)
        cfg = src["config"]
        assert cfg["_target_"] == "tiny.model.Transformer", f
        n_layers = cfg["n_layers"]
        d_model = cfg["d_model"]
        d_ffw = cfg["d_ffw"]
        n_heads = cfg["n_heads"]
        n_kv = cfg["n_kv"]
        vocab_size = cfg["vocab_size"]

        model = build_moe(n_layers, d_model, d_ffw, n_heads, n_kv, vocab_size)
        inp = jnp.zeros((1, SEQ_LEN), dtype=jnp.int32)
        params = model.init(rng, inp)
        active, total = count_params(params)
        fpt = flops_per_token(n_layers, d_model, d_ffw, n_heads, n_kv, vocab_size)

        out_name = MODEL_DIR / f"moe_{f.name}"
        out = {
            "learning_rate": src["learning_rate"],
            "flops_per_token": int(fpt),
            "params": int(active),
            "total_params": int(total),
            "config": {
                "_target_": "tiny.model.MixtureOfExpertsTransformer",
                "n_layers": n_layers,
                "d_model": d_model,
                "d_ffw": d_ffw,
                "n_heads": n_heads,
                "n_kv": n_kv,
                "vocab_size": vocab_size,
                "n_experts": N_EXPERTS,
                "k_experts": K_EXPERTS,
            },
        }
        with out_name.open("w") as fh:
            yaml.safe_dump(out, fh, sort_keys=False)
        print(f"  {f.stem:>6s} -> {out_name.name:>14s}  active={active:>9d}  total={total:>9d}  fpt={fpt}")


if __name__ == "__main__":
    main()
