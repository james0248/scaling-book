"""Approach 1: compute-optimal frontier from the loss envelope"""

import argparse
import collections
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d

from tiny.plot_style import CURVE_CMAP, GRAY, RED, annotate_optimum, human_format


def parse_log_and_metadata(path: str):
    with open(f"{path}/logs.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
        data = {key: np.array([item[key] for item in data]) for key in data[0].keys()}

    with open(f"{path}/.hydra/config.yaml", "r", encoding="utf-8") as f:
        try:
            metadata = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")

    # logged `tokens` used the old 3*(d+1) addition layout in early runs
    seq_len = 4 * metadata["data"]["max_digits"] + 2
    steps = np.arange(1, len(data["loss"]) + 1, dtype=np.float64)
    data["tokens"] = steps * metadata["batch_size"] * seq_len

    return data, metadata


# config flops_per_token is forward-only; backward costs ~2x forward
FLOPS_BWD_MULTIPLIER = 3


def _human_format(x, pos=None):
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            v = x / div
            return f"{v:.0f}{suf}" if v == int(v) else f"{v:.1f}{suf}"
    return f"{x:.0f}"


def load_runs(run_dir: str, params_key: str, sigma: float) -> dict[str, dict]:
    loss_map = {}
    for config_name in sorted(os.listdir(run_dir)):
        path = f"{run_dir}/{config_name}"
        if not os.path.isdir(path) or not os.path.exists(f"{path}/logs.jsonl"):
            continue
        data, metadata = parse_log_and_metadata(path)
        data["loss"] = gaussian_filter1d(data["loss"], sigma=sigma, mode="nearest")
        data["params"] = float(metadata["model"][params_key])
        data["flops"] = (data["tokens"] * metadata["model"]["flops_per_token"]
                         * FLOPS_BWD_MULTIPLIER)
        data["steps"] = len(data["loss"])
        loss_map[config_name] = data
    return loss_map


def build_envelope(loss_map: dict[str, dict], n_grid: int = 1500, min_models: int = 10):
    """Lowest interpolated loss at n_grid log-spaced FLOP values,
    truncated where fewer than min_models sizes still compete"""
    cmin = min(d["flops"][-1] for d in loss_map.values())
    cmax = max(d["flops"][-1] for d in loss_map.values())
    grid = np.logspace(np.log10(cmin), np.log10(cmax), n_grid)
    log_grid = np.log(grid)

    loss = np.full(n_grid, np.inf)
    params = np.zeros(n_grid)
    tokens = np.zeros(n_grid)
    frac = np.zeros(n_grid)
    owner = np.empty(n_grid, dtype=object)
    competing = [set() for _ in range(n_grid)]

    for name, d in loss_map.items():
        log_flops = np.log(d["flops"])
        inside = (log_grid >= log_flops[0]) & (log_grid <= log_flops[-1])
        if not inside.any():
            continue
        idx = np.where(inside)[0]
        for i in idx:
            competing[i].add(d["params"])
        vals = np.interp(log_grid[idx], log_flops, d["loss"])
        better = vals < loss[idx]
        sel = idx[better]
        loss[sel] = vals[better]
        params[sel] = d["params"]
        tokens[sel] = np.interp(log_grid[sel], log_flops, d["tokens"])
        frac[sel] = np.interp(log_grid[sel], log_flops,
                              np.arange(1, d["steps"] + 1)) / d["steps"]
        owner[sel] = name

    n_competing = np.array([len(c) for c in competing])
    ok = (params > 0) & (n_competing >= min_models)
    return dict(flops=grid[ok], loss=loss[ok], params=params[ok],
                tokens=tokens[ok], frac=frac[ok], owner=owner[ok],
                n_competing=n_competing[ok])


def fit_frontier(env: dict) -> dict[str, tuple[float, float]]:
    log_c = np.log(env["flops"])
    return {sym: tuple(np.polyfit(log_c, np.log(env[col]), 1))
            for col, sym in (("params", "N"), ("tokens", "D"))}


def report_diagnostics(env: dict):
    frac = env["frac"]
    print(f"\nenvelope: {len(frac)} grid points, "
          f"C = {env['flops'].min():.2e} .. {env['flops'].max():.2e}")
    print(f"  distinct model sizes competing: {env['n_competing'].min()} "
          f"(at the top) .. {env['n_competing'].max()} (at the bottom)")
    print("  paper's validity check (their footnote 4: 100% in the last 15% of a run)")
    print(f"    selected in last 15% of its run : {100 * (frac >= 0.85).mean():5.1f}%")
    print(f"    selected in last 50%            : {100 * (frac >= 0.50).mean():5.1f}%")
    print(f"    median fraction-of-training     : {np.median(frac):5.2f}")
    print("  most of the envelope is owned by:")
    for name, count in collections.Counter(env["owner"]).most_common(4):
        m = env["owner"] == name
        print(f"    {name:<22} {count:5d} pts ({100 * count / len(frac):4.1f}%)  "
              f"C = {env['flops'][m].min():.1e} .. {env['flops'][m].max():.1e}")


def plot_envelope(loss_map, env, prefix):
    params = np.array([d["params"] for d in loss_map.values()])
    norm = mpl.colors.LogNorm(vmin=params.min(), vmax=params.max())
    loss_ticks = [0.8, 1.0, 1.2, 1.6, 2.0, 2.4]

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for d in sorted(loss_map.values(), key=lambda d: -d["params"]):
        ax.plot(d["flops"], d["loss"], color=CURVE_CMAP(norm(d["params"])), lw=0.6, alpha=0.8)
    ax.scatter(env["flops"], env["loss"], color=GRAY, s=5, lw=0, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(env["flops"].min() / 2, env["flops"].max() * 2)
    ax.set_ylim(env["loss"].min() * 0.96, env["loss"].max() * 1.06)
    ax.set_yticks([t for t in loss_ticks
                   if env["loss"].min() * 0.96 <= t <= env["loss"].max() * 1.06])
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, p: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("FLOPs")
    ax.set_ylabel("Training loss")
    ax.grid(alpha=0.3)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=CURVE_CMAP)
    cbar = fig.colorbar(sm, ax=ax, ticks=[5e3, 2e4, 1e5, 5e5, 3e6])
    cbar.minorticks_off()
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(human_format))
    cbar.ax.tick_params(width=0.6)
    cbar.outline.set_linewidth(0.6)

    out = f"plots/{prefix}_envelope.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


N_TARGET = 1e7  # the ~10M transformer of the original exercise


def plot_frontier(env, fits, prefix):
    slope_n, icept_n = fits["N"]
    c_star = (N_TARGET / np.exp(icept_n)) ** (1 / slope_n)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, col, sym in ((axes[0], "params", "N"), (axes[1], "tokens", "D")):
        slope, intercept = fits[sym]
        y_star = np.exp(intercept) * c_star**slope
        x = np.array([env["flops"].min() / 3, c_star * 30])
        ax.scatter(env["flops"], env[col], s=5, color=GRAY, lw=0)
        ax.plot(x, np.exp(intercept) * x**slope, color=RED, lw=0.9, ls="--",
                label=f"{sym} = {np.exp(intercept):.2e}$\\cdot C^{{{slope:.3f}}}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*x)
        ax.set_ylim(env[col].min() / 3, y_star * 30)
        if col == "params":
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(human_format))
            ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        annotate_optimum(ax, c_star, y_star, human_format(y_star))
        ax.set_xlabel("FLOPs")
        ax.set_ylabel({"params": "Parameters", "tokens": "Tokens"}[col])
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = f"plots/{prefix}_frontier.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, nargs="+")
    parser.add_argument("--prefix", default="approach_1")
    parser.add_argument("--params_key", default="params")
    parser.add_argument("--n_grid", type=int, default=1500)
    parser.add_argument("--min_models", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=20.0)
    parser.add_argument("--max_flops", type=float, default=None)
    args = parser.parse_args()

    loss_map = {}
    for i, run_dir in enumerate(args.run_dir):
        for name, run in load_runs(run_dir, args.params_key, args.sigma).items():
            loss_map[f"{i}:{name}"] = run
    print(f"{len(loss_map)} runs from {', '.join(args.run_dir)} "
          f"(horizons: {sorted({d['steps'] for d in loss_map.values()})})")

    env = build_envelope(loss_map, args.n_grid, args.min_models)
    if args.max_flops is not None:
        keep = env["flops"] <= args.max_flops
        env = {k: v[keep] for k, v in env.items()}
        print(f"restricted to C <= {args.max_flops:.1e} ({keep.sum()} grid points)")

    report_diagnostics(env)
    fits = fit_frontier(env)
    print()
    for sym in ("N", "D"):
        print(f"{sym} ∝ C^{fits[sym][0]:.4f}  (intercept={fits[sym][1]:.4f})")

    plot_envelope(loss_map, env, args.prefix)
    plot_frontier(env, fits, args.prefix)


if __name__ == "__main__":
    main()
