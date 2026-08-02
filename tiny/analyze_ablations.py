import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from tiny.analyze_1 import FLOPS_BWD_MULTIPLIER, parse_log_and_metadata
from tiny.plot_style import FRONTIER_BLUE, GRAY, RED, TEAL

VARIANTS = [
    ("rope", "RoPE (baseline)", FRONTIER_BLUE),
    ("absolute", "Absolute PE", TEAL),
    ("nope", "NoPE", RED),
    ("noqk", "RoPE, no QK-norm", GRAY),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="outputs/ablations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=20.0)
    parser.add_argument("--out", default="plots/03_ablations.png")
    args = parser.parse_args()

    runs = {}
    for name, label, color in VARIANTS:
        data, metadata = parse_log_and_metadata(f"{args.run_dir}/{name}_s{args.seed}")
        data["flops"] = data["tokens"] * metadata["model"]["flops_per_token"] * FLOPS_BWD_MULTIPLIER
        data["loss_s"] = gaussian_filter1d(data["loss"], sigma=args.sigma, mode="nearest")
        data["acc_s"] = gaussian_filter1d(data["acc"], sigma=args.sigma, mode="nearest")
        data["exact_s"] = gaussian_filter1d(data["exact"], sigma=args.sigma, mode="nearest")
        runs[name] = data

    for name, label, color in VARIANTS:
        d = runs[name]
        below = np.where(d["loss_s"] < 1.0)[0]
        grok = (
            f"loss<1.0 at step {below[0] + 1} (C={d['flops'][below[0]]:.1e})"
            if len(below)
            else "never below 1.0"
        )
        print(
            f"{name:<9} final loss {d['loss_s'][-1]:.3f}  final acc {d['acc_s'][-1]:.3f}  "
            f"max exact {d['exact_s'].max():.4f}  {grok}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for name, label, color in VARIANTS:
        d = runs[name]
        axes[0].plot(d["flops"], d["loss_s"], color=color, label=label)
        axes[1].plot(d["flops"], d["acc_s"], color=color, label=label)

    axes[0].set_yscale("log")
    axes[0].set_ylabel("Training loss")
    yticks = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 2.4]
    lo = min(d["loss_s"].min() for d in runs.values())
    axes[0].set_yticks([t for t in yticks if t >= lo * 0.9])
    axes[0].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, p: f"{v:g}"))
    axes[0].yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axes[1].set_ylabel("Token accuracy")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("FLOPs")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
