import argparse
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter1d


def parse_log_and_metadata(path: str):
    with open(f"{path}/logs.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
        data = {key: np.array([item[key] for item in data]) for key in data[0].keys()}

    with open(f"{path}/.hydra/config.yaml", "r", encoding="utf-8") as f:
        try:
            metadata = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")

    return data, metadata


def _human_format(x, pos=None):
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            v = x / div
            return f"{v:.0f}{suf}" if v == int(v) else f"{v:.1f}{suf}"
    return f"{x:.0f}"


def plot_loss_curves(loss_map: dict[str, dict], lowest_rows: pd.DataFrame):
    params = np.array([d["params"] for d in loss_map.values()], dtype=float)
    norm = mpl.colors.LogNorm(vmin=params.min(), vmax=params.max())
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(8, 5))
    for data in loss_map.values():
        color = cmap(norm(data["params"]))
        ax.plot(data["flops"], data["loss"], color=color, lw=1, alpha=0.8)

    # Efficient frontier: lowest-loss point in each FLOPs bin.
    ax.scatter(lowest_rows["flops"], lowest_rows["loss"], s=10, color="gray", zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_xlim(left=1e11)
    # ax.set_ylim(top=1.0)
    ax.set_xlabel("FLOPs")
    ax.set_ylabel("loss")
    ax.set_title("Approach 1: loss curves colored by param count")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, ticks=[2e4, 5e4, 2e5, 5e5, 1e6, 3e6])
    cbar.minorticks_off()
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_human_format))
    cbar.set_label("params")

    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fig.savefig("plots/approach_1_loss_curves.png", dpi=150)


def plot_frontier(lowest_rows: pd.DataFrame) -> dict[str, tuple[float, float]]:
    flops = lowest_rows["flops"].to_numpy()
    log_flops = np.log(flops)

    fits = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, sym in [(axes[0], "params", "N"), (axes[1], "tokens", "D")]:
        y = lowest_rows[col].to_numpy()
        slope, intercept = np.polyfit(log_flops, np.log(y), 1)
        fits[sym] = (slope, intercept)

        # Fit line evaluated in log space, drawn against raw (log-scaled) axes.
        fit_x = np.array([flops.min(), flops.max()])
        fit_y = np.exp(intercept) * fit_x**slope

        ax.scatter(flops, y, s=12, alpha=0.6, label="per-bin best")
        ax.plot(
            fit_x,
            fit_y,
            color="red",
            lw=2,
            label=f"fit: {sym} = {np.exp(intercept):.2e}·C^{slope:.3f}",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        if col == "params":
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_human_format))
        ax.set_xlabel("FLOPs")
        ax.set_ylabel(col)
        ax.set_title(f"{sym} ∝ C^{slope:.3f}")
        ax.legend()
        print(f"{sym} ∝ C^{slope:.4f}  (slope={slope:.4f}, intercept={intercept:.4f})")

    fig.suptitle("Approach 1: compute-optimal frontier")
    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fig.savefig("plots/approach_1_frontier.png", dpi=150)
    return fits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()

    # Parse loss logs
    loss_map = {}
    for config_name in os.listdir(args.run_dir):
        if not os.path.isdir(f"{args.run_dir}/{config_name}"):
            continue

        data, metadata = parse_log_and_metadata(f"{args.run_dir}/{config_name}")
        data["loss"] = gaussian_filter1d(data["loss"], sigma=5, radius=10, mode="nearest")
        data["params"] = metadata["model"]["params"]
        data["flops"] = data["tokens"] * metadata["model"]["flops_per_token"]
        loss_map[config_name] = data

    # Find the parameter size / token count with the lowest loss in each FLOPs bin
    df = (
        pd.DataFrame(loss_map.values())
        .explode(column=["loss", "flops", "tokens"])
        .reset_index(drop=True)
    )
    df["loss"] = df["loss"].astype(np.float32)
    df["flops"] = df["flops"].astype(np.float32)
    df["params"] = df["params"].astype(np.float32)
    df["tokens"] = df["tokens"].astype(np.float32)
    df["flops_bins"] = pd.cut(np.log(df["flops"]), bins=1500)

    # Keep non-saturated band, and only the frontier from 1e13 FLOPs onward
    frontier_df = df[(df["loss"] <= 1.0) & (df["loss"] >= 5e-3) & (df["flops"] >= 1e13)]
    lowest_loss_idx = frontier_df.groupby("flops_bins", observed=True)["loss"].idxmin()
    lowest_rows = frontier_df.loc[lowest_loss_idx]

    # Plot 1: Training curves + efficient frontier
    plot_loss_curves(loss_map, lowest_rows)
    # Plot 2/3: FLOPs vs Parameters/Tokens power-law fits on the frontier
    plot_frontier(lowest_rows)


if __name__ == "__main__":
    main()
