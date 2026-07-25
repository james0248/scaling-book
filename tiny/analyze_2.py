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


def fit_isoflops(runs: pd.DataFrame) -> pd.DataFrame:
    # For each FLOP budget, fit loss = a*(logN)^2 + b*logN + c and take the vertex.
    results = []
    for budget, g in runs.groupby("budget"):
        log_n = np.log(g["params"].to_numpy())
        a, b, c = np.polyfit(log_n, g["loss"].to_numpy(), 2)
        log_n_opt = -b / (2 * a)

        # D_opt by interpolating log(tokens) vs log(params) at the optimum
        pt = np.polyfit(log_n, np.log(g["tokens"].to_numpy()), 1)
        results.append(
            {
                "budget": budget,
                "N_opt": np.exp(log_n_opt),
                "D_opt": np.exp(np.polyval(pt, log_n_opt)),
                "loss_min": np.polyval([a, b, c], log_n_opt),
                "coeffs": (a, b, c),
                "runs": g,
            }
        )
    return sorted(results, key=lambda r: r["budget"])


def plot_isoflops(results: list[dict]):
    budgets = np.array([r["budget"] for r in results])
    norm = mpl.colors.LogNorm(vmin=budgets.min(), vmax=budgets.max())
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        color = cmap(norm(r["budget"]))
        g = r["runs"]
        ax.scatter(g["params"], g["loss"], color=color, s=18, zorder=3)
        xs = np.linspace(np.log(g["params"].min()), np.log(g["params"].max()), 100)
        ax.plot(np.exp(xs), np.polyval(r["coeffs"], xs), color=color, lw=1.2, alpha=0.8)
        ax.scatter([r["N_opt"]], [r["loss_min"]], color=color, s=90, marker="*",
                   edgecolor="black", linewidth=0.5, zorder=4)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_human_format))
    ax.set_xlabel("params")
    ax.set_ylabel("final loss")
    ax.set_title("Approach 2: IsoFLOP profiles (★ = optimum)")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("FLOP budget")

    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fig.savefig("plots/approach_2_isoflop.png", dpi=150)


def plot_frontier(results: list[dict]) -> dict[str, tuple[float, float]]:
    budgets = np.array([r["budget"] for r in results])

    fits = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, sym in [(axes[0], "N_opt", "N"), (axes[1], "D_opt", "D")]:
        y = np.array([r[key] for r in results])
        slope, intercept = np.polyfit(np.log(budgets), np.log(y), 1)
        fits[sym] = (slope, intercept)

        fit_x = np.array([budgets.min(), budgets.max()])
        ax.scatter(budgets, y, s=40, color="black", zorder=3)
        ax.plot(
            fit_x,
            np.exp(intercept) * fit_x**slope,
            color="red",
            lw=2,
            label=f"{sym}$_{{opt}}$ = {np.exp(intercept):.2e}·C^{slope:.3f}",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        if key == "N_opt":
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_human_format))
        ax.set_xlabel("FLOPs")
        ax.set_ylabel(f"{sym}$_{{opt}}$")
        ax.set_title(f"{sym}$_{{opt}}$ ∝ C^{slope:.3f}")
        ax.legend()
        print(f"{sym}_opt ∝ C^{slope:.4f}  (slope={slope:.4f}, intercept={intercept:.4f})")

    fig.suptitle("Approach 2: compute-optimal frontier")
    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fig.savefig("plots/approach_2_frontier.png", dpi=150)
    return fits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()

    # Parse each run's final loss and its fixed FLOP budget
    rows = []
    for config_name in os.listdir(args.run_dir):
        if not os.path.isdir(f"{args.run_dir}/{config_name}"):
            continue

        data, metadata = parse_log_and_metadata(f"{args.run_dir}/{config_name}")
        loss = gaussian_filter1d(data["loss"], sigma=5, radius=10, mode="nearest")
        rows.append(
            {
                "params": float(metadata["model"]["params"]),
                "budget": float(metadata["total_flops"]),
                "tokens": float(data["tokens"][-1]),
                "loss": float(loss[-1]),
            }
        )
    runs = pd.DataFrame(rows)

    # For each budget find N_opt via parabola vertex, then fit power laws vs FLOPs
    results = fit_isoflops(runs)
    plot_isoflops(results)
    plot_frontier(results)


if __name__ == "__main__":
    main()
