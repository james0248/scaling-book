"""Approach 2 for the MoE sweep, run twice: N = active params and N = total params"""

import argparse
import os
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from tiny.analyze_1 import FLOPS_BWD_MULTIPLIER, N_TARGET
from tiny.analyze_2 import _human_format, _trim_fit, parse_log_and_metadata
from tiny.plot_style import (ISO_CMAP, RED, annotate_optimum, budget_label,
                             human_format, notable_param_ticks)


def fit_isoflops_huber(runs: pd.DataFrame):
    """Per-budget parabola with one-sided outlier rejection (same estimator as tiny.analyze_2)"""
    results = []
    for budget, g in runs.groupby("budget"):
        log_n = np.log(g["params"].to_numpy())
        y = g["loss"].to_numpy()

        (a, b, c), keep = _trim_fit(log_n, y)
        log_n_opt = -b / (2 * a) if a > 1e-6 else np.nan
        pt = np.polyfit(log_n, np.log(g["tokens"].to_numpy()), 1)
        results.append({
            "budget": budget,
            "N_opt": float(np.exp(log_n_opt)) if np.isfinite(log_n_opt) else np.nan,
            "D_opt": float(np.exp(np.polyval(pt, log_n_opt))) if np.isfinite(log_n_opt) else np.nan,
            "loss_min": float(np.polyval([a, b, c], log_n_opt)) if np.isfinite(log_n_opt) else np.nan,
            "coeffs": (a, b, c),
            "runs": g,
            "keep": keep,
        })
    return sorted(results, key=lambda r: r["budget"])


def load_runs(run_dir: str) -> pd.DataFrame:
    rows = []
    for name in os.listdir(run_dir):
        p = f"{run_dir}/{name}"
        if not os.path.isdir(p):
            continue
        if not os.path.exists(f"{p}/logs.jsonl"):
            continue
        data, meta = parse_log_and_metadata(p)
        loss = gaussian_filter1d(data["loss"], sigma=20, mode="nearest")
        rows.append({
            "active_params": float(meta["model"]["params"]),
            "total_params": float(meta["model"]["total_params"]),
            "budget": float(meta["total_flops"]) * FLOPS_BWD_MULTIPLIER,
            "tokens": float(data["tokens"][-1]),
            "loss": float(loss[-1]),
        })
    return pd.DataFrame(rows)


def run_view(df: pd.DataFrame, param_col: str, view_name: str, out_root: str = "plots/approach_2_moe"):
    view_df = df.rename(columns={param_col: "params"})
    results = fit_isoflops_huber(view_df)

    out_root = Path(out_root) / view_name
    per_dir = out_root / "a2_per_budget"
    shutil.rmtree(per_dir, ignore_errors=True)
    per_dir.mkdir(parents=True, exist_ok=True)

    # Overlaid isoflop
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for i, r in enumerate(results):
        color = ISO_CMAP(i / max(len(results) - 1, 1))
        g = r["runs"]
        ax.scatter(g["params"], g["loss"], color=color, s=22, lw=0, zorder=3,
                   label=budget_label(r["budget"]))
        xs = np.linspace(np.log(g["params"].min()), np.log(g["params"].max()), 100)
        ax.plot(np.exp(xs), np.polyval(r["coeffs"], xs), color=color, lw=0.8, ls="--",
                alpha=0.9)
        ax.scatter([r["N_opt"]], [r["loss_min"]], color=color, s=90, marker="*",
                   edgecolor="black", linewidth=0.5, zorder=4)
    ax.set_xscale("log")
    notable_param_ticks(ax, view_df["params"].min() / 1.4, view_df["params"].max() * 1.4)
    ax.set_xlabel(view_name.replace("_", " ").capitalize())
    ax.set_ylabel("Training loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, handletextpad=0.2, borderaxespad=0.4)
    fig.tight_layout()
    fig.savefig(out_root / "a2_isoflop.png", dpi=150)
    plt.close(fig)

    # Per-budget
    for r in results:
        g = r["runs"]
        fig1, ax1 = plt.subplots(figsize=(4.6, 3.5))
        keep = r["keep"]
        ax1.scatter(g["params"][keep], g["loss"][keep], s=24, color=ISO_CMAP(0.75), lw=0,
                    zorder=3, label=f"used ({int(keep.sum())})")
        if (~keep).any():
            ax1.scatter(g["params"][~keep], g["loss"][~keep], s=60, facecolor="none",
                        edgecolor=RED, linewidth=1.0, zorder=4,
                        label=f"dropped as outlier ({int((~keep).sum())})")
        xs = np.linspace(np.log(g["params"].min()), np.log(g["params"].max()), 100)
        ax1.plot(np.exp(xs), np.polyval(r["coeffs"], xs), color=RED, lw=0.9, ls="--")
        x_lo, x_hi = g["params"].min() / 1.3, g["params"].max() * 1.3
        ax1.set_xlim(x_lo, x_hi)
        ax1.scatter([r["N_opt"]], [r["loss_min"]], color=RED, s=110, marker="*",
                    edgecolor="black", linewidth=0.5, zorder=4,
                    label=f"N$_{{opt}}$ = {human_format(r['N_opt'])}, L={r['loss_min']:.3f}")
        ax1.set_xscale("log")
        notable_param_ticks(ax1, x_lo, x_hi)
        ax1.set_xlabel(view_name.replace("_", " ").capitalize())
        ax1.set_ylabel("Training loss")
        a, _, _ = r["coeffs"]
        ax1.set_title(f"C = {budget_label(r['budget'])}   (a={a:+.4f}, {len(g)} runs)")
        ax1.grid(alpha=0.3)
        ax1.legend(loc="best", fontsize=8)
        fig1.tight_layout()
        fig1.savefig(per_dir / f"{r['budget']:.0e}.png", dpi=150)
        plt.close(fig1)

    # Frontier, dropping concave / vertex-out-of-range budgets
    kept = []
    dropped = []
    for r in results:
        a, _, _ = r["coeffs"]
        g = r["runs"]
        if a > 0 and g["params"].min() <= r["N_opt"] <= g["params"].max():
            kept.append(r)
        else:
            dropped.append((r, a))

    if len(kept) < 2:
        print(f"[{view_name}] too few valid budgets to fit frontier")
        return

    C = np.array([r["budget"] for r in kept])
    No = np.array([r["N_opt"] for r in kept])
    Do = np.array([r["D_opt"] for r in kept])
    log_c = np.log(C)
    sN, iN = np.polyfit(log_c, np.log(No), 1)
    sD, iD = np.polyfit(log_c, np.log(Do), 1)
    c_star = (N_TARGET / np.exp(iN)) ** (1 / sN)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    fit_x = np.array([C.min() / 3, c_star * 30])
    for ax, y, sym, s, i, ylab in [(axes[0], No, "N", sN, iN, "Parameters"),
                                   (axes[1], Do, "D", sD, iD, "Tokens")]:
        y_star = np.exp(i) * c_star**s
        ax.plot(fit_x, np.exp(i) * fit_x ** s, color=RED, lw=0.9, ls="--",
                label=f"{sym}$_{{opt}}$ = {np.exp(i):.2e}$\\cdot C^{{{s:.3f}}}$")
        ax.scatter(C, y, s=28, color="black", zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*fit_x)
        ax.set_ylim(y.min() / 3, y_star * 30)
        if sym == "N":
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(human_format))
            ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        annotate_optimum(ax, c_star, y_star, human_format(y_star))
        ax.set_xlabel("FLOPs")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_root / "a2_frontier.png", dpi=150)
    plt.close(fig)

    print(f"\n[{view_name}] per-budget isoFLOP fits")
    print(f"{'C':>8} {'runs':>5} {'a':>8} {'N_opt':>9} {'L_min':>7} {'N range':>18}  {'spread':>7}")
    for r in results:
        g = r["runs"]
        a, _, _ = r["coeffs"]
        ok = "ok" if r in kept else "DROP"
        print(f"{r['budget']:8.0e} {len(g):5d} {a:+8.4f} {_human_format(r['N_opt']):>9} "
              f"{r['loss_min']:7.3f} {_human_format(g['params'].min()):>8}"
              f"–{_human_format(g['params'].max()):<9} "
              f"{g['loss'].max() - g['loss'].min():7.3f}  {ok}")
    print(f"\n[{view_name}] N_opt ∝ C^{sN:.4f}   D_opt ∝ C^{sD:.4f}   "
          f"(kept {len(kept)}/{len(results)} budgets)")
    for r, a in dropped:
        g = r["runs"]
        why = "concave" if a <= 0 else "vertex out of range"
        print(f"  dropped C={r['budget']:.0e}: {why} (a={a:+.4f}, N_opt={r['N_opt']:.0f}, "
              f"range=[{g['params'].min():.0f}, {g['params'].max():.0f}])")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="outputs/approach_2_moe")
    parser.add_argument("--out_root", default="plots/approach_2_moe")
    parser.add_argument("--exclude", default="", help="comma-separated training-FLOP "
                        "budgets to drop, e.g. 9e14,1.8e15 (matched within 1%)")
    args = parser.parse_args()

    df = load_runs(args.run_dir)
    print(f"loaded {len(df)} MoE runs from {args.run_dir}")
    for b in (float(x) for x in args.exclude.split(",") if x.strip()):
        drop = np.isclose(df["budget"], b, rtol=0.01)
        print(f"excluding budget C={b:.1e} ({int(drop.sum())} runs)")
        df = df[~drop]
    run_view(df, "active_params", "active_params", args.out_root)
    run_view(df, "total_params", "total_params", args.out_root)


if __name__ == "__main__":
    main()
