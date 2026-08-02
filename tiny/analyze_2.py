import argparse
import os
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from tiny.analyze_1 import FLOPS_BWD_MULTIPLIER, N_TARGET, parse_log_and_metadata
from tiny.plot_style import (ISO_CMAP, RED, annotate_optimum, budget_label,
                             human_format, notable_param_ticks)


def _human_format(x, pos=None):
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            v = x / div
            return f"{v:.0f}{suf}" if v == int(v) else f"{v:.1f}{suf}"
    return f"{x:.0f}"


def _trim_fit(x, y, k: float = 2.0, iters: int = 3):
    """Parabola with one-sided rejection: grokked runs only pull the profile down"""
    keep = np.ones(len(x), bool)
    coeffs = np.polyfit(x, y, 2)
    for _ in range(iters):
        coeffs = np.polyfit(x[keep], y[keep], 2)
        resid = y - np.polyval(coeffs, x)
        new = resid > -k * resid[keep].std()
        if new.sum() < 4 or (new == keep).all():
            break
        keep = new
    return np.polyfit(x[keep], y[keep], 2), keep


def _fit_slices(slices, pooled: bool):
    """slices: list of (budget, logN, loss) -> list of (a, b, c). pooled shares one `a`"""
    if pooled == "trim":
        return [_trim_fit(x, y)[0] for _, x, y in slices]
    if not pooled:
        return [np.polyfit(x, y, 2) for _, x, y in slices]

    nb = len(slices)
    init = [np.polyfit(x, y, 2) for _, x, y in slices]
    p0 = np.r_[np.median([c[0] for c in init]), [c[1] for c in init], [c[2] for c in init]]

    def residuals(p):
        a, bs, cs = p[0], p[1 : 1 + nb], p[1 + nb :]
        return np.concatenate(
            [a * x**2 + bs[i] * x + cs[i] - y for i, (_, x, y) in enumerate(slices)]
        )

    sol = least_squares(residuals, p0, loss="soft_l1", f_scale=0.03)
    a = sol.x[0]
    return [np.array([a, sol.x[1 + i], sol.x[1 + nb + i]]) for i in range(nb)]


def _vertex(coeffs):
    a, b, _ = coeffs
    if a <= 1e-6:
        return np.nan, np.nan
    log_n = -b / (2 * a)
    return float(np.exp(log_n)), float(np.polyval(coeffs, log_n))


def fit_isoflops(runs: pd.DataFrame, pooled: bool = "trim") -> list[dict]:
    """For each FLOP budget, fit the isoFLOP parabola and take the vertex."""
    groups = [(b, g) for b, g in runs.groupby("budget")]
    slices = [(b, np.log(g["params"].to_numpy()), g["loss"].to_numpy()) for b, g in groups]
    coeffs = _fit_slices(slices, pooled)
    keeps = ([_trim_fit(x, y)[1] for _, x, y in slices] if pooled == "trim"
             else [np.ones(len(x), bool) for _, x, y in slices])

    # the vertex falls between sampled models, so D_opt is interpolated
    tok_fits = [np.polyfit(np.log(g["params"].to_numpy()), np.log(g["tokens"].to_numpy()), 1)
                for _, g in groups]

    results = []
    for i, (budget, g) in enumerate(groups):
        n_opt, loss_min = _vertex(coeffs[i])
        results.append(
            {
                "budget": budget,
                "N_opt": n_opt,
                "D_opt": float(np.exp(np.polyval(tok_fits[i], np.log(n_opt)))),
                "loss_min": loss_min,
                "coeffs": coeffs[i],
                "runs": g,
                "resid_std": float(np.std(g["loss"].to_numpy() - np.polyval(coeffs[i], slices[i][1]))),
                "log_range": float(np.ptp(slices[i][1])),
                "keep": keeps[i],
            }
        )
    return sorted(results, key=lambda r: r["budget"])


def plot_isoflops(results: list[dict], prefix: str = "approach_2"):
    fig, ax = plt.subplots(figsize=(4.9, 3.6))
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
    notable_param_ticks(ax, min(r["runs"]["params"].min() for r in results) / 1.4,
                        max(r["runs"]["params"].max() for r in results) * 1.4)
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Training loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, handletextpad=0.2, borderaxespad=0.4)

    fig.tight_layout()
    out = f"plots/{prefix}_isoflop.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)

    per_dir = f"plots/{prefix}_per_budget"
    shutil.rmtree(per_dir, ignore_errors=True)
    os.makedirs(per_dir, exist_ok=True)
    for r in results:
        g = r["runs"]
        fig1, ax1 = plt.subplots(figsize=(4.6, 3.5))
        keep = r.get("keep", np.ones(len(g), bool))
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
                    label=f"N$_{{opt}}$ = {human_format(r['N_opt'])}")
        ax1.set_xscale("log")
        notable_param_ticks(ax1, x_lo, x_hi)
        ax1.set_xlabel("Parameters")
        ax1.set_ylabel("Training loss")
        a = r["coeffs"][0]
        ax1.set_title(f"C = {budget_label(r['budget'])}   (a={a:.3f}, {len(g)} runs, "
                      f"$\\sigma$={r['resid_std']:.3f}, span {np.exp(r['log_range']):.0f}$\\times$)")
        ax1.grid(alpha=0.3)
        ax1.legend(loc="best", fontsize=8)
        fig1.tight_layout()
        fig1.savefig(f"{per_dir}/{r['budget']:.0e}.png", dpi=150)
        plt.close(fig1)


def plot_frontier(results: list[dict], prefix: str = "approach_2") -> dict[str, tuple[float, float]]:
    # concave, or a vertex outside the sampled range -> no usable minimum
    bad = [r for r in results if not np.isfinite(r["N_opt"])
           or not (r["runs"]["params"].min() <= r["N_opt"] <= r["runs"]["params"].max())]
    for r in bad:
        why = ("no usable vertex" if not np.isfinite(r["N_opt"])
               else "vertex outside the sampled range")
        print(f"  dropped C={r['budget']:.0e} from the frontier: {why} "
              f"(a={r['coeffs'][0]:+.4f})")
    results = [r for r in results if r not in bad]
    budgets = np.array([r["budget"] for r in results])
    fits = {}
    slope_n, icept_n = np.polyfit(np.log(budgets),
                                  np.log([r["N_opt"] for r in results]), 1)
    c_star = (N_TARGET / np.exp(icept_n)) ** (1 / slope_n)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, sym in [(axes[0], "N_opt", "N"), (axes[1], "D_opt", "D")]:
        y = np.array([r[key] for r in results])
        slope, intercept = np.polyfit(np.log(budgets), np.log(y), 1)
        fits[sym] = (slope, intercept)
        y_star = np.exp(intercept) * c_star**slope

        fit_x = np.array([budgets.min() / 3, c_star * 30])
        ax.plot(fit_x, np.exp(intercept) * fit_x**slope, color=RED, lw=0.9, ls="--",
                label=f"{sym}$_{{opt}}$ = {np.exp(intercept):.2e}$\\cdot C^{{{slope:.3f}}}$")
        ax.scatter(budgets, y, s=28, color="black", zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*fit_x)
        ax.set_ylim(y.min() / 3, y_star * 30)
        if key == "N_opt":
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(human_format))
            ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        annotate_optimum(ax, c_star, y_star, human_format(y_star))
        ax.set_xlabel("FLOPs")
        ax.set_ylabel({"N_opt": "Parameters", "D_opt": "Tokens"}[key])
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        print(f"{sym}_opt ∝ C^{slope:.4f}  (slope={slope:.4f}, intercept={intercept:.4f})")

    fig.tight_layout()
    out = f"plots/{prefix}_frontier.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    return fits


# Per-budget model-size window, keyed by training FLOPs (--filter-range)
BUDGET_PARAM_RANGE = {
    1.8e13: (19_796, 151_612),
    3.0e13: (34_572, 205_540),
    9.0e13: (34_572, 533_708),
    1.8e14: (51_452, 590_092),
    3.0e14: (85_212, 727_324),
    9.0e14: (114_012, 1_004_132),
    1.8e15: (506_052, 2_026_812),
    3.0e15: (666_732, 3_172_172),
}


def _in_range(budget: float, params: float) -> bool:
    for b, (lo, hi) in BUDGET_PARAM_RANGE.items():
        if abs(budget - b) / b < 0.02:
            return lo <= params <= hi
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--prefix", default="approach_2")
    parser.add_argument("--filter-range", action="store_true",
                        help="apply the hand-tuned BUDGET_PARAM_RANGE window")
    parser.add_argument("--exclude", default="", help="comma-separated budgets to drop, "
                        "e.g. 1.8e15,3e15 (matched within 1%)")
    parser.add_argument("--fit", default="trim", choices=["pooled", "independent", "trim"],
                        help="pooled = one shared curvature; independent = per-budget (paper); "
                             "trim = per-budget with one-sided outlier rejection")
    args = parser.parse_args()

    rows = []
    for config_name in os.listdir(args.run_dir):
        if not os.path.isdir(f"{args.run_dir}/{config_name}"):
            continue

        data, metadata = parse_log_and_metadata(f"{args.run_dir}/{config_name}")
        loss = gaussian_filter1d(data["loss"], sigma=20, mode="nearest")
        rows.append(
            {
                "params": float(metadata["model"]["params"]),
                "budget": float(metadata["total_flops"]) * FLOPS_BWD_MULTIPLIER,
                "tokens": float(data["tokens"][-1]),
                "loss": float(loss[-1]),
            }
        )
    runs = pd.DataFrame(rows)
    for b in (float(x) for x in args.exclude.split(",") if x.strip()):
        drop = np.isclose(runs["budget"], b, rtol=0.01)
        print(f"excluding budget C={b:.1e} ({drop.sum()} runs)")
        runs = runs[~drop]
    if args.filter_range:
        n_before = len(runs)
        runs = runs[runs.apply(lambda r: _in_range(r["budget"], r["params"]), axis=1)]
        print(f"runs used: {len(runs)}/{n_before} after per-budget param-range filter")

    mode = {"pooled": True, "independent": False, "trim": "trim"}[args.fit]
    results = fit_isoflops(runs, pooled=mode)

    print(f"\n{'C':>9} {'runs':>5} {'span':>6} {'sigma':>6} {'a':>7} {'N_opt':>9} {'D_opt':>10}")
    for r in results:
        print(f"{r['budget']:9.0e} {len(r['runs']):5d} {np.exp(r['log_range']):5.0f}× "
              f"{r['resid_std']:6.3f} {r['coeffs'][0]:7.3f} {_human_format(r['N_opt']):>9} "
              f"{r['D_opt']:10.2e}")
    print()

    plot_isoflops(results, args.prefix)
    plot_frontier(results, args.prefix)


if __name__ == "__main__":
    main()
