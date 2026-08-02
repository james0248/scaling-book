import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.special import huber, logsumexp

from tiny.analyze_1 import FLOPS_BWD_MULTIPLIER, parse_log_and_metadata
from tiny.plot_style import (FRONTIER_BLUE, ISO_CMAP, budget_label, human_format,
                             notable_param_ticks)


def _human_format(x, pos=None):
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            v = x / div
            return f"{v:.0f}{suf}" if v == int(v) else f"{v:.1f}{suf}"
    return f"{x:.0f}"


def load_runs(run_dirs: list[str], params_key: str = "params"):
    """One (N, D, loss, C) point per run: the smoothed final loss"""
    N, D, L, C = [], [], [], []
    for run_dir in run_dirs:
        for name in os.listdir(run_dir):
            path = f"{run_dir}/{name}"
            if not os.path.isdir(path):
                continue
            if not os.path.exists(f"{path}/logs.jsonl"):
                continue
            data, metadata = parse_log_and_metadata(path)
            loss = gaussian_filter1d(data["loss"], sigma=20, mode="nearest")
            N.append(metadata["model"][params_key])
            D.append(float(data["tokens"][-1]))
            L.append(float(loss[-1]))
            C.append(float(data["tokens"][-1]) * metadata["model"]["flops_per_token"]
                     * FLOPS_BWD_MULTIPLIER)
    return np.array(N), np.array(D), np.array(L), np.array(C)


def predict_log_loss(theta, log_n, log_d):
    """log of E + A/N^alpha + B/D^beta, via log-sum-exp for stability"""
    a, b, e, alpha, beta = theta
    terms = np.stack([np.full_like(log_n, e), a - alpha * log_n, b - beta * log_d])
    return logsumexp(terms, axis=0)


def fit_parametric(N, D, L, delta: float = 1e-3):
    log_n, log_d, log_l = np.log(N), np.log(D), np.log(L)

    def objective(theta):
        resid = predict_log_loss(theta, log_n, log_d) - log_l
        return np.sum(huber(delta, resid))

    # theta = (log A, log B, log E, alpha, beta)
    best = None
    for a0 in (0.0, 10.0, 20.0):
        for b0 in (0.0, 10.0, 20.0):
            for e0 in (-1.0, 0.0):
                for alpha0 in (0.3, 0.6):
                    for beta0 in (0.3, 0.6):
                        res = minimize(objective, [a0, b0, e0, alpha0, beta0], method="L-BFGS-B")
                        if best is None or res.fun < best.fun:
                            best = res

    a, b, e, alpha, beta = best.x
    return {
        "A": np.exp(a),
        "B": np.exp(b),
        "E": np.exp(e),
        "alpha": alpha,
        "beta": beta,
        "theta": best.x,
    }


def frontier(fit):
    """N_opt(C) = G*(C/k)^a in closed form from the fitted exponents"""
    alpha, beta = fit["alpha"], fit["beta"]
    a = beta / (alpha + beta)
    G = (alpha * fit["A"] / (beta * fit["B"])) ** (1 / (alpha + beta))
    return G, a


def plot_isoloss(N, D, L, C, fit, budgets, prefix):
    theta = fit["theta"]
    fpt = C / D
    # C = r*N*D with r ~ 6 (training FLOPs per token per param), so N_opt = G*(C/r)^a
    r_ref = float(np.median(fpt / N))

    gn = np.linspace(np.log(N.min()), np.log(N.max()), 200)
    gc = np.linspace(np.log(C.min()), np.log(C.max()), 200)
    GC, GN = np.meshgrid(gc, gn)
    # fpt is near-proportional to N here, so log D = log C - log fpt(N)
    slope, icept = np.polyfit(np.log(N), np.log(fpt), 1)
    GD = GC - (slope * GN + icept)
    Lgrid = np.exp(predict_log_loss(theta, GN.ravel(), GD.ravel())).reshape(GN.shape)

    levels = np.geomspace(Lgrid.min(), Lgrid.max(), 25)
    loss_norm = mpl.colors.LogNorm(vmin=levels[0], vmax=levels[-1])
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    cs = ax.contour(np.exp(GC), np.exp(GN), Lgrid, levels=levels, cmap="magma",
                    norm=loss_norm, linewidths=0.7)
    ax.scatter(C, N, c=L, s=9, cmap="magma", norm=loss_norm,
               edgecolor="black", linewidth=0.25, zorder=3, label="Empirical data")

    G, a = frontier(fit)
    cc = np.exp(gc)
    ax.plot(cc, G * (cc / r_ref) ** a, color=FRONTIER_BLUE, lw=1.4,
            label="Efficient frontier")
    ax.set_ylim(N.min() / 1.5, N.max() * 1.5)
    for i, budget in enumerate(budgets):
        ax.axvline(budget, color=ISO_CMAP(i / max(len(budgets) - 1, 1)),
                   ls="--", lw=0.7, alpha=0.9,
                   label="IsoFLOPs slice" if i == 0 else None)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(human_format))
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("Training FLOPs")
    ax.set_ylabel("Model size")
    ax.set_title("IsoLoss contours")
    cbar = fig.colorbar(cs, ax=ax, label="Loss")
    cbar.set_ticks([t for t in (1.2, 1.5, 2, 3, 5, 8, 12)
                    if levels[0] <= t <= levels[-1]])
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, p: f"{v:g}"))
    cbar.ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    cbar.ax.tick_params(width=0.6)
    cbar.outline.set_linewidth(0.6)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = f"plots/{prefix}_isoloss_contours.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_isoflops(N, D, L, C, fit, budgets, prefix):
    theta = fit["theta"]
    slope, icept = np.polyfit(np.log(N), np.log(C / D), 1)
    norm = mpl.colors.LogNorm(vmin=budgets.min(), vmax=budgets.max())
    cmap = ISO_CMAP

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(N, L, c=C, s=14, cmap=cmap, norm=norm,
               edgecolor="black", linewidth=0.25, zorder=3)

    gn = np.linspace(np.log(N.min()), np.log(N.max()), 100)
    for budget in budgets:
        gd = np.log(budget) - (slope * gn + icept)
        loss = np.exp(predict_log_loss(theta, gn, gd))
        ax.plot(np.exp(gn), loss, color=cmap(norm(budget)), ls="--", lw=1.0,
                label=budget_label(budget))

    ax.set_xscale("log")
    notable_param_ticks(ax, N.min() / 1.4, N.max() * 1.4)
    ax.set_ylim(L.min() * 0.95, L.max() * 1.4)  # slice curves blow up off-frontier
    ax.set_xlabel("Model size")
    ax.set_ylabel("Loss")
    ax.set_title("IsoFLOPs slices")
    ax.grid(alpha=0.3)
    ax.legend(title="Train. FLOPs", fontsize=8, title_fontsize=8,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    out = f"plots/{prefix}_isoflop_slices.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="Approach 1 & 2 multirun directories")
    parser.add_argument("--prefix", default="approach_3")
    parser.add_argument("--params_key", default="params")
    parser.add_argument("--max_flops", type=float, default=None)
    args = parser.parse_args()

    N, D, L, C = load_runs(args.run_dirs, args.params_key)
    print(f"{len(N)} runs from {', '.join(args.run_dirs)}")
    if args.max_flops is not None:
        keep = C <= args.max_flops
        print(f"  keeping {keep.sum()}/{len(C)} runs with C <= {args.max_flops:.1e}")
        N, D, L, C = N[keep], D[keep], L[keep], C[keep]
    fit = fit_parametric(N, D, L)
    print(
        f"E={fit['E']:.4f}  A={fit['A']:.3e}  B={fit['B']:.3e}  "
        f"alpha={fit['alpha']:.4f}  beta={fit['beta']:.4f}"
    )

    _, a = frontier(fit)
    print(f"N_opt ∝ C^{a:.4f}   D_opt ∝ C^{1 - a:.4f}")

    budgets = np.logspace(np.log10(C.min()), np.log10(C.max()), 8)
    plot_isoloss(N, D, L, C, fit, budgets, args.prefix)
    plot_isoflops(N, D, L, C, fit, budgets, args.prefix)


if __name__ == "__main__":
    main()
