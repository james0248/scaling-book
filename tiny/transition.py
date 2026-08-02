"""Where the smooth scaling law stops describing this task

Fits L = E + A/N^alpha + B/D^beta on low-compute runs only, then measures how far
higher-compute runs fall below it. Sets the compute cut used by approach 3.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from tiny.analyze_3 import fit_parametric, load_runs, predict_log_loss

FAMILIES = {
    "dense": ["outputs/approach_1", "outputs/approach_2"],
    "MoE": ["outputs/approach_1_moe", "outputs/approach_2_moe"],
}


def analyse(run_dirs, fit_below: float):
    N, D, L, C = load_runs(run_dirs, "params")
    train = C <= fit_below
    fit = fit_parametric(N[train], D[train], L[train])
    pred = np.exp(predict_log_loss(fit["theta"], np.log(N), np.log(D)))
    return C, L - pred, fit, train


def transition_rate(C, resid, train, edges, k: float = 3.0):
    """Fraction of runs per compute bin sitting more than k sigma below the surface

    A rate, not a first crossing: dense has ~4x more runs, so it would always find
    the earliest transitioning run purely by sampling.
    """
    sigma = float(np.std(resid[train]))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (C >= lo) & (C < hi)
        out.append(
            (lo, hi, int(m.sum()), float((resid[m] < -k * sigma).mean()) if m.sum() else np.nan)
        )
    return out, sigma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit_below", type=float, default=3.0e14)
    args = parser.parse_args()

    edges = np.logspace(12.0, 15.8, 9)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"dense": "tab:blue", "MoE": "tab:red"}
    for name, dirs in FAMILIES.items():
        if not all(os.path.isdir(d) for d in dirs):
            continue
        C, resid, fit, train = analyse(dirs, args.fit_below)
        rates, sigma = transition_rate(C, resid, train, edges)
        print(
            f"\n{name}: n={len(C)}, surface fitted on {train.sum()} runs below "
            f"{args.fit_below:.0e} (E={fit['E']:.3f} α={fit['alpha']:.3f} "
            f"β={fit['beta']:.3f}), residual σ={sigma:.4f}"
        )
        print(f"  {'compute bin':>21} {'runs':>5} {'transitioned':>13} {'best loss':>10}")
        for lo, hi, n, frac in rates:
            m = (C >= lo) & (C < hi)
            best = f"{(resid[m] + 0)[np.argmin(resid[m])]:+.3f}" if n else "—"
            print(
                f"  {lo:9.1e}–{hi:9.1e} {n:5d} "
                f"{'—' if not n else f'{100 * frac:11.0f}%'} {best:>10}"
            )
        axes[0].scatter(C, resid, s=12, alpha=0.6, color=colors[name], label=name)
        mid = np.sqrt(edges[:-1] * edges[1:])
        axes[1].plot(mid, [r[3] for r in rates], "o-", color=colors[name], label=name)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("training FLOPs")
    axes[1].set_ylabel("fraction of runs $>3\\sigma$ below the surface")
    axes[1].set_title("Transition rate vs compute")
    axes[1].legend(fontsize=9)
    ax = axes[0]
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(
        args.fit_below,
        color="gray",
        ls="--",
        lw=1,
        label=f"surface fitted below {args.fit_below:.0e}",
    )
    ax.set_xscale("log")
    ax.set_xlabel("training FLOPs")
    ax.set_ylabel("final loss $-$ smooth-surface prediction (nats)")
    ax.set_title("Departure from the smooth scaling law")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fig.savefig("plots/01_transition.png", dpi=150)
    print("\nwrote plots/01_transition.png")


if __name__ == "__main__":
    main()
