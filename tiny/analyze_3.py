import argparse
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.special import huber, logsumexp


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


def load_runs(run_dirs: list[str]):
    # Collect every logged (N, D, loss) point across all Approach 1 & 2 runs.
    N, D, L = [], [], []
    for run_dir in run_dirs:
        for name in os.listdir(run_dir):
            path = f"{run_dir}/{name}"
            if not os.path.isdir(path):
                continue
            data, metadata = parse_log_and_metadata(path)
            loss = gaussian_filter1d(data["loss"], sigma=5, radius=10, mode="nearest")
            N.append(np.full(len(loss), metadata["model"]["params"]))
            D.append(data["tokens"])
            L.append(loss)
    return np.concatenate(N), np.concatenate(D), np.concatenate(L)


def predict_log_loss(theta, log_n, log_d):
    # log L^(N, D) with L^ = E + A/N^a + B/D^b, computed via log-sum-exp for stability.
    a, b, e, alpha, beta = theta
    terms = np.stack([np.full_like(log_n, e), a - alpha * log_n, b - beta * log_d])
    return logsumexp(terms, axis=0)


def fit_parametric(N, D, L, delta: float = 1e-3):
    log_n, log_d, log_l = np.log(N), np.log(D), np.log(L)

    def objective(theta):
        resid = predict_log_loss(theta, log_n, log_d) - log_l
        return np.sum(huber(delta, resid))

    # theta = (log A, log B, log E, alpha, beta); best fit over a grid of inits.
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
    # Compute-optimal N_opt(C) = G*(C/6)^a from the fitted exponents (Eq. 4).
    alpha, beta = fit["alpha"], fit["beta"]
    a = beta / (alpha + beta)
    G = (alpha * fit["A"] / (beta * fit["B"])) ** (1 / (alpha + beta))
    return G, a


def plot_isoloss(N, D, L, fit, budgets):
    C = 6 * N * D
    theta = fit["theta"]

    # Loss surface over a (FLOPs, model size) grid, with D implied by C = 6ND.
    gn = np.linspace(np.log(N.min()), np.log(N.max()), 200)
    gc = np.linspace(np.log(C.min()), np.log(C.max()), 200)
    GC, GN = np.meshgrid(gc, gn)
    GD = GC - np.log(6) - GN
    Lgrid = np.exp(predict_log_loss(theta, GN.ravel(), GD.ravel())).reshape(GN.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contour(np.exp(GC), np.exp(GN), Lgrid, levels=25, cmap="magma_r", linewidths=0.8)
    ax.scatter(C, N, c=L, s=4, cmap="viridis", norm=mpl.colors.LogNorm(), zorder=3)

    G, a = frontier(fit)
    cc = np.exp(gc)
    ax.plot(cc, G * (cc / 6) ** a, color="tab:blue", lw=2, label="efficient frontier")
    for budget in budgets:
        ax.axvline(budget, color="gray", ls="--", lw=0.6, alpha=0.6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_human_format))
    ax.set_xlabel("training FLOPs")
    ax.set_ylabel("model size")
    ax.set_title("Approach 3: IsoLoss contours")
    fig.colorbar(cs, ax=ax, label="loss")
    ax.legend()
    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fig.savefig("plots/approach_3_isoloss.png", dpi=150)


def plot_isoflops(N, D, L, fit, budgets):
    C = 6 * N * D
    theta = fit["theta"]
    norm = mpl.colors.LogNorm(vmin=budgets.min(), vmax=budgets.max())
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(N, L, c=C, s=6, cmap=cmap, norm=norm, zorder=3)

    # Fitted loss as a function of model size at each fixed FLOP budget.
    gn = np.linspace(np.log(N.min()), np.log(N.max()), 100)
    for budget in budgets:
        gd = np.log(budget) - np.log(6) - gn
        loss = np.exp(predict_log_loss(theta, gn, gd))
        ax.plot(np.exp(gn), loss, color=cmap(norm(budget)), ls="--", lw=1.2)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_human_format))
    ax.set_xlabel("model size")
    ax.set_ylabel("loss")
    ax.set_title("Approach 3: IsoFLOP slices")
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label="training FLOPs")
    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fig.savefig("plots/approach_3_isoflops.png", dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="Approach 1 & 2 multirun directories")
    args = parser.parse_args()

    N, D, L = load_runs(args.run_dirs)
    fit = fit_parametric(N, D, L)
    print(
        f"E={fit['E']:.4f}  A={fit['A']:.3e}  B={fit['B']:.3e}  "
        f"alpha={fit['alpha']:.4f}  beta={fit['beta']:.4f}"
    )

    _, a = frontier(fit)
    print(f"N_opt ∝ C^{a:.4f}   D_opt ∝ C^{1 - a:.4f}")

    C = 6 * N * D
    budgets = np.logspace(np.log10(C.min()), np.log10(C.max()), 8)
    plot_isoloss(N, D, L, fit, budgets)
    plot_isoflops(N, D, L, fit, budgets)


if __name__ == "__main__":
    main()
