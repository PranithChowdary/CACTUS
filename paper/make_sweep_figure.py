"""
make_sweep_figure.py

Plots the alpha sweep from results/alpha_sweep.json: exact-match against the
fraction of execution-dependent queries, for the memoryless classifier, the
sequential policy, and the blind-policy control, with the analytic ceiling for
any query-only model drawn as a reference line.

    python paper/make_sweep_figure.py [--results results/alpha_sweep.json]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

C_SUP = "#D55E00"    # vermillion  - memoryless classifier
C_PPO = "#0072B2"    # blue        - sequential policy with result channel
C_BLIND = "#999999"  # grey        - sequential policy, channel ablated
C_CEIL = "#009E73"   # green       - analytic bound

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "figure.dpi": 150, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(REPO, "results/alpha_sweep.json"))
    ap.add_argument("--out", default=os.path.join(REPO, "paper/figures/alpha_sweep.pdf"))
    args = ap.parse_args()

    with open(args.results) as f:
        res = json.load(f)

    alphas = sorted(float(a) for a in res)
    def series(key):
        m = np.array([res[str(a)][f"{key}_mean"] for a in alphas])
        s = np.array([res[str(a)][f"{key}_std"] for a in alphas])
        return m, s
    ceiling = np.array([res[str(a)]["static_ceiling"] for a in alphas])

    fig, ax = plt.subplots(figsize=(4.6, 3.2))

    ax.plot(alphas, ceiling, color=C_CEIL, ls="-", lw=1.4, marker="s", ms=4,
            label="Analytic ceiling, any query-only model")

    for key, color, ls, marker, label in [
        ("supervised", C_SUP, ":", "o", "Memoryless classifier"),
        ("ppo_blind", C_BLIND, "-.", "^", "Sequential policy (results hidden)"),
        ("ppo", C_PPO, "--", "D", "Sequential policy (observes results)"),
    ]:
        m, s = series(key)
        ax.plot(alphas, m, color=color, ls=ls, marker=marker, ms=4.5, lw=1.6, label=label)
        ax.fill_between(alphas, m - s, m + s, color=color, alpha=0.15)

    ax.set_xlabel(r"$\alpha$  (fraction of execution-dependent queries)")
    ax.set_ylabel("Sequence Exact Match")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(min(alphas) - 0.02, max(alphas) + 0.02)
    ax.legend(frameon=True, loc="lower left", fontsize=7.2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
