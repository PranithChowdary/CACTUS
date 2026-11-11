"""
make_figures.py

Regenerates every data figure in the paper from results/ artifacts.
Run from the repository root:

    python paper/make_figures.py

Writes vector PDFs into paper/figures/, which main.tex pulls in with
\\includegraphics. Nothing here hardcodes a number: every value is read from
results/final_comparison.json, results/significance.json, results/raw/*.json
and datasets/test.json, so the figures cannot drift from the reported results.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "paper", "figures")

# Colourblind-safe (Okabe-Ito). Colour never carries meaning alone -- every
# series is also separated by position and labelled directly.
C_CACTUS = "#0072B2"   # blue
C_LLM = "#D55E00"      # vermillion
C_BASE = "#999999"     # grey
C_TRUTH = "#009E73"    # bluish green

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # embed TrueType so the PDF is not bitmapped
})


def load():
    with open(os.path.join(REPO, "results/final_comparison.json")) as f:
        final = json.load(f)
    with open(os.path.join(REPO, "results/significance.json")) as f:
        sig = json.load(f)
    with open(os.path.join(REPO, "results/raw/PPO.json")) as f:
        ppo = [sorted(run, key=lambda r: r["index"]) for run in json.load(f)]
    with open(os.path.join(REPO, "results/raw/LLaMA-3.1-8B-Instruct.json")) as f:
        llama = sorted(json.load(f), key=lambda r: r["index"])
    with open(os.path.join(REPO, "datasets/test.json")) as f:
        test = json.load(f)
    return final, sig, ppo, llama, test


def fig_main_results(final, sig):
    """Grouped bars for SEM and F1. The story is the contrast between panels:
    CACTUS dominates on exact match but only ties the LLM on F1."""
    agents = ["Random", "Heuristic", "DenseRetrieval", "LLaMA-3.1-8B-Instruct", "PPO"]
    labels = ["Random", "Heuristic", "Dense\nRetrieval", "LLaMA-3.1\n-8B", "CACTUS\n(ours)"]
    colors = [C_BASE, C_BASE, C_BASE, C_LLM, C_CACTUS]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    for ax, metric, title in zip(
        axes, ["success", "f1"],
        ["Sequence Exact Match (SEM)", "Token-level F1"],
    ):
        means = [final[a][f"{metric}_mean"] for a in agents]
        errs = [final[a].get(f"{metric}_std", 0.0) for a in agents]
        x = np.arange(len(agents))
        bars = ax.bar(x, means, yerr=errs, capsize=3, color=colors,
                      edgecolor="black", linewidth=0.6, error_kw={"lw": 1.0})
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)  # title carries the metric name; no duplicate ylabel
        for b, m, e in zip(bars, means, errs):
            ax.text(b.get_x() + b.get_width() / 2, m + e + 0.03, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=7.5)
    axes[0].set_ylabel("Score")

    # Annotate the significance contrast that is the paper's point.
    p_sem = sig["comparisons"]["LLaMA-3.1-8B-Instruct|success"]["wilcoxon_p_value"]
    p_f1 = sig["comparisons"]["LLaMA-3.1-8B-Instruct|f1"]["wilcoxon_p_value"]
    axes[0].annotate("", xy=(3, 0.80), xytext=(4, 0.80),
                     arrowprops=dict(arrowstyle="<->", lw=1.0, color="black"))
    axes[0].text(3.5, 0.82, f"$p$={p_sem:.0e}", ha="center", fontsize=7.5)
    axes[1].annotate("", xy=(3, 0.93), xytext=(4, 0.93),
                     arrowprops=dict(arrowstyle="<->", lw=1.0, color="black"))
    axes[1].text(3.5, 0.95, f"n.s. ($p$={p_f1:.2f})", ha="center", fontsize=7.5)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "main_results.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def fig_mechanism(ppo, llama, test):
    """(a) why the metrics disagree: the LLM emits supersets.
       (b) why single-seed reporting misleads: the spread across seeds."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # --- (a) predicted set-size distribution ---
    truth_sizes = [len(t["Tool"].split(",")) for t in test]
    llama_sizes = [len(r["predicted"]) for r in llama]
    # steps includes the terminating DONE action, so tools selected = steps - 1
    cactus_sizes = [r["steps"] - 1 for run in ppo for r in run]

    ax = axes[0]
    bins = np.arange(-0.5, 7.5, 1)
    for data, color, label, ls in [
        (truth_sizes, C_TRUTH, "Ground truth", "-"),
        (cactus_sizes, C_CACTUS, "CACTUS", "--"),
        (llama_sizes, C_LLM, "LLaMA-3.1-8B", ":"),
    ]:
        counts, _ = np.histogram(data, bins=bins)
        ax.plot(np.arange(0, 7), counts / len(data), marker="o", ms=4,
                color=color, label=label, linestyle=ls, lw=1.6)
    ax.set_xlabel("Number of tools predicted")
    ax.set_ylabel("Fraction of queries")
    ax.set_title("(a) Predicted set size")
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=True, loc="upper right", fontsize=7.5)
    ax.text(0.03, 0.80,
            f"mean size\ntruth {np.mean(truth_sizes):.2f}\n"
            f"CACTUS {np.mean(cactus_sizes):.2f}\nLLM {np.mean(llama_sizes):.2f}",
            transform=ax.transAxes, fontsize=7, va="top", linespacing=1.4)

    # --- (b) per-seed spread ---
    ax = axes[1]
    per_seed = [np.mean([r["success"] for r in run]) for run in ppo]
    seeds = [42, 43, 44, 45, 46]
    mean, std = np.mean(per_seed), np.std(per_seed, ddof=1)
    ax.axhspan(mean - std, mean + std, color=C_CACTUS, alpha=0.15,
               label=f"mean $\\pm$ s.d. ({mean:.3f} $\\pm$ {std:.3f})")
    ax.axhline(mean, color=C_CACTUS, lw=1.4)
    ax.scatter(seeds, per_seed, color=C_CACTUS, zorder=5, s=36,
               edgecolor="black", linewidth=0.6, label="individual seed")
    # Call out the best seed only: it is the number a single-seed run would
    # have reported, and the gap to the mean is the point of this panel.
    best = int(np.argmax(per_seed))
    ax.annotate(f"best seed: {per_seed[best]:.3f}\n(a single-seed run\nwould report this)",
                xy=(seeds[best], per_seed[best]),
                xytext=(seeds[best] + 0.35, per_seed[best] + 0.015),
                fontsize=7, linespacing=1.3,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlim(41.5, 46.5)
    ax.set_xticks(seeds)
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Sequence Exact Match")
    ax.set_title("(b) Variation across seeds")
    ax.set_ylim(0.40, 0.75)
    ax.legend(frameon=True, loc="lower right", fontsize=7)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "mechanism.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    final, sig, ppo, llama, test = load()
    fig_main_results(final, sig)
    fig_mechanism(ppo, llama, test)


if __name__ == "__main__":
    main()
