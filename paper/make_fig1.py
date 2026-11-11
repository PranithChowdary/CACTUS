"""
make_fig1.py -- Figure 1 for paper/termination_draft.tex

Every number is read from results/, never hardcoded, so the figure cannot drift
from the tables.

Why two panels rather than one scatter: a single scatter of
tool-competence vs termination would be the most legible form, but the RL run
does not store per-episode action sequences, so recall@5 is not computable for
the agent. Putting the agent on that axis would require substituting a
different metric (static exact-match) and silently mixing two quantities on one
axis. Two panels, each using metrics that exist for its policies, is honest.

Palette: dataviz default categorical slots 1-3 (#2a78d6 blue, #eb6834 orange,
#1baf7a aqua), validated with scripts/validate_palette.js --mode light:
all checks pass, worst adjacent CVD dE 9.2 (deutan), one contrast WARN on aqua
(2.74:1) which the direct value labels and Table 1 discharge. "Stop rate" is
orange in BOTH panels -- colour follows the quantity, not the panel.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b2"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
})

# ---------------------------------------------------------------- panel A data
LLMS = [("LLaMA-3.2-1B", "llama-3.2-1b"), ("LLaMA-3.1-8B", "llama-3.1-8b"),
        ("LLaMA-3.1-70B", "llama-3.1-70b"), ("Qwen2.5-7B", "qwen2.5-7b")]
recall, stop, pdc = [], [], []
for _, tag in LLMS:
    d = json.load(open(f"{R}/llm_probe_bank61_{tag}.json"))
    a0 = next(r for r in d if r["scoring"] == "sum" and r["alpha"] == 0.0)
    eps = a0["episodes"]
    recall.append(sum(1 for e in eps
                      if set(e["gold"]) <= set([a for a in e["actions"] if a != "DONE"][:5])) / len(eps))
    stop.append(a0["done_rate"])
    pdc.append(a0["done_prob_at_complete"])

# ---------------------------------------------------------------- panel B data
d = json.load(open(f"{R}/nl_alpha_shift.json")); RUNS = d["runs"]
keys = sorted(RUNS)
def agg(k, path):
    v = [s["static_only"][path] if path != "cond" else s["by_eval_alpha"]["1.00"]["cond_probe_rate"]
         for s in RUNS[k]["seeds"]]
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
p_cond = [agg(k, "cond") for k in keys]
p_stat = [agg(k, "static_probe_rate") for k in keys]
p_stop = [agg(k, "done_rate") for k in keys]

# ----------------------------------------------------------------------- draw
# Layout notes: model names are two-line (they collided on one line), legends
# sit ABOVE the bars in a single row (both panels have bars reaching 1.0, so an
# in-plot legend overlapped data), and the P(stop|complete) row gets its own
# margin below the tick labels.
fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.7),
                               gridspec_kw={"width_ratios": [1.05, 1]})

def style(ax):
    ax.set_ylim(0, 1.30); ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.grid(axis="y", color=MUTED, lw=0.5, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, length=2.5)

# --- A: language models -----------------------------------------------------
x = np.arange(len(LLMS)); w = 0.36
axA.bar(x - w/2 - 0.012, recall, w, color=BLUE, label="tool recall@5", zorder=3)
axA.bar(x + w/2 + 0.012, stop, w, color=ORANGE, label="stop rate", zorder=3)
for i, v in enumerate(stop):                       # label the decisive series only
    axA.text(x[i] + w/2 + 0.012, v + 0.03, f"{v:.3f}", ha="center",
             va="bottom", fontsize=7.2, color=INK)
axA.set_xticks(x)
axA.set_xticklabels(["LLaMA-3.2\n1B", "LLaMA-3.1\n8B", "LLaMA-3.1\n70B",
                     "Qwen2.5\n7B"], fontsize=7.2)
axA.set_ylabel("fraction of scenarios"); style(axA)
axA.set_title("a  Language models: relevance without termination",
              loc="left", color=INK, pad=16)
axA.legend(frameon=False, ncol=2, loc="upper left",
           bbox_to_anchor=(0.0, 1.005), handlelength=1.1, columnspacing=1.2)
for i, p_ in enumerate(pdc):                       # decisive metric, own row
    axA.annotate("—" if np.isnan(p_) else f"{p_:.4f}", xy=(x[i], 0),
                 xytext=(0, -31), textcoords="offset points",
                 ha="center", fontsize=6.9, color=INK2, annotation_clip=False)
# Row label omitted deliberately: at 6.9pt it needs ~75pt of left margin and was
# clipped by the figure edge. The caption names the quantity instead.

# --- B: masked PPO ----------------------------------------------------------
xb = np.arange(len(keys)); wb = 0.26
for off, (vals, c, lab) in zip(
        (-wb-0.014, 0.0, wb+0.014),
        [(p_cond, BLUE, "probe | conditional"),
         (p_stop, ORANGE, "stop rate"),
         (p_stat, AQUA, "probe | static (waste)")]):
    m = [v[0] for v in vals]; e = [v[1] for v in vals]
    axB.bar(xb + off, m, wb, yerr=e, color=c, label=lab, zorder=3,
            error_kw=dict(ecolor=INK2, lw=0.7, capsize=1.6))
for i, (m, e) in enumerate(p_stat):                # label the decisive series only
    # A tall bar's label cannot sit level with the neighbouring stop bar, and
    # reversed ink inside the bar does not work either: at 7.2pt the string is
    # wider than the 0.26-unit bar, so its outer characters land on the white
    # surface and vanish. Tall values are labelled above the plot area instead.
    if m > 0.5:
        axB.text(xb[i] + wb + 0.014, 1.06, f"{m:.3f}", ha="center",
                 va="bottom", fontsize=7.2, color=INK, zorder=4)
    else:
        axB.text(xb[i] + wb + 0.014, m + e + 0.035, f"{m:.3f}", ha="center",
                 va="bottom", fontsize=7.2, color=INK)
# Extra right margin: the last group's value label sits on the outermost bar
# and was clipped by the axes edge.
axB.set_xlim(-0.58, len(keys) - 1 + 0.66)
axB.set_xticks(xb)
axB.set_xticklabels([r"$\alpha_{\mathrm{train}}$" + f"\n{k}" for k in keys], fontsize=7.2)
style(axB)
axB.set_title("b  Masked PPO, same task: both capabilities",
              loc="left", color=INK, pad=16)
axB.legend(frameon=False, ncol=2, loc="upper left",
           bbox_to_anchor=(0.0, 1.035), handlelength=1.1, columnspacing=1.0)

fig.subplots_adjust(left=0.095, right=0.985, top=0.86, bottom=0.26, wspace=0.17)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "fig1_dissociation")
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=220)
print("wrote", out + ".pdf / .png")
print("A recall@5:", [round(v,3) for v in recall], " stop:", [round(v,3) for v in stop])
print("B probe|stat:", [round(v[0],3) for v in p_stat], " stop:", [round(v[0],3) for v in p_stop])
