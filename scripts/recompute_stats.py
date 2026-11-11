"""
recompute_stats.py

Rebuilds results/final_comparison.json and results/significance.json from the
per-example records in results/raw/ alone -- no models, no GPU, no ML deps
beyond numpy/scipy.

Use it to re-analyse a finished run (e.g. to add a baseline, change a test, or
regenerate an artifact) without burning cluster time re-running the agents.
The numbers are computed from the same raw records evaluate_all.py wrote, so
they are identical to what a full re-run would produce.

    python scripts/recompute_stats.py
"""
import json
import os
import sys
import numpy as np
from scipy.stats import wilcoxon

RAW_DIR = "results/raw"
METRICS = ["reward", "f1", "success", "steps"]
LLAMA_NAME = "LLaMA-3.1-8B-Instruct"


def load_runs(name):
    """Returns a list of runs; each run is a list of per-example dicts."""
    path = os.path.join(RAW_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    # The LLM baseline is written as a flat list of examples, the agents as a
    # list of runs. Normalize to the list-of-runs shape.
    if data and isinstance(data[0], dict):
        data = [data]
    return [sorted(run, key=lambda r: r["index"]) for run in data]


def summarize(runs):
    per_run = {m: [] for m in METRICS if m in runs[0][0]}
    for run in runs:
        for m in per_run:
            per_run[m].append(float(np.mean([r[m] for r in run])))
    out = {}
    for m, vals in per_run.items():
        out[f"{m}_mean"] = float(np.mean(vals))
        out[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    out["n_runs"] = len(runs)
    return out


def per_example(runs, metric):
    return np.mean([[r[metric] for r in run] for run in runs], axis=0)


def paired_bootstrap_ci(a, b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = a - b
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = diffs[rng.integers(0, len(diffs), len(diffs))].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def main():
    # "Supervised" is the memoryless logistic probe (experiments/evaluate_supervised.py).
    # It is the bar the whole MDP formulation has to clear, so it is compared on
    # equal terms rather than mentioned in passing. "Supervised-topk" is reported
    # but excluded from the headline comparisons: it is handed the target
    # cardinality, which the RL agent is not.
    agents = ["Random", "Heuristic", "DenseRetrieval", "Supervised",
              "Supervised-topk", "PPO", LLAMA_NAME]
    runs = {name: load_runs(name) for name in agents}
    missing = [n for n, r in runs.items() if r is None]
    if missing:
        print(f"Note: no raw records for {missing} -- skipping those.")
    runs = {n: r for n, r in runs.items() if r is not None}

    if "PPO" not in runs:
        print("ERROR: results/raw/PPO.json missing -- nothing to compare. "
              "Run experiments/evaluate_all.py on a machine that has the "
              "checkpoints in logs/checkpoints/.")
        return 1

    summaries = {n: summarize(r) for n, r in runs.items()}
    for name, s in summaries.items():
        print(f"{name}: {s}")
    with open("results/final_comparison.json", "w") as f:
        json.dump(summaries, f, indent=4)

    n_examples = len(runs["PPO"][0])
    significance = {
        "n_examples": n_examples,
        "n_seeds": len(runs["PPO"]),
        "note": "CACTUS per-example scores are averaged across seeds before pairing. "
                "Baselines are single deterministic runs (Random is averaged across its repeats).",
        "comparisons": {},
    }

    print()
    # Heuristic is too weak to be informative; Supervised-topk is given k, so it
    # is summarized above but not used as a headline paired comparison.
    for b_name in [n for n in runs if n not in ("PPO", "Heuristic", "Supervised-topk")]:
        for metric in ["f1", "success"]:
            a = per_example(runs["PPO"], metric)
            b = per_example(runs[b_name], metric)
            if len(b) != len(a):
                print(f"Skipping {b_name}: {len(b)} examples vs {len(a)}")
                continue
            diff, lo, hi = paired_bootstrap_ci(a, b)
            try:
                stat, p = wilcoxon(a, b)
            except ValueError:
                stat, p = float("nan"), float("nan")
            significance["comparisons"][f"{b_name}|{metric}"] = {
                "cactus_mean": float(a.mean()),
                "baseline_mean": float(b.mean()),
                "mean_diff": diff,
                "bootstrap_95ci": [lo, hi],
                "wilcoxon_statistic": float(stat),
                "wilcoxon_p_value": float(p),
                "significant_at_0.05": bool(p < 0.05),
            }
            print(f"CACTUS vs {b_name:24s} [{metric:7s}] {a.mean():.3f} vs {b.mean():.3f}  "
                  f"diff={diff:+.3f}  95%CI=[{lo:+.3f},{hi:+.3f}]  p={p:.2e}"
                  f"{'' if p < 0.05 else '   (n.s.)'}")

    with open("results/significance.json", "w") as f:
        json.dump(significance, f, indent=2)
    print("\nWrote results/final_comparison.json and results/significance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
