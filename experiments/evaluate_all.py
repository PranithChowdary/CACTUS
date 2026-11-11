"""
evaluate_all.py

Evaluates every agent on the FULL test set, deterministically and exactly
once per example (env.reset(options={"index": i}) for i in range(len(test))),
so every agent is scored on identical episodes -- the previous version
sampled test episodes randomly with replacement, which meant agents were
compared on different, non-matched samples of the test set.

PPO is trained with multiple seeds (env.config.SEEDS); we report mean +/- std
across seeds, not a single run. Random is inherently stochastic so it is also
repeated across seeds. Heuristic and (post-calibration) DenseRetrieval are
deterministic given a fixed threshold, so one pass suffices for them.

Outputs:
  results/final_comparison.json  -- mean +/- std per agent (the headline table)
  results/raw/<agent>.json       -- per-example, per-seed raw records
  results/significance.json      -- paired bootstrap CI + Wilcoxon test,
                                     CACTUS vs. the strongest baseline
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import glob
import random
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from env.metatool_env import MetaToolEnv
from env.wrappers import TextEmbeddingWrapper
from env import config
from models.ppo_agent import PPOAgent
from models.baseline_agents import RandomAgent, HeuristicAgent, DenseRetrievalAgent

N_RANDOM_REPEATS = 5  # RandomAgent is stochastic; repeat it like a seed sweep.


def run_deterministic_pass(select_action_fn, env, test_df, is_ppo=False):
    """One full, ordered pass over every example in test_df. Returns a list
    of per-example dicts (index-aligned across agents so comparisons are paired)."""
    records = []
    for i in range(len(test_df)):
        obs, info = env.reset(options={"index": i})
        done = False
        total_reward = 0.0
        while not done:
            if is_ppo:
                action = select_action_fn(obs)
            else:
                selected = np.where(obs["selection"] == 1)[0].tolist()
                action = select_action_fn(info["prompt"], exclude_indices=selected)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        tp = len(set(info["selected_tools"]) & set(info["correct_tools"]))
        precision = tp / max(1, len(info["selected_tools"]))
        recall = tp / max(1, len(info["correct_tools"]))
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        records.append({
            "index": i,
            "reward": total_reward,
            "f1": f1,
            "success": 1.0 if set(info["selected_tools"]) == set(info["correct_tools"]) else 0.0,
            "steps": info["step_count"],
        })
    return records


def aggregate(records_per_run):
    """records_per_run: list of runs, each a list of per-example dicts (same
    order/length). Returns (summary mean+-std across runs, per-example mean
    across runs for the paired significance test)."""
    per_run_means = {k: [] for k in ["reward", "f1", "success", "steps"]}
    for run in records_per_run:
        df = pd.DataFrame(run)
        for k in per_run_means:
            per_run_means[k].append(df[k].mean())

    summary = {}
    for k, vals in per_run_means.items():
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    n_examples = len(records_per_run[0])
    per_example_mean = {"f1": np.zeros(n_examples), "success": np.zeros(n_examples)}
    for run in records_per_run:
        df = pd.DataFrame(run).sort_values("index").reset_index(drop=True)
        per_example_mean["f1"] += df["f1"].to_numpy()
        per_example_mean["success"] += df["success"].to_numpy()
    for k in per_example_mean:
        per_example_mean[k] /= len(records_per_run)

    return summary, per_example_mean


def paired_bootstrap_ci(a, b, n_boot=10000, seed=0):
    """95% CI for mean(a) - mean(b) via paired bootstrap over examples."""
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = a - b
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def main():
    with open(config.TEST_DATA, 'r') as f:
        test_df = pd.DataFrame(json.load(f))
    with open(config.VAL_DATA, 'r') as f:
        val_df = pd.DataFrame(json.load(f))
    with open(config.VOCAB_PATH, 'r') as f:
        vocab = json.load(f)

    base_env = MetaToolEnv(test_df, vocab, max_steps=config.MAX_STEPS, mode="multi-step", seed=config.SEED)
    embedding_model = config.resolve_embedding_model()
    wrapped_env = TextEmbeddingWrapper(base_env, model_name=embedding_model)

    os.makedirs("results/raw", exist_ok=True)
    all_summaries = {}

    # --- Random: stochastic, repeat like a seed sweep ---
    random_runs = []
    for rep in range(N_RANDOM_REPEATS):
        random.seed(1000 + rep)
        agent = RandomAgent(vocab)
        random_runs.append(run_deterministic_pass(agent.select_action, base_env, test_df))
    all_summaries["Random"], random_per_ex = aggregate(random_runs)
    with open("results/raw/Random.json", "w") as f:
        json.dump(random_runs, f, indent=2)

    # --- Heuristic: deterministic given fixed rules ---
    heuristic_agent = HeuristicAgent(vocab)
    heuristic_run = run_deterministic_pass(heuristic_agent.select_action, base_env, test_df)
    all_summaries["Heuristic"], _ = aggregate([heuristic_run])
    with open("results/raw/Heuristic.json", "w") as f:
        json.dump([heuristic_run], f, indent=2)

    # --- Dense Retrieval: calibrate its stop threshold on VAL, then eval on TEST ---
    dense_agent = DenseRetrievalAgent(vocab, model_name=embedding_model)

    def correct_tools_for(prompt):
        row = val_df[val_df["Query"] == prompt].iloc[0]
        return set(row["Tool"].split(","))

    dense_agent.calibrate_threshold(val_df["Query"].tolist(), correct_tools_for)
    dense_run = run_deterministic_pass(dense_agent.select_action, base_env, test_df)
    all_summaries["DenseRetrieval"], dense_per_ex = aggregate([dense_run])
    with open("results/raw/DenseRetrieval.json", "w") as f:
        json.dump([dense_run], f, indent=2)

    # --- PPO (CACTUS): one run per trained seed ---
    ckpts = sorted(glob.glob(os.path.join(config.CHECKPOINT_DIR, "ppo_metatool_seed*.zip")))
    if not ckpts:
        legacy = os.path.join(config.CHECKPOINT_DIR, "ppo_metatool_final.zip")
        if os.path.exists(legacy):
            print("Warning: no per-seed checkpoints found; falling back to legacy single checkpoint "
                  "(this was trained with the old, unmasked PPOAgent and is NOT comparable to the "
                  "masked results below -- retrain with experiments/train_ppo.py --all_seeds).")
            ckpts = [legacy]
        else:
            print("Warning: no PPO checkpoints found. Skipping PPO evaluation.")
            ckpts = []

    ppo_runs = []
    if ckpts:
        agent = PPOAgent(wrapped_env, seed=config.SEED)
        for ckpt in ckpts:
            agent.load(ckpt.replace(".zip", ""))
            run = run_deterministic_pass(agent.select_action, wrapped_env, test_df, is_ppo=True)
            ppo_runs.append(run)
            print(f"  [{os.path.basename(ckpt)}] mean F1={np.mean([r['f1'] for r in run]):.3f} "
                  f"success={np.mean([r['success'] for r in run]):.3f}")
        all_summaries["PPO"], ppo_per_ex = aggregate(ppo_runs)
        with open("results/raw/PPO.json", "w") as f:
            json.dump(ppo_runs, f, indent=2)

    for name, metrics in all_summaries.items():
        print(f"{name}: {metrics}")

    with open("results/final_comparison.json", "w") as f:
        json.dump(all_summaries, f, indent=4)

    # --- Paired significance: CACTUS vs. EVERY baseline, on both metrics ---
    # Both metrics matter and they can disagree: an LLM that over-predicts wins
    # partial credit on F1 while rarely matching the target set exactly, so a
    # significant SEM gap can sit next to a non-significant F1 gap. Reporting
    # only the favourable one would be cherry-picking.
    if ppo_runs:
        baseline_per_ex = {"Random": random_per_ex, "DenseRetrieval": dense_per_ex}

        # LLaMA is produced by a separate job (experiments/evaluate_llm_baseline.py);
        # fold it in when its per-example records are present. It is usually the
        # strongest baseline on F1, so omitting it understates the comparison.
        llama_raw_path = "results/raw/LLaMA-3.1-8B-Instruct.json"
        if os.path.exists(llama_raw_path):
            with open(llama_raw_path) as f:
                llama_records = sorted(json.load(f), key=lambda r: r["index"])
            if len(llama_records) == len(test_df):
                baseline_per_ex["LLaMA-3.1-8B-Instruct"] = {
                    "f1": np.array([r["f1"] for r in llama_records]),
                    "success": np.array([r["success"] for r in llama_records]),
                }
            else:
                print(f"Warning: {llama_raw_path} has {len(llama_records)} records but the test set "
                      f"has {len(test_df)} -- skipping it in the significance test.")
        else:
            print(f"Note: {llama_raw_path} not found -- run experiments/evaluate_llm_baseline.py "
                  f"to include the LLM baseline in the significance comparison.")

        significance = {
            "n_examples": len(ppo_per_ex["f1"]),
            "n_seeds": len(ppo_runs),
            "note": "CACTUS per-example scores are averaged across seeds before pairing. "
                    "Baselines are single deterministic runs (Random is averaged across its repeats).",
            "comparisons": {},
        }

        for b_name, b_scores in baseline_per_ex.items():
            for metric in ["f1", "success"]:
                a, b = ppo_per_ex[metric], b_scores[metric]
                diff_mean, ci_lo, ci_hi = paired_bootstrap_ci(a, b)
                try:
                    stat, p_value = wilcoxon(a, b)
                except ValueError:
                    stat, p_value = float("nan"), float("nan")
                significance["comparisons"][f"{b_name}|{metric}"] = {
                    "cactus_mean": float(a.mean()),
                    "baseline_mean": float(b.mean()),
                    "mean_diff": diff_mean,
                    "bootstrap_95ci": [ci_lo, ci_hi],
                    "wilcoxon_statistic": float(stat),
                    "wilcoxon_p_value": float(p_value),
                    "significant_at_0.05": bool(p_value < 0.05),
                }
                print(f"CACTUS vs {b_name:24s} [{metric:7s}] "
                      f"{a.mean():.3f} vs {b.mean():.3f}  diff={diff_mean:+.3f}  "
                      f"95%CI=[{ci_lo:+.3f},{ci_hi:+.3f}]  p={p_value:.2e}"
                      f"{'' if p_value < 0.05 else '   (n.s.)'}")

        with open("results/significance.json", "w") as f:
            json.dump(significance, f, indent=2)

    print("\nEvaluation complete. Results saved to results/final_comparison.json, "
          "results/raw/, results/significance.json")


if __name__ == "__main__":
    main()
