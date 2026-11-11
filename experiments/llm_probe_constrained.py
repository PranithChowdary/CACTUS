"""
llm_probe_constrained.py

Re-runs the information-gathering probe test with a CONSTRAINED LLM policy: at
each step the model scores every legal action (remaining tools + DONE) and the
argmax is taken. The LLM now acts over the same discrete, masked action set as
the PPO agent, so the comparison is between policies rather than between
interfaces.

This replaces the free-generation harness, which produced DONE in 0 of 240
turns and was therefore incapable of measuring stopping behaviour at all.

All three scorings are reported, not just the primary one
---------------------------------------------------------
Scorer choice is not a detail here: on a toy prompt where the task had not
started, DONE had the HIGHEST raw summed log-probability of all candidates
(-12.43) purely because it is a short, frequent string, while its prior
accounted for nearly all of that mass. Naive argmax-sum would have reported a
model that always stops; PMI reports one that rarely does. Since the scorer can
flip the conclusion, all three are logged per step and the headline uses PMI
(domain-conditional, Holtzman et al. 2021).

Metrics
-------
  exact_match          selected set == gold set
  done_rate            fraction of episodes that terminate by choosing DONE
  probe_first          conditional episodes whose first action is the probe
  conditions_on_result among episodes that probed, fraction recovering all gold
  over_probe_rate      STATIC episodes where the probe was called -- the probe
                       is not in static gold, so this is pure waste. Headline.
  done_prob_at_complete probability mass on DONE at the exact step where the
                       gold set has just been completed. Measures the stop
                       signal independently of whether DONE wins the argmax.

    sbatch scripts/run_llm_probe_constrained.sh
"""
import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from env.nl_conditional_env import NLConditionalEnv, SCENARIOS
from models.llm_policy import ConstrainedLLMPolicy


def run(policy, alpha, n_episodes, scoring, seed=0):
    envs, obs, infos = [], [], []
    for i in range(n_episodes):
        e = NLConditionalEnv(alpha=alpha, seed=seed * 100_000 + i)
        o, inf = e.reset(options={"scenario": i % len(SCENARIOS)})
        envs.append(e); obs.append(o); infos.append(inf)

    first = [None] * n_episodes
    actions = [[] for _ in range(n_episodes)]
    done = [False] * n_episodes
    chose_done = [False] * n_episodes
    done_prob_at_complete = []
    step_log = []
    t0 = time.time()

    step = 0
    while not all(done):
        step += 1
        for i in range(n_episodes):
            if done[i]:
                continue
            env = envs[i]
            # Legal set = unselected tools + DONE. Identical masking to the RL agent.
            cands = env.legal()
            action, scores = policy.select(obs[i], cands, scoring=scoring)

            # Was the gold set already complete BEFORE this action? If so, record
            # how much mass DONE held -- the stop signal, independent of argmax.
            if set(env.selected) == set(env.gold):
                dist = scores.distribution(scoring)
                done_prob_at_complete.append(dist.get("DONE", 0.0))

            if first[i] is None:
                first[i] = action
            if action == "DONE":
                chose_done[i] = True
            actions[i].append(action)
            step_log.append({
                "episode": i, "step": step, "action": action,
                "done_rank_pmi": scores.rank_of("DONE", "pmi"),
                "done_rank_sum": scores.rank_of("DONE", "sum"),
                "n_candidates": len(cands),
            })
            obs[i], r, d, infos[i] = env.step(action)
            done[i] = d
        print(f"    step {step}: {sum(not d for d in done)} active, "
              f"{time.time() - t0:.0f}s", flush=True)

    sem, pf, cond_ok, over, wasted, static_sem = [], [], [], [], [], []
    for i in range(n_episodes):
        info = infos[i]
        sel, gold = set(info["selected"]), set(info["gold"])
        sem.append(1.0 if sel == gold else 0.0)
        if info["is_conditional"]:
            pf.append(1.0 if first[i] == info["probe"] else 0.0)
            if info["probe"] in info["selected"]:
                cond_ok.append(1.0 if gold <= sel else 0.0)
        else:
            over.append(1.0 if info["probe"] in sel else 0.0)
            wasted.append(float(len(sel - gold)))
            static_sem.append(1.0 if sel == gold else 0.0)

    sem_at_k, probe_in_topk = [], []
    episodes = []
    for i in range(n_episodes):
        info = infos[i]
        gold = set(info["gold"])
        k = len(gold)
        topk = [a for a in actions[i] if a != "DONE"][:k]
        sem_at_k.append(1.0 if set(topk) == gold else 0.0)
        if not info["is_conditional"]:
            probe_in_topk.append(1.0 if info["probe"] in topk else 0.0)
        episodes.append({
            "episode": i, "scenario": info["scenario"],
            "is_conditional": info["is_conditional"],
            "gold": sorted(gold), "actions": list(actions[i]),
            "topk": topk, "sem_at_k": sem_at_k[-1],
        })

    m = lambda x: float(np.mean(x)) if len(x) else float("nan")
    return {
        "alpha": alpha, "n": n_episodes, "scoring": scoring,
        "static_ceiling": envs[0].static_ceiling(),
        "exact_match": m(sem),
        "done_rate": m([float(c) for c in chose_done]),
        "probe_first": m(pf),
        "conditions_on_result": m(cond_ok),
        "over_probe_rate": m(over),
        "wasted_tools_per_static_episode": m(wasted),
        "static_exact_match": m(static_sem),
        "done_prob_at_complete": m(done_prob_at_complete),
        "n_complete_states_seen": len(done_prob_at_complete),
        # Ranking quality, immune to the never-stopping degeneracy.
        "sem_at_k": m(sem_at_k),
        "probe_in_topk_static": m(probe_in_topk),
        "episodes": episodes,
        "mean_done_rank_pmi": m([s["done_rank_pmi"] for s in step_log]),
        "mean_done_rank_sum": m([s["done_rank_sum"] for s in step_log]),
        "wall_seconds": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--model_tag", default=None,
                    help="Short name used in the output filename; defaults to the "
                         "model's basename.")
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 1.0])
    # 'mean' is omitted by default: it gave numerically identical results to
    # 'sum' in all six configs of the first run. Pass it explicitly to re-check.
    ap.add_argument("--scorings", nargs="+", default=["sum", "pmi"])
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--out", default="results/llm_probe_constrained.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto", local_files_only=True)
    model.eval()
    policy = ConstrainedLLMPolicy(model, tok, batch_size=args.batch_size)
    print(f"model on {model.device}\n", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.model_tag:
        base, ext = os.path.splitext(args.out)
        args.out = f"{base}_{args.model_tag}{ext}"
    out = []
    for scoring in args.scorings:
        for alpha in args.alphas:
            print(f"\n[{scoring}] alpha={alpha} ...", flush=True)
            r = run(policy, alpha, args.episodes, scoring)
            r["model"] = args.model
            out.append(r)
            print(f"[{scoring}] alpha={alpha:.1f}  ceiling={r['static_ceiling']:.3f}  "
                  f"SEM={r['exact_match']:.3f}  DONE_rate={r['done_rate']:.3f}  "
                  f"SEM@k={r['sem_at_k']:.3f}  "
                  f"probe_in_topk={r['probe_in_topk_static']:.3f}  "
                  f"P(DONE|complete)={r['done_prob_at_complete']:.3f}  "
                  f"({r['wall_seconds']:.0f}s)", flush=True)
            with open(args.out, "w") as f:
                json.dump(out, f, indent=2)

    print("\n" + "=" * 70)
    head = [r for r in out if r["scoring"] == "sum" and r["alpha"] == 0.0]
    if head:
        r = head[0]
        print(f"HEADLINE (sum scoring, alpha=0, probe is NOT needed):")
        print(f"  SEM@k (ranking quality)    : {r['sem_at_k']:.3f} (ceiling 1.000)")
        print(f"  probe in top-k (over-gather): {r['probe_in_topk_static']:.3f}")
        print(f"  over-probe rate (all 5 steps): {r['over_probe_rate']:.3f}")
        print(f"  wasted tools per episode   : {r['wasted_tools_per_static_episode']:.2f}")
        print(f"  exact match                : {r['exact_match']:.3f} (ceiling 1.000)")
        print(f"  terminated via DONE        : {r['done_rate']:.3f}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
