"""
sweep_alpha.py

The experiment that decides whether sequential decision-making is justified for
tool selection, by sweeping how much of the task actually requires it.

Hypothesis
----------
If the advantage of a sequential policy comes from modelling information gained
by executing tools, then as alpha (the fraction of execution-dependent queries)
rises from 0 to 1, a memoryless classifier should degrade toward its
information-theoretic ceiling (1 - alpha) + alpha/2 while a sequential policy
that can observe tool results does not, producing a crossover at some alpha* < 1.

Competing prediction
--------------------
If the advantage instead comes from capacity, masking, or tuning, both methods
degrade in parallel and no crossover appears.

Kill criterion (committed before running)
-----------------------------------------
If at alpha = 1.0 the classifier still matches the sequential policy within seed
noise, the sequential framing is dead for this task family.

Controls
--------
- Both methods see the same environment, same query distribution, same episode
  budget, and the same frozen feature representation.
- The classifier is given the target cardinality via top-k, which if anything
  favours it -- the agent has to learn when to stop.
- BLIND-PPO: an identical PPO agent with the tool-result channel zeroed out.
  This separates "sequential policy" from "access to execution results". If
  masked PPO wins only because of the observation channel, BLIND-PPO should
  track the classifier, not the full agent.
- static_ceiling() is plotted as a analytic bound, so undertraining the
  classifier cannot masquerade as a win for RL.

Usage:
    python experiments/sweep_alpha.py --timesteps 60000 --seeds 3
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

from env.conditional_tool_env import ConditionalToolEnv
from models.supervised_baseline import SupervisedToolSelector, exact_match, macro_f1

N_EVAL = 400


def mask_fn(env):
    return env.unwrapped.action_masks()


class BlindWrapper(__import__("gymnasium").ObservationWrapper):
    """Zeroes the tool-result channel. Isolates the contribution of observing
    execution results from the contribution of being sequential at all."""

    def observation(self, obs):
        obs = dict(obs)
        obs["observation"] = np.zeros_like(obs["observation"])
        return obs

    def action_masks(self):
        return self.env.unwrapped.action_masks()


def collect_supervised_data(env, n, rng):
    X, Y = [], []
    for _ in range(n):
        obs, info = env.reset()
        X.append(obs["embedding"])
        y = np.zeros(env.n_actions - 1)
        for t in info["gold"]:
            y[t] = 1
        Y.append(y)
    return np.array(X), np.array(Y)


def eval_supervised(clf, env, n, k=2):
    X, Y = [], []
    conds = []
    for _ in range(n):
        obs, info = env.reset()
        X.append(obs["embedding"])
        y = np.zeros(env.n_actions - 1)
        for t in info["gold"]:
            y[t] = 1
        Y.append(y)
        conds.append(info["is_conditional"])
    X, Y = np.array(X), np.array(Y)
    golds = [set(np.where(r == 1)[0].tolist()) for r in Y]
    preds = clf.predict(X, k=k)
    return exact_match(preds, golds), macro_f1(preds, golds)


def eval_agent(model, env, n):
    sems, f1s = [], []
    for _ in range(n):
        obs, info = env.reset()
        done = False
        while not done:
            masks = env.unwrapped.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, r, term, trunc, info = env.step(int(action))
            done = term or trunc
        sel, gold = set(info["selected"]), set(info["gold"])
        sems.append(1.0 if sel == gold else 0.0)
        tp = len(sel & gold)
        p = tp / max(1, len(sel))
        rc = tp / max(1, len(gold))
        f1s.append(2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0)
    return float(np.mean(sems)), float(np.mean(f1s))


def train_agent(alpha, seed, timesteps, blind=False, ent_coef=0.01):
    def make():
        e = ConditionalToolEnv(alpha=alpha, seed=seed)
        if blind:
            e = BlindWrapper(e)
        return ActionMasker(e, mask_fn)
    venv = DummyVecEnv([make])
    # ent_coef > 0 matters here: probing has delayed value, and with pure
    # greedy exploration the agent converges to guessing the branch instead of
    # gathering the bit (probe-first rate collapses). See diagnose_agent.py.
    model = MaskablePPO("MultiInputPolicy", venv, seed=seed, verbose=0,
                        learning_rate=3e-4, batch_size=64, gamma=0.99,
                        ent_coef=ent_coef)
    model.learn(total_timesteps=timesteps)
    eval_env = ConditionalToolEnv(alpha=alpha, seed=seed + 10_000)
    if blind:
        eval_env = BlindWrapper(eval_env)
    return model, eval_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--timesteps", type=int, default=60000)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--out", type=str, default="results/alpha_sweep.json")
    args = ap.parse_args()

    results = {}
    for alpha in args.alphas:
        ceiling = ConditionalToolEnv(alpha=alpha).static_ceiling()
        row = {"static_ceiling": ceiling, "supervised": [], "ppo": [], "ppo_blind": []}

        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)

            # --- memoryless classifier -------------------------------------
            tr_env = ConditionalToolEnv(alpha=alpha, seed=seed)
            te_env = ConditionalToolEnv(alpha=alpha, seed=seed + 10_000)
            Xtr, Ytr = collect_supervised_data(tr_env, 4000, rng)
            clf = SupervisedToolSelector(n_tools=tr_env.n_actions - 1).fit(Xtr, Ytr)
            sem, f1 = eval_supervised(clf, te_env, N_EVAL, k=2)
            row["supervised"].append({"sem": sem, "f1": f1})

            # --- sequential policy WITH tool-result channel -----------------
            model, ev = train_agent(alpha, seed, args.timesteps, blind=False,
                                    ent_coef=args.ent_coef)
            sem, f1 = eval_agent(model, ev, N_EVAL)
            row["ppo"].append({"sem": sem, "f1": f1})

            # --- sequential policy WITHOUT the channel (control) ------------
            model_b, ev_b = train_agent(alpha, seed, args.timesteps, blind=True,
                                        ent_coef=args.ent_coef)
            sem, f1 = eval_agent(model_b, ev_b, N_EVAL)
            row["ppo_blind"].append({"sem": sem, "f1": f1})

        for key in ["supervised", "ppo", "ppo_blind"]:
            sems = [r["sem"] for r in row[key]]
            row[f"{key}_mean"] = float(np.mean(sems))
            row[f"{key}_std"] = float(np.std(sems, ddof=1)) if len(sems) > 1 else 0.0

        results[str(alpha)] = row
        print(f"alpha={alpha:.2f}  ceiling={ceiling:.3f}  "
              f"supervised={row['supervised_mean']:.3f}±{row['supervised_std']:.3f}  "
              f"PPO={row['ppo_mean']:.3f}±{row['ppo_std']:.3f}  "
              f"PPO-blind={row['ppo_blind_mean']:.3f}±{row['ppo_blind_std']:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
