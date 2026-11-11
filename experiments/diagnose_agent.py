"""
diagnose_agent.py

Before trusting any alpha-sweep result, establish which world we are in: is the
sequential policy failing because the hypothesis is wrong, or because it is
undertrained / miswired? A sweep run on a broken agent produces a confident
false negative.

Three checks, cheapest first:

1. CAPACITY CHECK (alpha=0). The task is static and a classifier solves it
   perfectly. If PPO cannot approach 1.0 here given budget, nothing at higher
   alpha is interpretable -- the agent underfits the easy case.

2. BEHAVIOURAL CHECK (alpha=1). Does the agent actually probe first? An agent
   that never executes the probe tool before committing cannot benefit from the
   result channel no matter how well the channel is encoded. This separates
   "did not learn to gather information" from "gathered it and ignored it".

3. CONDITIONING CHECK (alpha=1). Given the bit HAS been revealed, does the
   agent's next action depend on it? We force the probe action, then compare
   the greedy follow-up under the two possible reveals. If the action is
   identical, the policy is ignoring the channel.

Usage:
    python experiments/diagnose_agent.py --timesteps 150000
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

from env.conditional_tool_env import ConditionalToolEnv, OBS_BRANCH_A, OBS_BRANCH_B


def mask_fn(env):
    return env.unwrapped.action_masks()


def train(alpha, seed, timesteps, ent_coef=0.0):
    venv = DummyVecEnv([lambda: ActionMasker(ConditionalToolEnv(alpha=alpha, seed=seed), mask_fn)])
    model = MaskablePPO("MultiInputPolicy", venv, seed=seed, verbose=0,
                        learning_rate=3e-4, batch_size=64, gamma=0.99, ent_coef=ent_coef)
    model.learn(total_timesteps=timesteps)
    return model


def rollout_stats(model, env, n=400):
    sem, probe_first, correct_branch = [], [], []
    for _ in range(n):
        obs, info = env.reset()
        probe = int(env.probe_tool[info["query_type"]])
        first_action = None
        done = False
        while not done:
            masks = env.action_masks()
            a, _ = model.predict(obs, deterministic=True, action_masks=masks)
            a = int(a)
            if first_action is None:
                first_action = a
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
        sem.append(1.0 if set(info["selected"]) == set(info["gold"]) else 0.0)
        if info["is_conditional"]:
            probe_first.append(1.0 if first_action == probe else 0.0)
            correct_branch.append(1.0 if set(info["gold"]) <= set(info["selected"]) else 0.0)
    return (float(np.mean(sem)),
            float(np.mean(probe_first)) if probe_first else float("nan"),
            float(np.mean(correct_branch)) if correct_branch else float("nan"))


def conditioning_check(model, env, n=200):
    """Force the probe, then ask whether the greedy next action differs between
    the two possible reveals. Fraction differing = how much the policy uses the
    channel."""
    differs = []
    for _ in range(n):
        obs, info = env.reset(options={"is_conditional": True})
        qt = info["query_type"]
        probe = int(env.probe_tool[qt])
        env.step(probe)
        base = env._get_obs()
        acts = []
        for reveal in (OBS_BRANCH_A, OBS_BRANCH_B):
            o = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()}
            r = np.zeros(3, dtype=np.float32)
            r[reveal] = 1.0
            o["observation"] = r
            masks = env.action_masks()
            a, _ = model.predict(o, deterministic=True, action_masks=masks)
            acts.append(int(a))
        differs.append(1.0 if acts[0] != acts[1] else 0.0)
    return float(np.mean(differs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=150000)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    args = ap.parse_args()

    print("=" * 66)
    print("1. CAPACITY CHECK  (alpha=0, static task, classifier gets 1.000)")
    print("=" * 66)
    m0 = train(0.0, 0, args.timesteps, args.ent_coef)
    e0 = ConditionalToolEnv(alpha=0.0, seed=10_000)
    sem0, _, _ = rollout_stats(m0, e0)
    print(f"   PPO exact-match = {sem0:.3f}   (needs to approach 1.000)")
    print(f"   -> {'OK: agent can fit the easy task' if sem0 > 0.9 else 'UNDERFIT: budget/wiring problem, higher alpha is uninterpretable'}")

    print()
    print("=" * 66)
    print("2+3. BEHAVIOURAL + CONDITIONING CHECK  (alpha=1, ceiling=0.500)")
    print("=" * 66)
    m1 = train(1.0, 0, args.timesteps, args.ent_coef)
    e1 = ConditionalToolEnv(alpha=1.0, seed=10_000)
    sem1, pf, cb = rollout_stats(m1, e1)
    uses = conditioning_check(m1, e1)
    print(f"   PPO exact-match          = {sem1:.3f}   (ceiling for query-only models = 0.500)")
    print(f"   probes first             = {pf:.3f}   (needs ~1.0 to gather the bit)")
    print(f"   recovers full gold set   = {cb:.3f}")
    print(f"   next action depends on revealed bit = {uses:.3f}   (0 = channel ignored)")
    print()
    if sem1 > 0.55:
        print("   VERDICT: agent exceeds the analytic query-only ceiling -> the")
        print("            information-gathering hypothesis survives this test.")
    elif pf < 0.5:
        print("   VERDICT: agent does not probe first -> it never gathers the bit.")
        print("            Failure is exploration/credit-assignment, not the hypothesis.")
    elif uses < 0.3:
        print("   VERDICT: agent probes but ignores the revealed bit -> representation")
        print("            or capacity problem, not the hypothesis.")
    else:
        print("   VERDICT: agent probes AND conditions on the bit yet stays at ceiling")
        print("            -> this is genuine evidence against the hypothesis.")


if __name__ == "__main__":
    main()
