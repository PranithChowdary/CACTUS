"""
alpha_shift.py  --  Phase 2' : over-gathering under alpha-shift.

The headline experiment. Everything before it establishes that the task is
non-degenerate and that LLMs cannot terminate; this one asks whether an agent
trained where gathering is almost always necessary learns *to always gather*
rather than *when to gather*.

Hypothesis (from RESEARCH_PLAN.md, committed before running)
------------------------------------------------------------
> If agents trained where every task requires gathering learn to always gather
> rather than when to gather, then an agent trained at alpha=1 and evaluated at
> alpha=0 will probe on tasks where probing is wasteful, losing exact match
> relative to an agent trained at the evaluation alpha. If instead they learn
> the policy conditional on need, probe rate will track the true need and
> performance will be flat across the alpha-shift.

Kill criterion: if the alpha=1-trained agent shows no elevated probe rate at
alpha=0, there is no over-gathering phenomenon and the paper has no headline.

The headline metric is STATIC PROBE RATE
----------------------------------------
On a static episode the probe tool is deliberately absent from the gold set
(design requirement 1), so selecting it is pure waste: a wrong tool AND a
wasted step. The fraction of static episodes on which an agent calls the probe
is therefore a direct, sign-unambiguous measure of over-gathering:

    a policy that gathers conditionally -> 0.0
    a policy that always gathers        -> 1.0

It is measured on a FIXED static-only evaluation set, so it is a property of
the agent and does not depend on the evaluation alpha. That matters: at
alpha=1 there are no static episodes to measure it on, so folding it into the
mixed-alpha curve would make the metric undefined exactly where the most
over-gathering agent lives.

The out-of-distribution confound, and how it is controlled
----------------------------------------------------------
The query embedding is drawn around one of TWO prototypes per query type:
index 1 for conditional episodes, index 0 for static ones. An agent trained at
alpha=1.0 has therefore NEVER SEEN a static prototype, so elevated probing at
alpha=0 could be dismissed as a response to out-of-distribution *input* rather
than as a learned always-gather policy. That dismissal would be correct and
would sink the result.

So alpha_train=0.9 is run alongside 1.0. At 0.9 the agent sees static
prototypes in 10% of training episodes -- static inputs are in-distribution,
only their FREQUENCY is low. If over-gathering survives at 0.9 the confound is
ruled out and the effect is about the learned policy, not about novel inputs.
Report both; 0.9 is the one that carries the claim.

Controls
--------
- MATCHED agent: trained at the evaluation alpha. This is what makes the result
  a transfer failure rather than just "this task is hard". Same architecture,
  same budget, same seeds.
- Analytic memoryless ceiling (1-a) + a/2, which no query-only model can beat.
- Paired evaluation: every agent is evaluated on the identical fixed episode
  list (same query types, same latent bits, same embedding noise), so
  differences are not evaluation-draw noise and a paired test is valid.
- Capacity gate: an alpha=0-trained agent must reach high exact match at
  alpha=0 or nothing downstream is interpretable. A 40k-timestep pilot of the
  alpha sweep produced a CONFIDENT FALSE NEGATIVE (PPO below the chance
  ceiling) purely from undertraining, so this gate is checked and reported
  before any shift number is read.

Usage
-----
    python experiments/alpha_shift.py --timesteps 200000 --seeds 5
    sbatch scripts/run_alpha_shift.sh
"""
import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

from env.conditional_tool_env import ConditionalToolEnv


def build_env(kind, alpha, seed):
    """Both environments expose the same reset options and info keys, so every
    gate, metric and report below is shared rather than duplicated per env."""
    if kind == "synthetic":
        return ConditionalToolEnv(alpha=alpha, seed=seed)
    from env.nl_gym_env import NLGymEnv, load_embeddings
    global _NL_EMB
    try:
        _NL_EMB
    except NameError:
        _NL_EMB = load_embeddings(verbose=True)
    return NLGymEnv(alpha=alpha, seed=seed, embeddings=_NL_EMB)

# Gate: an alpha=0-trained agent evaluated at alpha=0 must clear this, or the
# training budget is too small and every shift number is uninterpretable.
CAPACITY_GATE = 0.80
# Every agent must also be competent on its OWN training distribution before
# its transfer behaviour means anything. Without this, an agent that simply
# failed to learn reports static_probe_rate ~ 0 and trips the KILL CRITERION --
# "no over-gathering" -- when the truth is "no policy". A 25k-step smoke run
# did exactly that: the alpha=0.9 agent scored 0.020 at alpha=1 and the script
# announced the phenomenon was absent.
OWN_ALPHA_GATE = 0.50


def mask_fn(env):
    return env.unwrapped.action_masks()


# --------------------------------------------------------------- eval fixtures

def make_eval_set(n, alpha, n_query_types, seed):
    """A fixed list of fully-specified episodes.

    Every field that varies per episode is pinned here -- query type, whether
    the episode is conditional, the latent branch bit, and the seed that drives
    the embedding noise -- so that every agent is evaluated on byte-identical
    observations. Without this, two agents differ partly because they saw
    different episodes, and a paired test is not valid.
    """
    rng = np.random.default_rng(seed)
    return [{
        "query_type": int(rng.integers(0, n_query_types)),
        "is_conditional": bool(rng.random() < alpha),
        "latent_bit": int(rng.integers(0, 2)),
        "seed": int(seed * 1_000_003 + i),
    } for i in range(n)]


def rollout(model, env, spec):
    """Run one pinned episode; return the per-episode record."""
    obs, info = env.reset(seed=spec["seed"], options={
        "query_type": spec["query_type"],
        "is_conditional": spec["is_conditional"],
        "latent_bit": spec["latent_bit"],
    })
    probe = info["probe"]
    done = False
    chose_done = False
    while not done:
        masks = env.unwrapped.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        action = int(action)
        if action == env.unwrapped.done_idx:
            chose_done = True
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc

    sel, gold = set(info["selected"]), set(info["gold"])
    tp = len(sel & gold)
    p = tp / max(1, len(sel))
    rc = tp / max(1, len(gold))
    return {
        "is_conditional": bool(info["is_conditional"]),
        "sem": 1.0 if sel == gold else 0.0,
        "f1": (2 * p * rc / (p + rc)) if (p + rc) > 0 else 0.0,
        "probed": 1.0 if probe in sel else 0.0,
        "wasted": float(len(sel - gold)),
        "n_selected": float(len(sel)),
        "chose_done": 1.0 if chose_done else 0.0,
        "steps": int(info["step_count"]),
    }


def evaluate(model, env, specs):
    recs = [rollout(model, env, s) for s in specs]
    stat = [r for r in recs if not r["is_conditional"]]
    cond = [r for r in recs if r["is_conditional"]]
    m = lambda xs, k: float(np.mean([x[k] for x in xs])) if xs else float("nan")
    return {
        "sem": m(recs, "sem"), "f1": m(recs, "f1"),
        "done_rate": m(recs, "chose_done"),
        "mean_selected": m(recs, "n_selected"),
        # HEADLINE: probing a static episode is pure waste.
        "static_probe_rate": m(stat, "probed"),
        "static_sem": m(stat, "sem"),
        "static_wasted": m(stat, "wasted"),
        "cond_probe_rate": m(cond, "probed"),
        "cond_sem": m(cond, "sem"),
        "n_static": len(stat), "n_cond": len(cond),
        "per_episode_sem": [r["sem"] for r in recs],
        "per_static_probed": [r["probed"] for r in stat],
    }


# ------------------------------------------------------------------- training

def train(alpha, seed, timesteps, ent_coef, env_kind="synthetic", net_arch=None):
    def make():
        return ActionMasker(build_env(env_kind, alpha, seed), mask_fn)
    venv = DummyVecEnv([make])
    # ent_coef > 0 is load-bearing: probing pays off only at the NEXT step, and
    # under greedy exploration the agent converges to guessing the branch
    # instead of gathering the bit.
    # Net width is load-bearing on the NL env and must be stated, not left to
    # the default. The observation is ~806-d there (384 query + 384 result + 37
    # selection + 1 flag) and 122 distinct query->gold mappings have to be
    # represented; SB3's default 2x64 trunk is a hard bottleneck. Choosing
    # capacity so the policy class can REACH the task's ceiling is the same
    # correction made for embedding_dim in the synthetic env -- otherwise the
    # agent is handicapped representationally and its loss says nothing about
    # sequential decision-making.
    pk = {"net_arch": list(net_arch)} if net_arch else None
    model = MaskablePPO("MultiInputPolicy", venv, seed=seed, verbose=0,
                        learning_rate=3e-4, batch_size=64, gamma=0.99,
                        ent_coef=ent_coef, policy_kwargs=pk)
    model.learn(total_timesteps=timesteps)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_alphas", type=float, nargs="+",
                    default=[0.0, 0.5, 0.9, 1.0],
                    help="0.9 is the confound control: static inputs are "
                         "in-distribution there, only rare.")
    ap.add_argument("--eval_alphas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--timesteps", type=int, default=200000)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--n_eval", type=int, default=400)
    ap.add_argument("--env", choices=["synthetic", "nl"], default="synthetic",
                    help="'nl' runs the SAME 61-scenario natural-language task "
                         "the constrained LLM policy plays, making the LLM-vs-RL "
                         "contrast same-environment instead of cross-environment.")
    ap.add_argument("--net_arch", type=int, nargs="*", default=None,
                    help="Policy/value trunk widths, e.g. --net_arch 256 256. "
                         "Required on --env nl; the default 2x64 underfits.")
    ap.add_argument("--out", default="results/alpha_shift.json")
    args = ap.parse_args()

    probe_env = build_env(args.env, 0.0, 0)
    n_qt = probe_env.n_query_types
    print(f"env={args.env}  query types/scenarios={n_qt}  "
          f"actions={probe_env.action_space.n}", flush=True)

    # Fixed evaluation sets, built ONCE and shared by every agent.
    # Each training alpha must be an evaluation point too, or an agent's
    # competence on its own distribution cannot be checked.
    all_eval = sorted(set(args.eval_alphas) | set(args.train_alphas))
    args.eval_alphas = all_eval
    eval_sets = {a: make_eval_set(args.n_eval, a, n_qt, seed=777)
                 for a in all_eval}
    # Static-only set for the headline metric. alpha=0.0 -> every episode static.
    static_set = make_eval_set(args.n_eval, 0.0, n_qt, seed=4242)

    out = {
        "config": vars(args),
        # Key format must match the f"{a:.2f}" used everywhere else; str(a)
        # gives '0.0' and report() then raises KeyError on '0.00'.
        "static_ceiling": {f"{a:.2f}": (1 - a) + a / 2 for a in args.eval_alphas},
        "runs": {},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t_start = time.time()

    for ta in args.train_alphas:
        key = f"{ta:.2f}"
        out["runs"][key] = {"train_alpha": ta, "seeds": []}
        for seed in range(args.seeds):
            t0 = time.time()
            model = train(ta, seed, args.timesteps, args.ent_coef, args.env,
                          args.net_arch)
            env = build_env(args.env, 0.0, seed + 10_000)

            rec = {"seed": seed,
                   "static_only": evaluate(model, env, static_set),
                   "by_eval_alpha": {}}
            for ea in args.eval_alphas:
                rec["by_eval_alpha"][f"{ea:.2f}"] = evaluate(model, env, eval_sets[ea])

            out["runs"][key]["seeds"].append(rec)
            sp = rec["static_only"]["static_probe_rate"]
            # Report the extremes of whatever was actually requested rather than
            # hardcoding "0.00"/"1.00", which KeyErrors on a custom --eval_alphas.
            lo_k = f"{min(args.eval_alphas):.2f}"
            hi_k = f"{max(args.eval_alphas):.2f}"
            s0 = rec["by_eval_alpha"][lo_k]["sem"]
            s1 = rec["by_eval_alpha"][hi_k]["sem"]
            print(f"  train_a={ta:.2f} seed={seed}  static_probe={sp:.3f}  "
                  f"SEM@a{lo_k}={s0:.3f}  SEM@a{hi_k}={s1:.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

            with open(args.out, "w") as f:
                json.dump(out, f, indent=2)

        agg = lambda path: [
            _dig(r, path) for r in out["runs"][key]["seeds"]]
        sps = agg(["static_only", "static_probe_rate"])
        out["runs"][key]["static_probe_mean"] = float(np.mean(sps))
        out["runs"][key]["static_probe_std"] = float(np.std(sps, ddof=1)) if len(sps) > 1 else 0.0
        print(f"train_alpha={ta:.2f}  STATIC PROBE RATE = "
              f"{out['runs'][key]['static_probe_mean']:.3f} "
              f"+/- {out['runs'][key]['static_probe_std']:.3f}\n", flush=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)

    report(out, args)
    print(f"\nwrote {args.out}   total {time.time()-t_start:.0f}s")


def _dig(d, path):
    for p in path:
        d = d[p]
    return d


def report(out, args):
    runs = out["runs"]
    print("\n" + "=" * 78)
    print("CAPACITY GATE (must pass before anything below is interpretable)")
    print("=" * 78)
    gate_key = "0.00"
    gate_eval = f"{min(args.eval_alphas):.2f}"
    if gate_key in runs and gate_eval == "0.00":
        v = [_dig(r, ["by_eval_alpha", gate_eval, "sem"]) for r in runs[gate_key]["seeds"]]
        mv = float(np.mean(v))
        ok = mv >= CAPACITY_GATE
        print(f"  alpha=0-trained agent, SEM at alpha=0: {mv:.3f} "
              f"(gate {CAPACITY_GATE:.2f})  -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            print("  !! FAIL: the agent is undertrained, not over-gathering. A 40k")
            print("  !! pilot previously scored BELOW the chance ceiling for this")
            print("  !! reason and inverted the conclusion. Raise --timesteps.")
    else:
        print("  (needs alpha_train=0.0 AND alpha_eval=0.0 -- gate not evaluated)")

    print("\n" + "=" * 78)
    print("HEADLINE: static probe rate by training alpha")
    print("  (probe is NOT in static gold -> this is pure waste)")
    print("=" * 78)
    print(f"  {'train_alpha':>12}{'static_probe':>17}{'static_SEM':>13}{'wasted/ep':>12}")
    for k in sorted(runs):
        ss = [r["static_only"] for r in runs[k]["seeds"]]
        f = lambda n: float(np.mean([s[n] for s in ss]))
        g = lambda n: float(np.std([s[n] for s in ss], ddof=1)) if len(ss) > 1 else 0.0
        print(f"  {k:>12}{f('static_probe_rate'):>9.3f}+/-{g('static_probe_rate'):<5.3f}"
              f"{f('static_sem'):>13.3f}{f('static_wasted'):>12.2f}")

    print("\n" + "=" * 78)
    print("TRANSFER: SEM by (train_alpha, eval_alpha); MATCHED is the diagonal")
    print("=" * 78)
    eas = [f"{a:.2f}" for a in args.eval_alphas]
    hdr = "train \\ eval"
    print(f"  {hdr:>12}" + "".join(f"{e:>9}" for e in eas))
    for k in sorted(runs):
        cells = []
        for e in eas:
            v = [_dig(r, ["by_eval_alpha", e, "sem"]) for r in runs[k]["seeds"]]
            cells.append(f"{float(np.mean(v)):>9.3f}")
        print(f"  {k:>12}" + "".join(cells))
    print(f"  {'ceiling*':>12}" + "".join(
        f"{out['static_ceiling'][e]:>9.3f}" for e in eas))
    print("  * analytic bound for any memoryless (query-only) model, not for the agent")

    print("\n" + "=" * 78)
    print("OWN-DISTRIBUTION COMPETENCE (gate for reading the rows above)")
    print("  each agent evaluated at the alpha it was TRAINED on")
    print("=" * 78)
    own = {}
    for k in sorted(runs):
        ta = runs[k]["train_alpha"] if "train_alpha" in runs[k] else float(k)
        kk = f"{ta:.2f}"
        vals = [_dig(r, ["by_eval_alpha", kk, "sem"]) for r in runs[k]["seeds"]
                if kk in r["by_eval_alpha"]]
        if not vals:
            continue
        mv = float(np.mean(vals))
        own[k] = mv
        print(f"  train_alpha={k}  SEM at own alpha = {mv:.3f} "
              f"(gate {OWN_ALPHA_GATE:.2f})  -> {'PASS' if mv >= OWN_ALPHA_GATE else 'FAIL -- transfer numbers UNINTERPRETABLE'}")

    # The claim, stated as a number rather than a narrative.
    print("\n" + "=" * 78)
    if "0.00" in runs and "0.90" in runs:
        lo = float(np.mean([r["static_only"]["static_probe_rate"] for r in runs["0.00"]["seeds"]]))
        hi = float(np.mean([r["static_only"]["static_probe_rate"] for r in runs["0.90"]["seeds"]]))
        print(f"OVER-GATHERING (confound-controlled, train_alpha=0.90):")
        print(f"  static probe rate {lo:.3f} (matched, a_train=0) -> {hi:.3f} (a_train=0.9)")
        bad = [k for k in ("0.00", "0.90") if own.get(k, 0.0) < OWN_ALPHA_GATE]
        if bad:
            print(f"  NOT INTERPRETABLE: agent(s) {bad} failed the own-distribution")
            print(f"  gate, so a low probe rate means 'no policy', not 'no")
            print(f"  over-gathering'. Raise --timesteps and re-run before reading")
            print(f"  anything into {lo:.3f} -> {hi:.3f}.")
        elif hi - lo > 0.10:
            print(f"  Delta = +{hi - lo:.3f}. The agent probes on tasks where probing is")
            print(f"  wasteful, and static inputs were IN-distribution during its")
            print(f"  training. Over-gathering is a learned-policy failure.")
        else:
            print(f"  Delta = {hi - lo:+.3f}. KILL CRITERION MET: no elevated probe rate.")
            print(f"  There is no over-gathering phenomenon to report.")
    print("=" * 78)


if __name__ == "__main__":
    main()
