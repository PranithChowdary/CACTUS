"""
compare_models.py

Aggregates every results/llm_probe_constrained*.json into the cross-model table
that answers the question the single-model run could not: is the termination
failure a property of LLaMA-3.1-8B, or of instruct models at this scale?

The decomposition is what matters, so each ability is reported separately
rather than collapsed into one score:

  recall  gold set contained in the first 5 picks -- does it know the tools?
  rank    SEM@k at alpha=0   -- can it pick the right tools FIRST?
  probe   probe_first at alpha=1 -- does it gather when it should?
  exploit right branch immediately after probing -- does it use what it learned?
  stop    DONE rate, mean rank of DONE, P(DONE | gold complete)

A model that scores high on recall/rank/probe and zero on stop reproduces the
LLaMA-3.1-8B pattern. A model that stops would falsify the generalization.

Two reporting defects this file used to have, both of which changed the reading
of the results:

1. The DONE-rank column was hardcoded to `mean_done_rank_pmi` while the rest of
   the table used `HEADLINE_SCORING` ("sum"). Under PMI no model ever stops, so
   every model printed 35.0 and the scale trend was invisible. With the sum
   ranks it is 28.6 (1B) -> 28.5 (8B) -> 21.9 (70B) -> 2.3 (Qwen).

2. Rates were printed over `--episodes` (48), but at alpha=0 the harness runs
   12 scenarios x 4 repeats and the policy is deterministic argmax, so all four
   repeats of a scenario are byte-identical. The effective n is 12 and every
   alpha=0 rate is a multiple of 1/12. The table now prints the scenario-level
   count next to each rate so a 1-in-12 result cannot read as "8%".

    python scripts/compare_models.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HEADLINE_SCORING = "sum"   # SEM@k 0.833 vs 0.167 for PMI; see RESEARCH_PLAN.md
RECALL_CUTOFF = 5          # gold contained in the first N tool picks


def exploit_rate(entry):
    """Of the episodes that correctly probed first, how often is the revealed
    branch chosen immediately at step 2?"""
    hits = tot = 0
    for e in entry.get("episodes", []):
        if not e["is_conditional"]:
            continue
        acts = [a for a in e["actions"] if a != "DONE"]
        gold = set(e["gold"])
        if not acts or acts[0] not in gold:
            continue
        tot += 1
        other = [g for g in gold if g != acts[0]]
        if other and len(acts) > 1 and acts[1] == other[0]:
            hits += 1
    return (hits / tot) if tot else float("nan"), tot


def recall_at(entry, cutoff):
    """Is the whole gold set among the first `cutoff` tool picks? Separates
    'does not know the tools' from 'does not rank them adjacently'."""
    eps = entry.get("episodes", [])
    if not eps:
        return float("nan")
    hit = 0
    for e in eps:
        tools = [a for a in e["actions"] if a != "DONE"][:cutoff]
        if set(e["gold"]) <= set(tools):
            hit += 1
    return hit / len(eps)


def scenario_stats(entry):
    """Collapse repeats to scenarios -- the real unit of evidence at alpha=0.

    Returns (n_scenarios, n_scenarios_that_stop, all_repeats_identical).
    """
    eps = entry.get("episodes", [])
    by = {}
    for e in eps:
        by.setdefault(e["scenario"], []).append(e)
    stops = sum(1 for reps in by.values()
                if any("DONE" in r["actions"] for r in reps))
    identical = all(len({tuple(r["actions"]) for r in reps}) == 1
                    for reps in by.values())
    return len(by), stops, identical


def main():
    ap = argparse.ArgumentParser()
    # The 61-scenario re-run writes results/llm_probe_bank61_*.json; the
    # 12-scenario files are kept under their old names as provenance, so the
    # pattern has to be selectable rather than hardcoded.
    ap.add_argument("--pattern", default="results/llm_probe_constrained_*.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files and args.pattern.startswith("results/llm_probe_constrained"):
        # Only the untagged original exists (pre-multi-model runs).
        files = sorted(glob.glob("results/llm_probe_constrained*.json"))
    if not files:
        print(f"no files matching {args.pattern!r}")
        return 1

    rows = []
    for f in files:
        data = json.load(open(f))
        tag = (os.path.basename(f).replace("llm_probe_constrained", "")
               .replace("llm_probe_bank61", "").replace(".json", "").strip("_"))
        tag = tag or (data[0].get("model", "?").split("/")[-1])
        a0 = next((d for d in data if d["scoring"] == HEADLINE_SCORING and d["alpha"] == 0.0), None)
        a1 = next((d for d in data if d["scoring"] == HEADLINE_SCORING and d["alpha"] == 1.0), None)
        if not (a0 and a1):
            print(f"  (skipping {f}: no '{HEADLINE_SCORING}' scoring at both alphas)")
            continue
        ex, ex_n = exploit_rate(a1)
        n_scen, n_stop_scen, identical = scenario_stats(a0)
        rows.append({
            "model": tag,
            "recall_at_5": recall_at(a0, RECALL_CUTOFF),
            "rank_sem_at_k": a0["sem_at_k"],
            "probe_first": a1["probe_first"],
            "exploit": ex, "exploit_n": ex_n,
            "done_rate": max(a0["done_rate"], a1["done_rate"]),
            # Was hardcoded to _pmi while the table header said "sum".
            "done_rank": a0[f"mean_done_rank_{HEADLINE_SCORING}"],
            "p_done_complete": max(a0["done_prob_at_complete"], a1["done_prob_at_complete"]),
            "spurious_probe": a0["probe_in_topk_static"],
            "n_episodes": a0["n"],
            "n_scenarios": n_scen,
            "stop_scenarios": n_stop_scen,
            "repeats_identical": identical,
        })

    print(f"\nscoring = {HEADLINE_SCORING}   (PMI reported as ablation in the raw JSON)")
    print(f"alpha=0 unit of evidence = SCENARIOS, not episodes "
          f"(deterministic argmax makes repeats identical)\n")
    print(f"{'model':<16}{'recall':>8}{'rank':>8}{'probe':>8}{'exploit':>14}"
          f"{'stop':>8}{'DONErank':>10}{'P(DONE)':>10}{'spur':>7}")
    print(f"{'':<16}{'@5':>8}{'SEM@k':>8}{'a=1':>8}{'branch@2':>14}"
          f"{'scen':>8}{'of ~35':>10}{'|cmpl':>10}{'a=0':>7}")
    print("-" * 89)
    for r in rows:
        ex = "nan" if np.isnan(r["exploit"]) else f"{r['exploit']:.3f}"
        ex_cell = f"{ex} (n={r['exploit_n']})"
        stop_cell = f"{r['stop_scenarios']}/{r['n_scenarios']}"
        print(f"{r['model']:<16}{r['recall_at_5']:>8.3f}{r['rank_sem_at_k']:>8.3f}"
              f"{r['probe_first']:>8.3f}{ex_cell:>14}{stop_cell:>8}"
              f"{r['done_rank']:>10.1f}{r['p_done_complete']:>10.4f}"
              f"{r['spurious_probe']:>7.3f}")

    print()
    # Report termination two ways, because they can disagree: a model can trip
    # a done-RATE threshold on a single scenario while holding ~0 probability
    # mass on DONE at the moment the gold set is actually complete. Both are
    # printed so the claim is auditable rather than threshold-dependent.
    stoppers = [r for r in rows if r["done_rate"] > 0.05]
    calibrated = [r for r in rows
                  if not np.isnan(r["p_done_complete"]) and r["p_done_complete"] > 0.10]
    if not rows:
        pass
    elif not stoppers:
        print(f"VERDICT: no model terminates ({len(rows)} tested). The termination")
        print("         failure is not specific to LLaMA-3.1-8B. Report as a")
        print("         property of this model class at this scale.")
    else:
        print("TERMINATION, two ways:")
        for r in rows:
            print(f"  {r['model']:<16} stops in {r['stop_scenarios']}/{r['n_scenarios']} scenarios"
                  f"   P(DONE | gold complete) = {r['p_done_complete']:.4f}")
        names = ", ".join(r["model"] for r in stoppers)
        cnames = ", ".join(r["model"] for r in calibrated) or "none"
        print(f"\n  crosses done_rate > 0.05        : {names}")
        print(f"  holds >10% mass on DONE at cmpl : {cnames}")
        if set(r["model"] for r in stoppers) != set(r["model"] for r in calibrated):
            print("\n  These two sets DIFFER. A model in the first but not the second")
            print("  reaches the threshold without the stop signal being calibrated --")
            print("  check how many distinct scenarios drive its done_rate before")
            print("  describing it as a model that terminates.")

    # With one episode per scenario there are no repeats, so the identical-repeat
    # test is vacuously true and the old wording printed "n_effective = [61] and
    # NOT [61]". Only warn when episodes actually exceed scenarios.
    degen = [r for r in rows if r["repeats_identical"] and r["n_episodes"] > r["n_scenarios"]]
    clean = [r for r in rows if r["n_episodes"] == r["n_scenarios"]]
    if degen:
        ns = sorted({r["n_scenarios"] for r in degen}); ne = sorted({r["n_episodes"] for r in degen})
        print(f"\n  note: for {len(degen)}/{len(rows)} models every alpha=0 repeat is identical "
              f"within\n        each scenario, so n_effective = {ns} and NOT {ne}. Every "
              f"alpha=0 rate\n        above is a multiple of 1/{ns[0]}; CIs must be taken over "
              f"scenarios.")
    elif clean:
        ns = sorted({r["n_scenarios"] for r in clean})
        print(f"\n  note: one episode per scenario, so n = {ns[0]} with no repeat inflation. "
              f"Rates\n        are multiples of 1/{ns[0]}; CIs are over scenarios.")

    out = args.out or ("results/model_comparison_bank61.json"
                       if "bank61" in args.pattern else "results/model_comparison.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
