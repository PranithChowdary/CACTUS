# Research plan: measuring over-gathering in information-seeking agents

**Status:** Phase 0 (gates). Last updated after the literature review of 2026-10-05.

---

## The claim

> Information-gathering agents are evaluated only on tasks that require information
> gathering. We introduce a controlled benchmark that varies the fraction `α` of
> execution-dependent tasks, derive the analytic ceiling for any memoryless policy,
> and show that methods trained where every task needs gathering **over-gather**
> when tasks do not — a cost invisible to current evaluations.

This is an evaluation/analysis contribution, not a method contribution. That is
deliberate: see "What the literature already owns".

## Why this and not the original CACTUS paper

On MetaTool, a logistic-regression probe over the same frozen embeddings, trained
on the same 397 examples, is **statistically tied** with masked PPO on exact match
and **significantly better** on F1:

| | SEM | F1 |
|---|---|---|
| CACTUS (MaskablePPO, 5 seeds) | 0.573 ± 0.067 | 0.806 ± 0.031 |
| Supervised probe (threshold tuned on val) | 0.588 | 0.850 (p = 0.027 vs CACTUS) |
| Supervised probe (top-k=2) | 0.745 | 0.873 |

`results/significance.json`. The gold tool set is a deterministic function of the
query, so the interaction history carries no information the query lacks and the
MDP is degenerate. Sequential decision-making can only pay off where the agent
**gains information by acting** — which is what the α-benchmark isolates.

## What the literature already owns (do not claim these)

| Claim | Owned by |
|---|---|
| Teaching models *when to stop* gathering | CaRT, arXiv:2510.08517 |
| RL reward for *eliciting genuinely new information* | arXiv:2507.21389 (evidence-sentence reward) |
| RL generalizes where SFT memorizes | Chu et al. 2501.17161 and follow-ups |
| RL > SFT specifically for information gathering | arXiv:2507.21389 (their SFT scores *below* the base model) |

## What is open (confirmed against both papers' own stated limitations)

1. **Neither controls task composition.** CaRT: "Neither dataset isolates a
   controlled subset requiring no information gathering." 2507.21389: "no
   measurement of naturally-occurring task clarity levels." So neither can
   separate *learned when to gather* from *learned to always gather*.
2. **Neither measures over-gathering.** CaRT: "No explicit measurement of
   over-gathering." 2507.21389: "does not explicitly measure over-asking on
   already-specified tasks."
3. **Neither derives a ceiling or runs a memoryless baseline**, so neither can
   say how much of its gain required sequentiality at all.

Corroborating hook: CaRT itself reports that "RL tends toward longer conversations
without clear benefit" — over-gathering, observed and unexplained, in the nearest
paper.

## Result (2026-10-05): the thesis was wrong, and the corrected one is sharper

The constrained-decoding run (`results/llm_probe_constrained.json`, 48 episodes
x 2 alpha x 3 scorings on LLaMA-3.1-8B) decomposes the failure precisely. The
original hypothesis -- that agents over-gather because they misjudge whether
information is needed -- does **not** survive contact with the data.

| Ability | Measurement | Verdict |
|---|---|---|
| Rank relevant tools | alpha=0, gold occupies ranks 1-2 in **40/48**; SEM@k = 0.833 | **good** |
| Decide to gather | alpha=1, probes correctly at step 1 in **44/48** (0.917) | **good** |
| Act on what it learned | of those 44, only **13** pick the right branch at step 2; 23 get there later, 8 never | **poor** |
| Stop | `DONE` chosen **0/288**, ranked **35th of 35** at every step, P(DONE \| gold complete) = 0.0000 | **absent** |

Over-gathering here is **not** a relevance failure and **not** a misjudged
information need. The model knows which tools and knows to probe. It cannot
stop, and it does not immediately exploit the result it just obtained. It
spuriously probes on static tasks only 17% of the time -- the wasted calls come
from exhausting the step budget, not from wanting the probe.

Revised claim:

> Over-gathering in LLM tool agents is a **termination failure, not a relevance
> failure**. Given a masked action set, LLaMA-3.1-8B ranks the correct tools
> first in 83% of static tasks and probes before committing in 92% of
> execution-dependent ones, yet assigns the stop action zero probability mass
> (bottom rank of 35 candidates at every step, under three scoring schemes).
> A secondary failure follows: after probing, the revealed branch is ranked
> first only 30% of the time.

This explains mechanically why CaRT needed counterfactual supervision for
termination specifically, and gives the quantity that work lacks.

### Methodological note: PMI calibration was the wrong primary scorer

PMI was chosen a priori to remove surface-form competition, and it does reduce
length bias (r = -0.149 vs -0.535 for raw sum). But it **ranks worse on the
actual task**: SEM@k 0.167 vs 0.833 for raw sum at alpha=0. Removing length
bias is not the same as improving task fit; PMI over-corrects toward rare
strings. The headline should use summed log-probability, with PMI reported as
an ablation. Logging all three scorings is what made this visible -- had the
primary been fixed in advance, the ranking result would have been reported as
0.167 and read as "the model cannot rank tools".

## Hypothesis (falsifiable, with the competing prediction)

> If agents trained where every task requires gathering learn *to always gather*
> rather than *when to gather*, then an agent trained at α=1 and evaluated at α=0
> will probe on tasks where probing is wasteful, losing exact match relative to an
> agent trained at the evaluation α. If instead they learn the policy conditional
> on need, probe rate will track the true need and performance will be flat
> across the α-shift.

**Kill criterion:** if an α=1-trained agent shows no elevated probe rate at α=0,
there is no over-gathering phenomenon to report and the paper has no headline.

## Phases

| Phase | Work | Gate / output | Cost |
|---|---|---|---|
| **0a** | Literature review | **DONE** — gap confirmed above | — |
| **0b** | Supervised probe on MetaTool | **DONE** — `results/raw/Supervised.json` | — |
| **0c** | Fix the NL harness: probe must be wasteful at α=0; expand scenario bank past n=7; correct the over-probing metric | `llm_probe_test` interpretable | days |
| **1** | α-sweep, 5 α × 5 seeds, gated on `diagnose_agent.py` passing the α=0 capacity check | `results/alpha_sweep.json` + figure | hours (cluster) |
| **2′** | **Over-gathering under α-shift.** Train at α=1, evaluate across α. Measure probe rate, wasted steps, exact match. This is the headline figure. | the paper's core result | days |
| **3** | One real benchmark with genuine execution-dependence (τ-bench / AppWorld / TaskBench — verify contents first) | external validity | 2–4 weeks |
| **4** | Paper → arXiv → venue | | |

Phase 2 of the earlier plan (LoRA-RL to show RL > SFT) is **cut**: 2507.21389
already established it.

Realistic total: **4–6 weeks**.

## Design requirements the benchmark must satisfy

These are not optional; each one was a bug caught in review:

1. **At α=0 the probe must NOT be in the gold set.** Otherwise probing is never
   wasteful and over-gathering cannot be measured. (Violated by the original
   design, where `static_partner == branch[0]` and the probe was always gold.)
2. **The query must signal whether the task is conditional**, but never which
   branch. This keeps the ceiling at `(1-α) + α/2` while making the probe
   decision learnable in-distribution — so over-gathering is a *transfer*
   failure, not an impossible task.
3. **The latent bit must be unrecoverable from the query.** Verified
   empirically: a classifier predicting it from the query scores 0.488 (chance).
4. **Scenario bank must be large enough that greedy decoding does not collapse
   the effective sample size.** The 7-scenario bank gave n_effective = 7, not
   140; every metric was a multiple of 1/7.

## Venue

NeurIPS Datasets & Benchmarks, or an analysis paper at ICLR/ICML. Not a method
paper — the method claims are taken.
