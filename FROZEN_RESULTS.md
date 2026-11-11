# Frozen result set

**Frozen 2026-10-07; LLM side re-measured at n=61 on 2026-11-11.** No new hypotheses. One job outstanding (below); everything
else is final. Every number here is machine-read from `results/`, with the job
log that produced it named for provenance.

## The claim

> Instruct-tuned LLMs fail to terminate at every scale tested, while masked PPO
> on the identical task learns both conditional information-gathering and
> termination — but sequential policies beat a memoryless classifier only when
> roughly 70% or more of tasks genuinely require gathering.

Three supporting results, one refuted hypothesis, and the boundary condition.

## 1. LLMs rank tools near-perfectly and cannot stop

`results/llm_probe_bank61_*.json` · logs `bank61_65126`, `bank61_70b_65133`
· **61 scenarios, one episode each, so n = 61 with no repeat inflation**

| model | recall@5 | rank SEM@k | probe α=1 | stops (scenarios) | DONE rank /35 | P(DONE \| complete) |
|---|---|---|---|---|---|---|
| llama-3.2-1b | 0.033 | 0.000 | 0.000 | 0/61 | 26.4 | — |
| llama-3.1-8b | **1.000** | 0.951 | 0.787 | **0/61** | 30.1 | **0.0020** |
| llama-3.1-70b | **1.000** | 0.541 | 0.836 | **5/61** | 21.9 | **0.0107** |
| qwen2.5-7b | 0.951 | 0.656 | 0.721 | 60/61 | 2.1 | 0.5753 |

- 8B and 70B reach **perfect tool recall** (gold ⊆ first 5 picks, 61/61) while
  holding 0.2–1.1% of probability mass on DONE when the gold set is complete.
  The failure is termination alone, not tool knowledge.
- Across a **70× parameter range** termination does not arrive. 70B's
  P(DONE|complete) is 5× the 8B value and still ≈1%; it stops in 5 of 61
  scenarios. Qwen-7B holds 57.5% at one tenth the size: the determining factor
  is post-training, not capacity.
- 70B's lower SEM@k is **not** a capability regression — it interleaves a
  plausible same-domain tool between the two gold tools. Coverage by cutoff:
  m=2 0.541, m=3 0.803, m=5 1.000.
- Qwen stops but is **miscalibrated**: 24/61 exactly, 34/61 late, 1/61 never.
  Even a model that can stop over-gathers in 56% of scenarios.
- Scoring verified per model (6/6 critical checks) before use.

**Confirmed against the earlier 12-scenario bank.** Every direction is preserved
at 5× the sample size, and 8B's ranking improves (SEM@k 0.833 → 0.951). The old
files are kept under `results/llm_probe_constrained_*.json` as provenance.

**Dataset complete.** The 70B job initially hit its 8-hour wall during the fourth
config (the 140 GB model was loaded twice — once by the verifier, once by the
probe — with one load crawling off the shared filesystem; scoring itself was only
3.6 h). Recovered by `scripts/run_llm_bank61_70b_topup.sh` and merged with
`scripts/merge_topup.py`, which verified the three original configs byte-identical
before adding the fourth. All four models now have all four configs at n=61.

**PMI ablation.** Under domain-conditional PMI the stop rate is **0.000 in all 8
model×α configs** — no model ever terminates — while summed scoring ranks tools
far better (SEM@k 0.951 vs 0.049 on 8B). The scorer choice can invert the central
observation, which is why all three are logged per step and the primary was
selected on ranking evidence rather than a priori.

## 2. RL terminates and gathers conditionally — on the same task

`results/nl_alpha_shift.json` · log `nlshift_59325` · 20 agents (4 α × 5 seeds),
300k steps, `net_arch=[256,256]`, n_eval=400

Gates (read first, both pass): capacity 0.855 ± 0.092; own-distribution
0.668–0.855 across all four training α.

| train_α | probe \| static | probe \| conditional | gap | done_rate |
|---|---|---|---|---|
| 0.00 | 0.009 | 0.175 | 0.166 | 0.992 |
| 0.50 | 0.008 | 0.910 | 0.902 | 1.000 |
| **0.90** | **0.027** | **0.989** | **0.963** | 1.000 |
| 1.00 | 0.806 | 0.991 | 0.184 | 0.955 |

`done_rate` is 0.955–1.000 for every agent, on the **same 61-scenario NL
environment** where 8B stops in 0/12 scenarios and 70B in 1/12. Earlier this
contrast was measured across two different environments and was therefore
confounded; `env/nl_gym_env.py` removed that.

## 3. Over-gathering: hypothesis refuted on both environments

The pre-registered hypothesis (agents trained where gathering is always needed
learn *to always gather* rather than *when to gather*) **met its kill criterion**.

| environment | matched | α_train=0.90 | Δ |
|---|---|---|---|
| synthetic (`alpha_shift.json`, log `alphashift_59061`) | 0.000 | 0.005 | +0.005 |
| natural language (`nl_alpha_shift.json`) | 0.009 | 0.027 | **+0.018** |

α_train=1.00 shows a large effect (0.806 on NL, 0.156 synthetic) **but that arm
never sees a static query**, so its probing is confounded with
out-of-distribution input. The α_train=0.90 arm, where static queries are
in-distribution but rare, shows nothing. Reporting the 1.00 arm alone would have
claimed a +0.80 effect that does not exist.

The α_train=0.90 agent is near-invariant across evaluation α
(0.662 / 0.672 / 0.691 / 0.688 / 0.692 / 0.696) — the signature of a policy
conditioned on need rather than a fixed gathering habit.

## 4. The boundary: sequential policies only pay off at high α

Best agent vs. the analytic memoryless ceiling `(1-α) + α/2`:

| eval α | NL best PPO | ceiling | synthetic best PPO | ceiling |
|---|---|---|---|---|
| 0.50 | 0.691 | 0.750 | 0.642 | 0.750 |
| 0.75 | **0.688** | 0.625 | 0.563 | 0.625 |
| 0.90 | **0.692** | 0.550 | 0.607 | 0.550 |
| 1.00 | **0.727** | 0.500 | 0.670 | 0.500 |

α\* ≈ **0.6–0.75** on natural language, ≈ **0.85–0.9** synthetically. Below it a
logistic regression on the same frozen embeddings wins — consistent with the
MetaTool finding that began this work.

## Environment validity

`results/scenario_bank_check.json` · 61 scenarios, 36 tools + DONE

| check | fit | cv | gate | |
|---|---|---|---|---|
| conditionality predictable | 1.000 | 0.804 | ≥0.95 fit | PASS |
| scenario identity separable | 1.000 | 0.680 | ≥0.95 fit | PASS |
| static gold representable by policy class | 1.000 | — | ≥0.90 | PASS |
| latent bit leaks | 0.500 | 0.107 | ≤0.60 cv | PASS |
| conditionality, markers replaced (length-matched) | 1.000 | 0.876 | — | — |
| memoryless linear top-2 baseline (*not* a gate) | 0.705 | 0.000 | — | — |

The latent branch bit stays at chance, so the `(1-α) + α/2` ceiling is an
information-theoretic bound rather than an empirical artifact. The conditional
signal survives marker replacement at matched length (11.3 vs 11.6 mean words),
so it is carried by meaning — static queries name their actions, conditional
ones do not — rather than by one keyword.

## Outstanding: nothing

All experiments are complete. Both sides of the comparison use the same
61-scenario bank; all four models have all four scoring×α configs; both RL
environments have 20 agents each at 5 seeds. The remaining work is writing, not
computing — see `paper/termination_draft.tex`, whose open items are the
bibliography (only one entry is verified) and a venue decision.

## Deliberately not done (v2)

- `scripts/sweep_alpha_cluster.sh` — adds a blind-PPO control isolating the
  observation channel. Never submitted; the transfer tables largely subsume it.
- Mistral-7B as a third family — blocked on incomplete weights in the shared
  cache (`no pytorch_model.bin or model.safetensors`), a data problem not a code
  one.
- A real benchmark (τ-bench / AppWorld / TaskBench) — the main external-validity
  gap and the biggest reviewer target. Not blocking arXiv; blocking ICML.
- Scenario-level held-out split — `env/nl_scenarios.split()` exists but is not
  used. A memoryless model scores **0.000** on held-out scenarios, so the split
  makes tool identity unlearnable for both the classifier and PPO; only an LLM
  can transfer. That is a different paper.

## Reproduction

```
python experiments/check_scenario_bank.py                  # environment gates
python scripts/compare_models.py                           # LLM table (12-scenario)
sbatch  scripts/run_alpha_shift.sh                         # RL, synthetic
sbatch  scripts/run_nl_alpha_shift.sh                      # RL, natural language
```

`--net_arch 256 256` is **required** on `--env nl`. With SB3's default 2×64
trunk, SEM at α=0 is 0.583 at 150k and 0.683 at 400k — *below* the 0.705 a
logistic regression gets on the same embeddings. Omitting the flag reproduces
"a linear probe beats the RL agent" as an artifact of policy capacity rather
than as a finding.
