# Frozen result set

**Frozen 2026-10-07; LLM side re-measured at n=61 on 2026-11-11.** No new hypotheses. One job outstanding (below); everything
else is final. Every number here is machine-read from `results/`, with the job
log that produced it named for provenance.

## The claim

> Knowing which tools a task needs and knowing when to stop calling them are
> separable capabilities. LLaMA checkpoints identify the right tools at every
> scale tested and never terminate; masked PPO on the identical task learns
> both. Termination moves by 75x under manipulations that move tool recall by
> 1.15x — and we could not pin which manipulation is responsible.

Three supporting results, **two refuted hypotheses** (over-gathering, §3; our own
post-training attribution, §1b), and one crossover we explicitly decline to call
a boundary condition (§4).

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

## 1a. The stop action is not an interface artifact

`results/native_stop_*.json` · log `nativestop_67879` · α=0, each model's native
end-of-turn token scored as an extra stop candidate at **every** step

| model | steps | compl. states | P(DONE\|c) | P(DONE\|¬c) | elevation | P(native\|c) |
|---|---|---|---|---|---|---|
| llama-3.1-8b | 305 | 58 | 0.00200 | 0.00188 | **1.06×** | 2.4e-09 |
| qwen2.5-7b | 226 | 40 | 0.57531 | 0.19660 | **2.93×** | 9.8e-11 |

- **Objection killed.** The native channel carries 6–10 orders of magnitude *less*
  mass than DONE (8.5e5× less for LLaMA, 5.9e9× for Qwen). Neither model signals
  completion through the channel it was trained on, so the missing stop is a
  property of the models, not of our action set.
- **Claim sharpened.** LLaMA's elevation of 1.06× means its stop probability is
  statistically **independent of task completion** — a stronger statement than
  "low probability". This is now the form the main result is stated in.
- **Our own speculation corrected.** We had flagged that Qwen's rank-2.1 stop
  might be pure surface-form prior, making 0.5753 an upper bound. A pure prior
  predicts 1.0× elevation; Qwen is at 2.93×, so its stopping *is* conditioned on
  task state. The prior is still real (0.1966 before completion) and is the
  mechanism behind the 34/61 late stops.

## 1b. Base vs instruct: the post-training attribution REFUTED

`results/llm_probe_basevsinst_*.json` (job `basevsinst_67978`, template on) and
`results/llm_probe_notmpl_*.json` (job `tmplabl_68547`, template off).
Summed scoring, 61 scenarios, one episode each, mangala.

**alpha = 0, the full 2x2:**

| checkpoint | template | recall@5 | stops | P(DONE\|c) | SEM |
|---|---|---|---|---|---|
| qwen2.5-7b-base | on | 0.967 | 0/61 | 0.0076 | 0.000 |
| qwen2.5-7b-base | off | 0.869 | **25/61** | **0.1912** | 0.131 |
| qwen2.5-7b-inst | on | 0.951 | 60/61 | **0.5753** | 0.377 |
| qwen2.5-7b-inst | off | 1.000 | 45/61 | **0.1265** | 0.098 |

alpha = 1: base on 0/61 @ 0.0174 · base off 9/61 @ 0.0253 · inst on 54/61 @
0.5782 · inst off 32/61 @ 0.0907.

- **The 75x is a prompt-formatting artifact.** Removing the chat template moves
  base UP 25x (0.0076 -> 0.1912) and instruct DOWN 4.5x (0.5753 -> 0.1265). The
  instruct/base ratio goes 75.4x -> **0.66x**: it reverses, base ends up higher.
- **Neither cell is a clean control.** With the template the comparison is unfair
  to base; without it, unfair to instruct (recall@5 1.000 but SEM 0.098 — ranks
  tools fine, performs badly in an unfamiliar format). The format a checkpoint
  expects *is* a product of its post-training, so released checkpoints cannot
  separate the two. **Claim withdrawn from the paper.**
- **What still survives:** instruct's stop RATE exceeds base's under both formats
  (60/61 vs 0/61; 45/61 vs 25/61). The direction holds on the argmax measure; it
  is only the probability mass that reverses.
- **What is strengthened:** across all four cells recall@5 spans 0.869–1.000
  (1.15x) while P(DONE|c) spans 0.0076–0.5753 (75x) and stops span 0/61–60/61.
  Two manipulations that barely touch relevance reorganise termination
  completely — the dissociation is robust to post-training AND prompt format.
- Settling this needs fine-tuning one base checkpoint under controlled recipes,
  evaluated under one fixed format. Not run.

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

## 4. Our agent clears the memoryless ceiling only above α ≈ 0.75

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

**This is NOT a boundary condition on sequential policies and the paper says so
explicitly (§6.8).** An optimal sequential policy scores 1.000 at every α and so
beats `(1-α)+α/2` for every α > 0. Ours reaches 0.855 at α=0. The crossover is
where our agent's optimisation deficit becomes smaller than the ceiling's
decline — a fact about this PPO setup at this budget, not about the policy
class. A better-optimised agent crosses earlier. Do not restate it as a bound.

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

## Outstanding

**`llama-3.1-8b-base` — gated on HuggingFace per account**, requested 2026-11-21,
not granted. Would show whether the template reversal in §1b reproduces in a
second family. `python scripts/fetch_base_models.py --check` re-probes; the job
script preflights and skips cleanly, so just resubmit once granted.

**The experiment that would actually settle what installs termination:**
fine-tune one base checkpoint under recipes we control and evaluate every variant
under one fixed prompt format. Released checkpoints cannot do it — see §1b. Not
run, and flagged as open in the paper.

Everything else is complete: both sides use the same 61-scenario bank, all four
scoring×α configs, both RL environments at 20 agents × 5 seeds, the chat-template
2×2, and a bibliography verified against primary sources.

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
python scripts/compare_models.py --pattern 'results/llm_probe_bank61_*.json'
python scripts/compare_models.py --pattern 'results/llm_probe_basevsinst_*.json'
sbatch  scripts/run_alpha_shift.sh                         # RL, synthetic
sbatch  scripts/run_nl_alpha_shift.sh                      # RL, natural language
```

`--net_arch 256 256` is **required** on `--env nl`. With SB3's default 2×64
trunk, SEM at α=0 is 0.583 at 150k and 0.683 at 400k — *below* the 0.705 a
logistic regression gets on the same embeddings. Omitting the flag reproduces
"a linear probe beats the RL agent" as an artifact of policy capacity rather
than as a finding.
