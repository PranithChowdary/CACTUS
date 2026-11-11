# CACTUS: Combinatorial Action Masking for Tool-Augmented Sequential Reasoning

Tool selection as a sequential MDP, solved with PPO under strict autoregressive
action masking. Once a tool is selected it is removed from the action space, so
repeated selections are impossible by construction and every trajectory
terminates within `|A|` steps.

## Results

Held-out test split (51 queries, 15-tool vocabulary). CACTUS and Random are
mean ± s.d. over five training seeds; deterministic methods are a single pass.

| Agent | SEM ↑ | F1 ↑ | Trajectory length ↓ |
|---|---|---|---|
| Uniform Random | 0.000 ± 0.000 | 0.183 ± 0.021 | 8.25 ± 0.64 |
| Heuristic rules | 0.000 | 0.105 | 1.16 |
| Dense Retrieval (calibrated) | 0.137 | 0.565 | 2.80 |
| LLaMA-3.1-8B-Instruct (zero-shot) | 0.255 | 0.744 | — |
| **CACTUS (ours)** | **0.573 ± 0.067** | **0.806 ± 0.031** | 2.79 ± 0.08 |

The headline finding is the *split between metrics*: CACTUS more than doubles
the 8B LLM's exact-match rate (+0.318, 95% CI [0.184, 0.451], Wilcoxon
p = 7.0e-5), but its F1 advantage over the same model is **not** statistically
significant (+0.062, p = 0.059). The LLM emits a strict superset of the required
tools on 64.7% of queries — earning partial credit while rarely producing the
exact set a pipeline can execute.

All numbers are reproduced from `results/`; the SLURM logs of the run that
produced them are in `logs/final_run/`.

## Layout

```
env/          Gymnasium environment (MDP, action masking) + config
models/       MaskablePPO agent and the baseline agents
experiments/  train_ppo.py, evaluate_all.py, evaluate_llm_baseline.py
scripts/      SLURM job scripts + login-node setup + offline re-analysis
paper/        main.tex, references.bib, figure-generation script
results/      Aggregated metrics, per-example raw records, significance tests
datasets/     train/val/test splits and tool vocabulary
```

## Reproducing

Requires a SLURM cluster with one GPU. Compute nodes without outbound internet
are supported — all network-dependent setup happens on the login node.

```bash
# 1. On the LOGIN node: install deps and vendor the sentence encoder locally
bash scripts/setup_login_node.sh

# 2. Train all five seeds (single job; a job array trips per-user QOS limits)
TRAIN_JOB=$(sbatch --parsable scripts/train_ppo_cluster.sh)

# 3. Evaluate once training finishes, and run the LLM baseline in parallel
sbatch --dependency=afterok:${TRAIN_JOB} scripts/evaluate_cluster.sh
sbatch scripts/run_llm_baseline.sh
```

To re-analyse finished runs without re-running any model (numpy + scipy only):

```bash
python scripts/recompute_stats.py   # rebuilds final_comparison.json + significance.json
python paper/make_figures.py        # rebuilds paper/figures/*.pdf from results/
```

## Evaluation protocol

Three choices make the comparison falsifiable, and are worth preserving in any
extension of this work:

- **Five seeds, always.** Per-seed exact-match ranges from 0.490 to 0.667; a
  single-seed number misstates the method by up to ±0.09.
- **Deterministic, matched evaluation.** Every agent takes one ordered pass over
  all 51 test queries, visiting each exactly once, so comparisons are paired.
- **Baselines can stop.** Dense Retrieval gets a similarity-threshold stopping
  rule calibrated on the *validation* split. A baseline that cannot terminate is
  forced to over-predict and is not a fair comparison.

## Building the paper

```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Known limitations

Every query in this benchmark requires exactly two tools, so the target space is
the 105 two-element subsets of a 15-tool vocabulary — considerably milder than
the large-vocabulary regime that motivates the approach. See the Limitations
section of the paper.
