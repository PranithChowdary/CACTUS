#!/bin/bash
#SBATCH --job-name=nlshift
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=62G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# PHASE 2' ON THE NATURAL-LANGUAGE ENVIRONMENT -- the run that makes the
# headline contrast same-environment.
#
# Until now the LLM results (NLConditionalEnv, natural language, 36 tools) and
# the RL results (ConditionalToolEnv, synthetic Gaussian prototypes, 20 tools)
# came from DIFFERENT tasks. "Instruct LLMs cannot terminate; RL learns to" was
# therefore comparing a text policy on one environment against a PPO policy on
# another -- a training-signal difference, not a capability difference. This job
# puts PPO on the identical 61-scenario NL task the LLM plays: same tools, same
# gold sets, same masking, same episode budget. Only the policy class differs.
#
# WHAT THIS RUN DECIDES
# The synthetic env MET its kill criterion: at the confound-controlled
# alpha_train=0.90 the static probe rate was 0.005 vs 0.000 matched, so there
# was no over-gathering. A single-seed NL pilot at alpha_train=1.00 instead gave
# static probe rate 0.817 vs 0.000 -- a large effect. But alpha_train=1.00 is
# the arm where the agent has NEVER seen a static query, so elevated probing
# there is confounded with out-of-distribution input. alpha_train=0.90, where
# static queries are in-distribution but rare, is the arm that decides it.
# Read the 0.90 row, not the 1.00 row.
#
# --net_arch 256 256 IS REQUIRED, NOT A TUNING PREFERENCE
# The NL observation is ~806-d (384 query + 384 probe-result + 37 selection + 1
# flag) and 122 query->gold mappings must be representable. With SB3's default
# 2x64 trunk, SEM at alpha=0 is 0.583 at 150k and 0.683 at 400k -- BELOW the
# 0.705 scored by a plain logistic regression on the same embeddings. That
# would have reproduced "a linear probe beats the RL agent" as an artifact of
# policy capacity rather than as a finding. At 256x256 it is 0.933 at 150k.
# Omitting this flag silently inverts the result.
#
# READ THE TWO GATES FIRST, BEFORE ANY PROBE RATE:
#   1. CAPACITY   alpha=0-trained agent must reach SEM >= 0.80 at alpha=0.
#   2. OWN-ALPHA  every agent must reach SEM >= 0.50 at its own training alpha.
# Single-seed variance is about 0.12 here, which is why this runs 5 seeds; judge
# the gates on the means, not on one seed.
#
# 4 train alphas x 5 seeds = 20 agents at 300k steps, ~4-5 min each on CPU, so
# ~2h against the 8h limit. Results are written after every agent.
#
#   sbatch scripts/run_nl_alpha_shift.sh

set -uo pipefail

source ~/miniconda3/bin/activate
cd /users/staff/pranith/CACTUS/
conda activate ./cactus_env

export PYTHONPATH=.
export HF_HOME=/janaki/common/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p logs results data

echo "========================================="
echo "0. Scenario bank gates"
echo "   (bank must be able to carry the"
echo "    comparison at all)"
echo "========================================="
srun python -u experiments/check_scenario_bank.py
if [ $? -ne 0 ]; then
    echo "!! BANK CHECK FAILED -- not training. Fix env/nl_scenarios.py first."
    exit 1
fi

echo ""
echo "========================================="
echo "1. Build the embedding cache up front"
echo "   (so a missing model or an offline-mode"
echo "    problem fails here, not 20 agents in)"
echo "========================================="
srun python -u -c "from env.nl_gym_env import load_embeddings; load_embeddings(verbose=True)"

echo ""
echo "========================================="
echo "2. Alpha-shift on the NL environment"
echo "========================================="
srun python -u experiments/alpha_shift.py \
    --env nl \
    --net_arch 256 256 \
    --train_alphas 0.0 0.5 0.9 1.0 \
    --eval_alphas 0.0 0.25 0.5 0.75 1.0 \
    --seeds 5 \
    --timesteps 300000 \
    --ent_coef 0.01 \
    --n_eval 400 \
    --out results/nl_alpha_shift.json

echo ""
echo "Done. Pull back with:  bash scripts/fetch_results.sh"
