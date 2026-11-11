#!/bin/bash
#SBATCH --job-name=alphashift
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=62G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# PHASE 2' -- the headline experiment.
#
# Train agents where gathering is (almost) always needed, then evaluate them
# where it is not. If they learned "always gather" rather than "gather when
# needed", they will call the probe tool on static episodes, where the probe is
# deliberately absent from the gold set and calling it is pure waste.
#
# Two training alphas carry the claim:
#   alpha_train = 1.00  every episode conditional. The agent has NEVER seen a
#                       static query prototype, so elevated probing at alpha=0
#                       is confounded with out-of-distribution INPUT.
#   alpha_train = 0.90  static episodes are 10% of training -- in-distribution,
#                       just rare. If the effect survives here the confound is
#                       ruled out. THIS is the number to report.
#
# BUDGET IS LOAD-BEARING. A 40k-timestep pilot of the alpha sweep produced a
# confident FALSE NEGATIVE: PPO scored 0.180 at alpha=1, below the 0.5 chance
# ceiling, because it never learned to probe at all. At 150k with ent_coef=0.01
# the same setup scored 0.552. Do NOT cut --timesteps to save wall-clock; an
# undertrained agent here does not give a weaker result, it gives the opposite
# result.
#
# TWO gates catch this, and both print before any shift number -- read them first:
#   1. CAPACITY      alpha=0-trained agent must reach SEM >= 0.80 at alpha=0.
#   2. OWN-ALPHA     EVERY agent must reach SEM >= 0.50 at the alpha it was
#                    trained on. Without this an agent that simply failed to
#                    learn reports static_probe_rate ~ 0 and the script
#                    announces "KILL CRITERION MET: no over-gathering" when the
#                    truth is "no policy". A 25k smoke run did exactly that --
#                    the alpha=0.9 agent scored 0.020 at alpha=1.
#
# If gate 2 fails at 200k, RAISE --timesteps and resubmit; do not read the
# over-gathering numbers. A prior alpha-sweep reached only 0.552 at alpha=1
# with 150k steps, so the alpha=0.9 agent is the one most likely to be short.
#
# 4 train alphas x 5 seeds = 20 agents at 200k steps. Results are written after
# every single agent, so a timeout loses at most one agent's work.
#
#   sbatch scripts/run_alpha_shift.sh

set -uo pipefail

source ~/miniconda3/bin/activate
cd /users/staff/pranith/CACTUS/
conda activate ./cactus_env

export PYTHONPATH=.
export HF_HOME=/janaki/common/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p logs results

echo "========================================="
echo "0. Is the agent trainable at all?"
echo "   (if the alpha=0 capacity check fails,"
echo "    nothing below is interpretable)"
echo "========================================="
srun python -u experiments/diagnose_agent.py --timesteps 200000

echo ""
echo "========================================="
echo "1. Alpha-shift: 4 train alphas x 5 seeds"
echo "========================================="
srun python -u experiments/alpha_shift.py \
    --train_alphas 0.0 0.5 0.9 1.0 \
    --eval_alphas 0.0 0.25 0.5 0.75 1.0 \
    --seeds 5 \
    --timesteps 200000 \
    --ent_coef 0.01 \
    --n_eval 400 \
    --out results/alpha_shift.json

echo ""
echo "Done. Pull back with:  bash scripts/fetch_results.sh"
