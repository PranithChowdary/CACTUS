#!/bin/bash
#SBATCH --job-name=llm_constr
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=62G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Probe test with a CONSTRAINED LLM policy: the model scores every legal action
# (remaining tools + DONE) and the argmax is taken, so it acts over the same
# masked action set as the PPO agent. Replaces the free-generation harness,
# which produced DONE in 0/240 turns and could not measure stopping at all.
#
# Reports PMI (primary), sum and mean scorings: on a toy prompt the scorer
# choice FLIPPED the answer, so it is reported rather than assumed.
# the low exact-match that aggregates cannot tell apart:
#   (a) never emits DONE -> runs to max_steps -> exact match impossible
#   (b) parse_tool mis-resolves outputs
#   (c) genuine over-prediction (a real finding)
#
# alpha=0.0 is the diagnostic case: a static task a logistic probe solves
# perfectly, where the model scored 0.14. If it cannot do THAT, nothing at
# alpha=1 is interpretable.
#
#   sbatch scripts/run_llm_constrectories.sh
#   tail -f logs/llm_constr_<jobid>.out

set -euo pipefail

source ~/miniconda3/bin/activate
cd /users/staff/pranith/CACTUS/
conda activate ./cactus_env

export PYTHONPATH=.
export HF_HOME=/janaki/common/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Verify the scoring maths first; a failure here invalidates everything after.
# Verified on the REAL model, not a stand-in: gpt2 is not in the
# cluster's offline HF cache, and checking the maths on the model
# actually used is stronger evidence anyway.
python -u experiments/verify_llm_policy.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct || { echo 'SCORING VERIFICATION FAILED -- aborting'; exit 1; }

python -u experiments/llm_probe_constrained.py --episodes 48 --alphas 0.0 1.0
