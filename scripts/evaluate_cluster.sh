#!/bin/bash
#SBATCH --job-name=cactus_eval
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=62G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Runs the full deterministic evaluation (Random x5 repeats, Heuristic,
# DenseRetrieval calibrated-on-val, and every trained PPO seed found in
# logs/checkpoints/ppo_metatool_seed*.zip), producing:
#   results/final_comparison.json   (mean +/- std per agent)
#   results/raw/<agent>.json        (per-example records)
#   results/significance.json       (paired bootstrap + Wilcoxon vs best baseline)
#
# Does NOT run the LLaMA baseline -- that's scripts/run_llm_baseline.sh,
# which has no dependency on PPO checkpoints, so submit it separately
# (in parallel with training is fine, it saves wall-clock time).
#
# Run after all 5 seeds from train_ppo_cluster.sh have finished:
#   sbatch --dependency=afterok:<train_job_id> scripts/evaluate_cluster.sh
# or standalone once you already have checkpoints:
#   sbatch scripts/evaluate_cluster.sh

set -euo pipefail

source ~/miniconda3/bin/activate
cd /users/staff/pranith/CACTUS/

if [ ! -d "cactus_env" ]; then
    echo "ERROR: cactus_env not found. Run scripts/train_ppo_cluster.sh first (it creates the env), or build it manually."
    exit 1
fi
conda activate ./cactus_env
pip install -q -r requirements.txt || echo "WARNING: pip install failed (no internet on compute node?) -- continuing with the existing env. Run scripts/setup_login_node.sh on the LOGIN node if deps are missing."

mkdir -p logs results
export PYTHONPATH=.
# Shared HF cache -- avoids the sentence-transformers embedding model trying
# to hit the internet from the compute node (see logs/cactus_train_32445.err
# for what that looks like when it fails: repo.anaconda.com unreachable).
export HF_HOME=/janaki/common/huggingface
# Never attempt a network call: the compute node has no outbound internet,
# so an un-warmed cache should fail fast with a clear error instead of
# retrying against an unreachable huggingface.co. Warm the cache first on
# the LOGIN node with: bash scripts/setup_login_node.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "========================================="
echo "Evaluating all agents (PPO seeds + baselines)"
echo "========================================="
srun python experiments/evaluate_all.py

echo "========================================="
echo "Evaluation complete. See results/final_comparison.json, results/significance.json"
echo "========================================="
