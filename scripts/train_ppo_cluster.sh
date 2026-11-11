#!/bin/bash
#SBATCH --job-name=cactus_train
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=62G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Trains all 5 MaskablePPO seeds (env/config.py SEEDS) SEQUENTIALLY inside
# ONE job. A SLURM array (one job record per seed) hits this account's
# QOSMaxSubmitJobPerUser limit -- looping in-process avoids that entirely at
# the cost of some wall-clock time, which is cheap here (each seed trains in
# minutes given the dataset size). A single-seed PPO number is an anecdote --
# this is what the paper's mean +/- std across seeds should come from.
#
# Submit with:
#   sbatch scripts/train_ppo_cluster.sh
# Then, once it finishes, evaluate everything with:
#   sbatch --dependency=afterok:<this_job_id> scripts/evaluate_cluster.sh

set -euo pipefail

source ~/miniconda3/bin/activate
cd /users/staff/pranith/CACTUS/

mkdir -p logs results logs/checkpoints

if [ ! -d "cactus_env" ]; then
    echo "Creating new local conda environment 'cactus_env'..."
    conda create --prefix ./cactus_env python=3.12 -y
fi
conda activate ./cactus_env

# Always (re-)install requirements, even if the env already existed from a
# previous run -- otherwise a pre-existing env silently misses new deps
# (sb3-contrib, scipy) added when the code was rewritten to use masking.
pip install -q -r requirements.txt || echo "WARNING: pip install failed (no internet on compute node?) -- continuing with the existing env. Run scripts/setup_login_node.sh on the LOGIN node if deps are missing."

export PYTHONPATH=.
# Shared HF cache so the sentence-transformers embedding model doesn't try
# to reach huggingface.co from the compute node -- the compute nodes on this
# partition appear to lack outbound internet (see the anaconda.org
# connection failures in logs/cactus_train_32445.err from an earlier run).
# Pre-warm this cache from a LOGIN node before submitting (see chat).
export HF_HOME=/janaki/common/huggingface
# Never attempt a network call: the compute node has no outbound internet,
# so an un-warmed cache should fail fast with a clear error instead of
# retrying against an unreachable huggingface.co. Warm the cache first on
# the LOGIN node with: bash scripts/setup_login_node.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Must match env/config.py SEEDS exactly.
SEEDS=(42 43 44 45 46)

for SEED in "${SEEDS[@]}"; do
    echo "========================================="
    echo "Training MaskablePPO, seed=${SEED}"
    echo "========================================="
    srun python experiments/train_ppo.py --seed "${SEED}"
    echo "Seed ${SEED} done. Checkpoint: logs/checkpoints/ppo_metatool_seed${SEED}.zip"
done

echo "========================================="
echo "All 5 seeds trained."
echo "========================================="
