#!/bin/bash
#SBATCH --job-name=cactus_quickcheck
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
# Quick single-seed smoke run: trains ONE MaskablePPO seed with a short
# timestep budget and evaluates it, just to confirm the pipeline still works
# end to end after an edit -- NOT the real experiment. For the real,
# reportable result use scripts/train_ppo_cluster.sh (all 5 seeds, one job)
# followed by scripts/evaluate_cluster.sh.

set -euo pipefail

source ~/miniconda3/bin/activate
cd /users/staff/pranith/CACTUS/

mkdir -p logs results logs/checkpoints

if [ ! -d "cactus_env" ]; then
    echo "Creating new local conda environment 'cactus_env'..."
    conda create --prefix ./cactus_env python=3.12 -y
fi
conda activate ./cactus_env
pip install -q -r requirements.txt || echo "WARNING: pip install failed (no internet on compute node?) -- continuing with the existing env. Run scripts/setup_login_node.sh on the LOGIN node if deps are missing."

export PYTHONPATH=.
export HF_HOME=/janaki/common/huggingface
# Never attempt a network call: the compute node has no outbound internet,
# so an un-warmed cache should fail fast with a clear error instead of
# retrying against an unreachable huggingface.co. Warm the cache first on
# the LOGIN node with: bash scripts/setup_login_node.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "========================================="
echo "Quick-check: training 1 seed for 10k timesteps"
echo "========================================="
srun python experiments/train_ppo.py --seed 42 --timesteps 10000

echo "-----------------------------------------"
echo "Evaluating (Random, Heuristic, DenseRetrieval, PPO)..."
echo "-----------------------------------------"
srun python experiments/evaluate_all.py

echo "========================================="
echo "Quick-check complete. Results in results/ directory."
echo "This was a single under-trained seed -- do not report these numbers."
echo "========================================="
