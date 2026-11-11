#!/bin/bash
#SBATCH --job-name=bank61_70b_topup
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Recovers the ONE config lost when job 65133 hit its 8 h wall:
# llama-3.1-70b, pmi scoring, alpha=1.0. The three others (sum at both alphas,
# pmi at alpha=0) completed and are in results/llm_probe_bank61_llama-3.1-70b.json.
# Only the PMI ablation is affected; the headline summed-scoring results are intact.
#
# Why 65133 ran out of time even though scoring needs only ~4.8 h: the 140 GB
# model was loaded TWICE -- once by verify_llm_policy.py and once by the probe --
# and the second load crawled at ~40 s/iteration off the shared filesystem. Two
# fixes here: the verification step is skipped (this model has already passed
# 6/6 critical checks on two separate runs, and its tokenizer has not changed),
# and --time is raised to 24 h, which this QoS accepts.
#
# !! DO NOT point --out at the existing result file. llm_probe_constrained.py
# !! starts from an empty list and rewrites the file, so reusing that name would
# !! replace three good configs with this single one. It writes a separate file
# !! and scripts/merge_topup.py folds it in.
#
#   sbatch scripts/run_llm_bank61_70b_topup.sh

set -uo pipefail

source ~/miniconda3/bin/activate
cd /users/staff/pranith/CACTUS/
conda activate ./cactus_env

export PYTHONPATH=.
export HF_HOME=/janaki/common/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p logs results

srun python -u experiments/llm_probe_constrained.py \
    --model meta-llama/Llama-3.1-70B-Instruct --model_tag llama-3.1-70b \
    --episodes 61 --alphas 1.0 --scorings pmi \
    --batch_size 8 \
    --out results/llm_probe_bank61_topup.json

echo ""
echo "Done. Then, locally after fetching:"
echo "  python scripts/merge_topup.py"
