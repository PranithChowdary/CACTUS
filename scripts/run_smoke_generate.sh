#!/bin/bash
#SBATCH --job-name=smoke_gen
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=62G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Isolates the LLM generation path in four escalating stages. Job 50364 sat on
# a GPU for three hours without completing a single batched generate() call;
# this finds out why in under a minute of compute.
#
# Deliberately short (--time=00:20:00): if generation is healthy this finishes
# almost immediately, and if it hangs we want it killed quickly, not in 4h.
#
#   sbatch scripts/run_smoke_generate.sh
#   tail -f logs/smoke_gen_<jobid>.out

set -euo pipefail

source ~/miniconda3/bin/activate
cd /users/staff/pranith/CACTUS/
conda activate ./cactus_env

export PYTHONPATH=.
export HF_HOME=/janaki/common/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Unbuffered, so a kill cannot swallow the output that localizes the stall.
python -u experiments/smoke_generate.py
