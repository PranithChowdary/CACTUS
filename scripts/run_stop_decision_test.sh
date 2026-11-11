#!/bin/bash
#SBATCH --job-name=stop_dec
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=62G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Separates "can the model RECOGNISE completion?" from "will it EMIT a stop?".
# the low exact-match that aggregates cannot tell apart:
#   (a) never emits DONE -> runs to max_steps -> exact match impossible
#   (b) parse_tool mis-resolves outputs
#   (c) genuine over-prediction (a real finding)
#
# alpha=0.0 is the diagnostic case: a static task a logistic probe solves
# perfectly, where the model scored 0.14. If it cannot do THAT, nothing at
# alpha=1 is interpretable.
#
#   sbatch scripts/run_stop_decectories.sh
#   tail -f logs/stop_dec_<jobid>.out

set -euo pipefail

source ~/miniconda3/bin/activate
cd /users/staff/pranith/CACTUS/
conda activate ./cactus_env

export PYTHONPATH=.
export HF_HOME=/janaki/common/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u experiments/stop_decision_test.py
