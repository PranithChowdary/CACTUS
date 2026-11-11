#!/bin/bash
#SBATCH --job-name=llm_probe_test
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
# KILL TEST #1: does prompted LLaMA-3.1-8B already probe before committing on
# execution-dependent tasks? Gates the whole RL-vs-SFT information-gathering
# program -- if the answer is yes, there is nothing to teach and we stop.
#
#   sbatch scripts/run_llm_probe_test.sh

set -euo pipefail

source ~/miniconda3/bin/activate
cd /users/staff/pranith/CACTUS/
conda activate ./cactus_env

export PYTHONPATH=.
export HF_HOME=/janaki/common/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u experiments/llm_probe_test.py --episodes 240 --batch_size 32

echo "See results/llm_probe_test.json"
