#!/bin/bash
#SBATCH --job-name=dump_traj
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=62G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Prints the model's actual turn-by-turn outputs, to separate three causes of
# the low exact-match that aggregates cannot tell apart:
#   (a) never emits DONE -> runs to max_steps -> exact match impossible
#   (b) parse_tool mis-resolves outputs
#   (c) genuine over-prediction (a real finding)
#
# alpha=0.0 is the diagnostic case: a static task a logistic probe solves
# perfectly, where the model scored 0.14. If it cannot do THAT, nothing at
# alpha=1 is interpretable.
#
#   sbatch scripts/run_dump_trajectories.sh
#   tail -f logs/dump_traj_<jobid>.out

set -euo pipefail

source ~/miniconda3/bin/activate
cd /users/staff/pranith/CACTUS/
conda activate ./cactus_env

export PYTHONPATH=.
export HF_HOME=/janaki/common/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "------------ alpha = 0.0  (static; classifier ceiling is 1.000) ----------------"
srun python -u experiments/dump_trajectories.py --alpha 0.0 --episodes 12 \
    --out results/trajectories_alpha0.json

echo
echo "---------------- alpha = 0.0, FEW-SHOT ----------------"
python -u experiments/dump_trajectories.py --alpha 0.0 --episodes 12 --few_shot \
    --out results/trajectories_alpha0_fewshot.json

echo
echo "------------ alpha = 1.0  (every episode needs the probe) ------------------"
srun python -u experiments/dump_trajectories.py --alpha 1.0 --episodes 12 \
    --out results/trajectories_alpha1.json

echo
echo "---------------- alpha = 1.0, FEW-SHOT ----------------"
python -u experiments/dump_trajectories.py --alpha 1.0 --episodes 12 --few_shot \
    --out results/trajectories_alpha1_fewshot.json
