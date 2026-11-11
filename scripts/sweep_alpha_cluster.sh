#!/bin/bash
#SBATCH --job-name=cactus_sweep
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=62G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# The alpha sweep: does a sequential policy beat a memoryless classifier once
# the task genuinely requires gathering information by executing tools?
#
# Budget matters here. A 40k-timestep pilot produced a CONFIDENT FALSE NEGATIVE
# (PPO scored 0.180 at alpha=1, below the 0.5 chance ceiling, because it never
# learned to probe). At 150k with ent_coef=0.01 the same setup scored 0.552,
# above the analytic ceiling. Do not shrink --timesteps to save wall-clock: an
# undertrained agent here does not produce a weaker result, it produces the
# opposite result.
#
# Before reading any output, confirm the agent is trainable at all:
#   python experiments/diagnose_agent.py --timesteps 150000
# If the alpha=0 capacity check is below ~0.9, nothing else is interpretable.
#
# Submit:
#   sbatch scripts/sweep_alpha_cluster.sh

set -euo pipefail

source ~/miniconda3/bin/activate
cd /users/staff/pranith/CACTUS/

mkdir -p logs results

if [ ! -d "cactus_env" ]; then
    echo "ERROR: cactus_env not found. Run scripts/setup_login_node.sh on the LOGIN node first."
    exit 1
fi
conda activate ./cactus_env
pip install -q -r requirements.txt || echo "WARNING: pip install failed (no internet on compute node?) -- continuing."

export PYTHONPATH=.
export HF_HOME=/janaki/common/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "========================================="
echo "0. Sanity: is the agent trainable at all?"
echo "========================================="
srun python experiments/diagnose_agent.py --timesteps 200000

echo "========================================="
echo "1. Alpha sweep (5 alphas x 5 seeds x 2 agents)"
echo "========================================="
srun python experiments/sweep_alpha.py \
    --alphas 0.0 0.25 0.5 0.75 1.0 \
    --seeds 5 \
    --timesteps 200000 \
    --ent_coef 0.01 \
    --out results/alpha_sweep.json

echo "========================================="
echo "2. Figure"
echo "========================================="
srun python paper/make_sweep_figure.py

echo "Done. See results/alpha_sweep.json and paper/figures/alpha_sweep.pdf"
