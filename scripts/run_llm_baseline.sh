#!/bin/bash
#SBATCH --job-name=llm_eval
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

source ~/miniconda3/bin/activate
cd /users/staff/pranith/CACTUS/

conda activate ./cactus_env

# Point HuggingFace to the shared cache so it doesn't download anything!
# (evaluate_llm_baseline.py already passes local_files_only=True, so these
# are belt-and-braces -- they just keep the failure mode loud and immediate
# rather than five retries against an unreachable host.)
export HF_HOME=/janaki/common/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Evaluating LLaMA-3.1-8B-Instruct..."
srun python experiments/evaluate_llm_baseline.py
