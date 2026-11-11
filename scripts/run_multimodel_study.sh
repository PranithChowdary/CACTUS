#!/bin/bash
#SBATCH --job-name=multimodel
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=62G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Does the termination failure generalize beyond LLaMA-3.1-8B?
#
# On LLaMA-3.1-8B the model ranked correct tools first in 83% of static tasks
# and probed before committing in 92% of conditional ones, yet chose DONE in
# 0/288 decisions (rank 35 of 35 at every step). "One 8B model cannot stop" is
# a much weaker claim than "8B-class instruct models cannot stop", so this runs
# the identical protocol across FAMILIES at fixed size and across SCALE within
# a family.
#
#   family (7-8B) : Qwen2.5-7B, Mistral-7B-v0.3  vs the LLaMA-3.1-8B we have
#   scale (LLaMA) : 3.2-1B -> 3.1-8B (done) -> 3.1-70B (separate job, needs >1 GPU)
#
# Each model is verified before it is trusted: verify_llm_policy.py re-runs the
# scoring-maths checks on that model's own tokenizer, since chat templates and
# BPE differ per family and a silent tokenization mismatch would corrupt every
# score. A critical failure skips that model rather than aborting the study.
#
# ~40 min per model (2 scorings x 2 alpha x 48 episodes). 1B is much faster.
#
#   sbatch scripts/run_multimodel_study.sh

set -uo pipefail   # NOT -e: one bad model must not kill the whole study

source ~/miniconda3/bin/activate
cd /users/staff/pranith/CACTUS/
conda activate ./cactus_env

export PYTHONPATH=.
export HF_HOME=/janaki/common/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# tag                model id
MODELS=(
  "qwen2.5-7b|Qwen/Qwen2.5-7B-Instruct"
  "mistral-7b|mistralai/Mistral-7B-Instruct-v0.3"
  "llama-3.2-1b|meta-llama/Llama-3.2-1B-Instruct"
)

for entry in "${MODELS[@]}"; do
    TAG="${entry%%|*}"
    MODEL="${entry##*|}"
    echo ""
    echo "-------------------------------------------------------------"
    echo "# $TAG   ($MODEL)"
    echo "-------------------------------------------------------------"

    echo "--- verifying scoring maths on this model's tokenizer ---"
    srun python -u experiments/verify_llm_policy.py --model "$MODEL"
    if [ $? -ne 0 ]; then
        echo "!! CRITICAL verification failure for $TAG -- SKIPPING this model."
        echo "!! (its scores would be invalid; the rest of the study continues)"
        continue
    fi

    echo "--- running probe study ---"
    srun python -u experiments/llm_probe_constrained.py \
        --model "$MODEL" --model_tag "$TAG" \
        --episodes 48 --alphas 0.0 1.0
done

echo ""
echo "-------------------------------------------------------------"
echo "# cross-model comparison"
echo "-------------------------------------------------------------"
srun python -u scripts/compare_models.py
