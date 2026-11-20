#!/bin/bash
#SBATCH --job-name=basevsinst
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# JOB 2 OF 2 -- isolates post-training from everything it is currently
# confounded with.
#
# The paper claims termination "tracks post-training rather than scale" on the
# strength of ONE cross-family pair: LLaMA-3.1-8B stops in 0/61 scenarios,
# Qwen2.5-7B in 60/61. Those two models differ in pretraining corpus,
# tokenizer, architecture, chat template AND post-training recipe at the same
# time, so the attribution to post-training is an inference from a single
# contrast, not a controlled result. A reviewer will say so.
#
# Base vs Instruct WITHIN a family changes post-training and holds everything
# else fixed. That is the controlled version of the same comparison:
#   LLaMA-3.1-8B   base vs instruct
#   Qwen2.5-7B     base vs instruct
#
# Expected readings:
#   base ~ instruct on stopping  -> post-training is NOT what installs or
#                                   suppresses termination; the cross-family
#                                   difference is something else and the
#                                   paper's second claim must be withdrawn.
#   instruct differs from base, in opposite directions for the two families
#                                -> the post-training claim is supported
#                                   properly for the first time.
#
# NOTE ON BASE MODELS: a base model has no chat template. verify_llm_policy.py
# is run first for each and a critical failure SKIPS that model rather than
# producing scores that look valid and are not. If the base checkpoints are not
# in the offline cache this job will fail fast at the load -- download them on
# the LOGIN node first, which has internet:
#   python -c "from huggingface_hub import snapshot_download as d; \
#     d('meta-llama/Meta-Llama-3.1-8B'); d('Qwen/Qwen2.5-7B')"
#
# ~1.5 GPU-hours for four models at 61 episodes, one scoring rule, alpha 0 and 1.
#
#   sbatch scripts/run_base_vs_instruct.sh

set -uo pipefail

source ~/miniconda3/bin/activate
cd /users/staff/pranith/CACTUS/
conda activate ./cactus_env

export PYTHONPATH=.
export HF_HOME=/janaki/common/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

mkdir -p logs results

# tag                    model id                              chat template?
MODELS=(
  "llama-3.1-8b-base|meta-llama/Meta-Llama-3.1-8B|no"
  "llama-3.1-8b-inst|meta-llama/Meta-Llama-3.1-8B-Instruct|yes"
  "qwen2.5-7b-base|Qwen/Qwen2.5-7B|no"
  "qwen2.5-7b-inst|Qwen/Qwen2.5-7B-Instruct|yes"
)

for entry in "${MODELS[@]}"; do
    TAG="${entry%%|*}"; REST="${entry#*|}"; MODEL="${REST%%|*}"
    echo ""
    echo "-------------------------------------------------------------"
    echo "# $TAG   ($MODEL)"
    echo "-------------------------------------------------------------"

    echo "--- verifying scoring maths on this model's tokenizer ---"
    srun python -u experiments/verify_llm_policy.py --model "$MODEL"
    if [ $? -ne 0 ]; then
        echo "!! CRITICAL verification failure for $TAG -- SKIPPING."
        echo "!! Base models have no chat template; scores would be invalid."
        continue
    fi

    srun python -u experiments/llm_probe_constrained.py \
        --model "$MODEL" --model_tag "$TAG" \
        --episodes 61 --alphas 0.0 1.0 --scorings sum \
        --out results/llm_probe_basevsinst.json
done

echo ""
echo "-------------------------------------------------------------"
echo "# comparison"
echo "-------------------------------------------------------------"
srun python -u scripts/compare_models.py --pattern 'results/llm_probe_basevsinst_*.json'

echo ""
echo "Done. Pull back with:  bash scripts/fetch_results.sh"
