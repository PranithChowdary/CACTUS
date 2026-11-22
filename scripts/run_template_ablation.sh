#!/bin/bash
#SBATCH --job-name=tmplabl
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Closes the one confound left in the base-vs-instruct result.
#
# ConstrainedLLMPolicy applies the tokenizer's chat template by default and
# run_base_vs_instruct.sh did not override it. That is correct for instruct
# checkpoints and questionable for the base one: Qwen2.5-7B base never saw a
# ChatML wrapper in pretraining. So a sceptic can argue the 75x gap in
# P(DONE|complete) is partly prompt formatting rather than post-training.
#
# This completes the 2x2. We already have the top row with the template on;
# this job runs the bottom row with it off.
#
#                     template ON          template OFF
#   base              0.0076  (have)       <- this job
#   instruct          0.5753  (have)       <- this job
#
# Readings:
#   base stays near zero  -> formatting is NOT the explanation, the confound is
#                            dead and the 75x stands. Expected.
#   base rises sharply    -> part of the gap was formatting. Report the
#                            no-template number as the headline instead and
#                            rewrite sec:basevsinst accordingly.
#   instruct falls        -> interesting in its own right: Qwen's termination
#                            would depend on being addressed in chat format,
#                            which is a finding, not a problem.
#
# Instruct is included deliberately rather than running base alone. Without it
# a change in the base number cannot be separated from a template effect that
# applies to both, and a reviewer will ask.
#
# ~0.5 GPU-hours. One job, well inside the 4-job QOS limit.
#
#   sbatch scripts/run_template_ablation.sh

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

BASE_DIR=/users/staff/pranith/models

# tag                      | model path or repo id            | source
MODELS=(
  "qwen2.5-7b-base-notmpl|$BASE_DIR/Qwen2.5-7B|local"
  "qwen2.5-7b-inst-notmpl|Qwen/Qwen2.5-7B-Instruct|cache"
)

for entry in "${MODELS[@]}"; do
    TAG="${entry%%|*}"; REST="${entry#*|}"
    MODEL="${REST%%|*}"; SOURCE="${REST##*|}"
    echo ""
    echo "-------------------------------------------------------------"
    echo "# $TAG   ($MODEL)   chat template OFF"
    echo "-------------------------------------------------------------"

    if [ "$SOURCE" = "local" ] && [ ! -f "$MODEL/config.json" ]; then
        echo "!! $TAG NOT PRESENT -- no config.json at $MODEL"
        echo "!! login node:  python scripts/fetch_base_models.py --check"
        continue
    fi

    # No verification pass here: verify_llm_policy.py already runs with
    # use_chat_template=False, so the 6/6 it returned in job 67978 IS the
    # verification for this configuration. Re-running it would test nothing new.

    srun python -u experiments/llm_probe_constrained.py \
        --model "$MODEL" --model_tag "$TAG" --no_chat_template \
        --episodes 61 --alphas 0.0 1.0 --scorings sum \
        --out results/llm_probe_notmpl.json
done

echo ""
echo "-------------------------------------------------------------"
echo "# the 2x2 -- compare against results/llm_probe_basevsinst_*.json"
echo "-------------------------------------------------------------"
srun python -u scripts/compare_models.py --pattern 'results/llm_probe_notmpl_*.json'

echo ""
echo "Template ON numbers for reference (job 67978, summed scoring, alpha 0):"
echo "  qwen2.5-7b-base   P(DONE|cmpl) 0.0076   stops 0/61"
echo "  qwen2.5-7b-inst   P(DONE|cmpl) 0.5753   stops 60/61"
echo ""
echo "Done. Pull back with:  bash scripts/fetch_results.sh"
