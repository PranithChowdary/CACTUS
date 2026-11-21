#!/bin/bash
#SBATCH --job-name=basevsinst
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# DO NOT repoint this at cn3_anandi. Run 67880 did, and verify_llm_policy.py
# failed checks 1 and 2 there (batch invariance 1.35e-01, manual agreement
# 0.135 nats) for the very instruct models that pass at exactly 0.00e+00 on
# mangala. The header and the skip message below both depend on this.
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
# NOTE ON HARDWARE: the verification must pass on the SAME node type that runs
# the experiment. It loads the model in float32 on CPU while the experiment runs
# fp16 on GPU, so it is checking the scoring logic rather than the exact kernels
# -- but batched-vs-unbatched agreement still depends on the BLAS path, and that
# differs between anandi and mangala. Treat a failure as a stop sign.
#
# NOTE ON BASE MODELS: a base model has no chat template. verify_llm_policy.py
# is run first for each and a critical failure SKIPS that model rather than
# producing scores that look valid and are not.
#
# The base checkpoints CANNOT live in /janaki/common/huggingface -- that cache
# is read-only to this account (writing a ref there fails with EACCES), so a
# repo id for a base model will never resolve under HF_HUB_OFFLINE. They are
# loaded from BASE_DIR below as filesystem paths instead; both the verifier and
# the probe pass --model straight to from_pretrained, so a path is equivalent
# to an id. The instruct models DO resolve from the shared cache, which already
# holds them, so HF_HOME still points there.
#
# To populate BASE_DIR, on the LOGIN node (which has internet):
#   python scripts/fetch_base_models.py            # download
#   python scripts/fetch_base_models.py --check    # probe access only
# That script overrides the read-only HF_HOME, probes gating by resolving a
# real file (model_info() does NOT work -- metadata on a gated repo is public,
# so it returns success for a repo you cannot download), skips the ~16 GB of
# duplicate original/*.pth weights, and verifies shard completeness afterwards.
#
# Note the LLaMA base id is Llama-3.1-8B, not Meta-Llama-3.1-8B. It is gated
# and approval is per-account: the shared cache holds only -Instruct variants,
# so an admin's access does not transfer. Accept the licence at
# https://huggingface.co/meta-llama/Llama-3.1-8B first.
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

# Base checkpoints are local directories (see NOTE ON BASE MODELS); instruct
# checkpoints are repo ids resolved from the read-only shared cache.
BASE_DIR=/users/staff/pranith/models

# tag                 | model path or repo id            | source
MODELS=(
  "llama-3.1-8b-base|$BASE_DIR/Llama-3.1-8B|local"
  "llama-3.1-8b-inst|meta-llama/Meta-Llama-3.1-8B-Instruct|cache"
  "qwen2.5-7b-base|$BASE_DIR/Qwen2.5-7B|local"
  "qwen2.5-7b-inst|Qwen/Qwen2.5-7B-Instruct|cache"
)

for entry in "${MODELS[@]}"; do
    TAG="${entry%%|*}"; REST="${entry#*|}"
    MODEL="${REST%%|*}"; SOURCE="${REST##*|}"
    echo ""
    echo "-------------------------------------------------------------"
    echo "# $TAG   ($MODEL)"
    echo "-------------------------------------------------------------"

    # Preflight for local checkpoints, so a missing download reports itself as
    # a missing download and not as a verification failure.
    if [ "$SOURCE" = "local" ] && [ ! -f "$MODEL/config.json" ]; then
        echo "!! $TAG NOT PRESENT -- no config.json at $MODEL"
        echo "!! This is a missing download, NOT a scoring problem. See the"
        echo "!! login node:  python scripts/fetch_base_models.py --check"
        echo "!! Skipping $TAG; the remaining models still run."
        continue
    fi

    echo "--- verifying scoring maths on this model's tokenizer ---"
    srun python -u experiments/verify_llm_policy.py --model "$MODEL"
    if [ $? -ne 0 ]; then
        echo "!! CRITICAL verification failure for $TAG -- SKIPPING."
        echo "!! Read the numbered checks above for the ACTUAL reason. Two"
        echo "!! distinct causes look alike here:"
        echo "!!   checks 1-2 failing  -> numerical: batched and unbatched"
        echo "!!      scoring disagree. This is hardware/BLAS dependent and has"
        echo "!!      been seen on anandi while passing at 0.00e+00 on mangala."
        echo "!!      Run on mangala; do NOT loosen the tolerance to pass."
        echo "!!   model fails to LOAD -> the checkpoint is not in the offline"
        echo "!!      cache. Download it on the login node (see header)."
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
