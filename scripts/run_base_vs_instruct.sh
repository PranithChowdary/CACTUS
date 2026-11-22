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
<<<<<<< HEAD
# 0.135 nats) for the very instruct models that pass at exactly 0.00e+00 on
# mangala. The header and the skip message below both depend on this.
=======
# 0.135 nats) for the very models that pass at exactly 0.00e+00 on mangala.
>>>>>>> 6d553f0 (fin runs)
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
# OUTCOME (job 67978, 2026-11-21). The second reading held for Qwen:
#   qwen2.5-7b-base   P(DONE|cmpl) 0.0076   stops 0/61    recall@5 0.967
#   qwen2.5-7b-inst   P(DONE|cmpl) 0.5753   stops 60/61   recall@5 0.951
# A 75x change with corpus, tokenizer and architecture fixed, while relevance
# moved only 0.967 -> 0.951. Instruction tuning installs termination; the
# post-training claim is now an intervention rather than a correlation.
#
# The LLaMA half is UNRESOLVED: llama-3.1-8b-base is gated per account and was
# not granted. llama-3.1-8b-inst (0.0020) sits BELOW qwen base (0.0076), so two
# readings survive -- LLaMA's recipe failed to install termination, or it
# suppressed it. Only the base checkpoint separates them. Resubmit this script
# once access is granted; the preflight skips the model cleanly meanwhile.
#
# CAVEAT: under PMI the contrast is absent (DONE rank 35.0 base vs 35.5
# instruct, both last of ~35). The 75x is specific to summed scoring.
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
<<<<<<< HEAD
# The base checkpoints CANNOT live in /janaki/common/huggingface -- that cache
# is read-only to this account (writing a ref there fails with EACCES), so a
# repo id for a base model will never resolve under HF_HUB_OFFLINE. They are
# loaded from BASE_DIR below as filesystem paths instead; both the verifier and
# the probe pass --model straight to from_pretrained, so a path is equivalent
# to an id. The instruct models DO resolve from the shared cache, which already
# holds them, so HF_HOME still points there.
=======
# CONFOUND: ConstrainedLLMPolicy defaults use_chat_template=True and the probe
# does not override it, so each tokenizer's chat template IS applied -- to the
# base checkpoint too, which was not trained on it. Pass --no_chat_template to
# llm_probe_constrained.py to run the ablation that settles how much of the
# base/instruct gap is formatting (~0.3 GPU-h for the base model alone).
#
# The base checkpoints CANNOT live in /janaki/common/huggingface -- that cache
# is read-only to this account (writing a ref there fails with EACCES), so a
# repo id for a base model will never resolve under HF_HUB_OFFLINE. They are
# loaded from BASE_DIR below as filesystem paths; both the verifier and the
# probe pass --model straight to from_pretrained, so a path is equivalent to an
# id. The instruct models DO resolve from the shared cache, which already holds
# them, so HF_HOME still points there.
>>>>>>> 6d553f0 (fin runs)
#
# To populate BASE_DIR, on the LOGIN node (which has internet):
#   python scripts/fetch_base_models.py            # download
#   python scripts/fetch_base_models.py --check    # probe access only
<<<<<<< HEAD
# That script overrides the read-only HF_HOME, probes gating by resolving a
# real file (model_info() does NOT work -- metadata on a gated repo is public,
# so it returns success for a repo you cannot download), skips the ~16 GB of
# duplicate original/*.pth weights, and verifies shard completeness afterwards.
#
# Note the LLaMA base id is Llama-3.1-8B, not Meta-Llama-3.1-8B. It is gated
# and approval is per-account: the shared cache holds only -Instruct variants,
# so an admin's access does not transfer. Accept the licence at
# https://huggingface.co/meta-llama/Llama-3.1-8B first.
=======
# That script overrides the read-only HF_HOME before importing huggingface_hub,
# probes gating by resolving a real FILE (model_info() does NOT work -- metadata
# on a gated repo is public, so it succeeds for a repo you cannot download),
# skips the ~16 GB of duplicate original/*.pth weights, and verifies shard
# completeness afterwards.
#
# The LLaMA base id is Llama-3.1-8B, not Meta-Llama-3.1-8B. Gating is
# per-account: the shared cache holds only -Instruct variants, so an admin's
# access does not transfer.
>>>>>>> 6d553f0 (fin runs)
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
<<<<<<< HEAD
    # a missing download and not as a verification failure.
    if [ "$SOURCE" = "local" ] && [ ! -f "$MODEL/config.json" ]; then
        echo "!! $TAG NOT PRESENT -- no config.json at $MODEL"
        echo "!! This is a missing download, NOT a scoring problem. See the"
=======
    # a missing download and not as a verification failure. Run 67880 conflated
    # the two and cost a debugging round.
    if [ "$SOURCE" = "local" ] && [ ! -f "$MODEL/config.json" ]; then
        echo "!! $TAG NOT PRESENT -- no config.json at $MODEL"
        echo "!! This is a missing download, NOT a scoring problem."
>>>>>>> 6d553f0 (fin runs)
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
<<<<<<< HEAD
        echo "!!      scoring disagree. This is hardware/BLAS dependent and has"
        echo "!!      been seen on anandi while passing at 0.00e+00 on mangala."
        echo "!!      Run on mangala; do NOT loosen the tolerance to pass."
        echo "!!   model fails to LOAD -> the checkpoint is not in the offline"
        echo "!!      cache. Download it on the login node (see header)."
=======
        echo "!!      scoring disagree. Hardware/BLAS dependent; seen on anandi"
        echo "!!      while passing at 0.00e+00 on mangala. Do NOT loosen the"
        echo "!!      tolerance to pass."
        echo "!!   model fails to LOAD -> checkpoint not in the offline cache."
>>>>>>> 6d553f0 (fin runs)
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
