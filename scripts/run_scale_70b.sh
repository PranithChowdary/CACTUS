#!/bin/bash
#SBATCH --job-name=scale70b
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# SCALE AXIS: does termination emerge with model size?
#
# On LLaMA-3.1-8B the model ranked correct tools first in 83% of static tasks
# and probed before committing in 92% of conditional ones, yet chose DONE in
# 0/288 decisions (rank 35 of 35 at every step). "One 8B model cannot stop" is
# a much weaker claim than "8B-class instruct models cannot stop", so this runs
# the identical protocol across FAMILIES at fixed size and across SCALE within
# a family.
#
# This is the most informative single run in the study. If 70B also never
# chooses DONE, the claim is about the model class rather than about capacity,
# and scaling is not the answer. If 70B DOES stop, termination is an emergent
# capability -- a different and equally publishable result.
#
# 70B in fp16 is ~140GB, so this needs 2 GPUs (2 x 96GB) and device_map="auto"
# shards it automatically. Runtime is several times the 8B run.
#
# --batch_size 8 (not the default 48): the forward pass materialises a
# [batch, seq, vocab] logits tensor, and with vocab=128256 and seq~700 that is
# ~0.18 GB per row in fp16. At batch 48 it sits beside an 89 GB sharded model
# and the first attempt died with a 10.86 GiB OOM. The scoring code now slices
# before casting to float32 (models/llm_policy.py), which removed a 12.6 GB
# allocation, but the forward-pass logits still scale with batch size.
#
# Each model is verified before it is trusted: verify_llm_policy.py re-runs the
# scoring-maths checks on that model's own tokenizer, since chat templates and
# BPE differ per family and a silent tokenization mismatch would corrupt every
# score. A critical failure skips that model rather than aborting the study.
#
# ~40 min per model (2 scorings x 2 alpha x 48 episodes). 1B is much faster.
#
#   sbatch scripts/run_scale_70b.sh

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
  "llama-3.1-70b|meta-llama/Llama-3.1-70B-Instruct"
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
        --episodes 48 --alphas 0.0 1.0 --batch_size 8
done

echo ""
echo "-------------------------------------------------------------"
echo "# cross-model comparison"
echo "-------------------------------------------------------------"
srun python -u scripts/compare_models.py
