#!/bin/bash
#SBATCH --job-name=nativestop
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=92G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# JOB 1 OF 2 -- and the one that matters most.
#
# Tests whether this paper's central claim survives its most serious objection:
# that scoring a literal "DONE" string measures an interface mismatch rather
# than a missing capability. These models were post-trained to end a turn with
# a native end-of-turn token, not by naming a pseudo-tool. This job scores each
# model's own native stop token alongside DONE at every step, changing nothing
# else.
#
# It also closes a second gap. We currently log the mass on stopping only at
# the completing step, so we cannot tell a model that RECOGNISES completion
# from one with a high constant prior on a short string -- the open question
# about Qwen2.5-7B, whose stop action ranks 2.1 averaged over ALL steps. This
# job logs both channels at every step, giving the before/after contrast.
#
# Two models only, both small: the question is about the interface, and the
# 70B model adds cost without adding evidence on that question. ~1 GPU-hour
# total at 61 episodes, based on the bank61 timings (8B 36 min, Qwen 29 min
# for four configs; this runs one).
#
# READ THE INTERPRETATION BLOCK THE SCRIPT PRINTS. If native stop carries
# substantial mass where DONE does not, the paper's framing has to change
# before submission, not after review.
#
#   sbatch scripts/run_native_stop_probe.sh

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

MODELS=(
  "llama-3.1-8b|meta-llama/Meta-Llama-3.1-8B-Instruct"
  "qwen2.5-7b|Qwen/Qwen2.5-7B-Instruct"
)

for entry in "${MODELS[@]}"; do
    TAG="${entry%%|*}"; MODEL="${entry##*|}"
    echo ""
    echo "-------------------------------------------------------------"
    echo "# $TAG   ($MODEL)"
    echo "-------------------------------------------------------------"
    srun python -u experiments/native_stop_probe.py \
        --model "$MODEL" --model_tag "$TAG" \
        --episodes 61 --alphas 0.0 \
        --out results/native_stop.json
done

echo ""
echo "Done. Pull back with:  bash scripts/fetch_results.sh"
