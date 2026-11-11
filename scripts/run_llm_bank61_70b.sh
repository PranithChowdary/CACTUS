#!/bin/bash
#SBATCH --job-name=bank61_70b
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# 70B on the 61-scenario bank -- the scale axis at the same n as every other
# model. Split out from scripts/run_llm_bank61.sh purely on cost: 70B took
# 57.4 min per config at 48 episodes, so at 61 it is ~73 min x 4 configs
# ~= 4.9 h plus a ~17 min shard load. It does not fit beside the small models.
#
# --batch_size 8 (not the default 48): the forward pass materialises a
# [batch, seq, vocab] logits tensor and with vocab=128256, seq~700 that is
# ~0.18 GB per row in fp16, sitting beside an 89 GB sharded model. The first
# 70B attempt died with a 10.86 GiB OOM at batch 48 -- AFTER passing every
# verification check, 17 minutes of loading in. models/llm_policy.py now slices
# the logits before casting to float32 (that removed a separate 12.6 GB
# allocation and is numerically identical, verified -20.8731 both ways), but
# the forward-pass logits still scale with batch size. Do not raise this.
#
# 2 GPUs: 70B in fp16 is ~140 GB, device_map="auto" shards it.
#
# Results are written after each config, so a timeout keeps completed configs.
#
#   sbatch scripts/run_llm_bank61_70b.sh

# cn4_mangala caps cpus-per-task at 8 and memory at ~94 GB per GPU. Earlier
# jobs in this project ran with 16 CPUs and were accepted, but 8 is the stated
# limit, so these conform rather than rely on that. 90G of CPU RAM is ample for streaming the 70B shards.
set -uo pipefail

source ~/miniconda3/bin/activate
cd /users/staff/pranith/CACTUS/
conda activate ./cactus_env

export PYTHONPATH=.
export HF_HOME=/janaki/common/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p logs results

echo "bank size check:"
srun python -u -c "from env.nl_conditional_env import SCENARIOS; print(f'  {len(SCENARIOS)} scenarios'); assert len(SCENARIOS)==61, 'expected the 61-scenario bank'"
if [ $? -ne 0 ]; then echo "!! wrong scenario bank on this host -- rsync env/nl_scenarios.py"; exit 1; fi

TAG="llama-3.1-70b"
MODEL="meta-llama/Llama-3.1-70B-Instruct"

echo "--- verifying scoring maths on this model's tokenizer ---"
srun python -u experiments/verify_llm_policy.py --model "$MODEL"
if [ $? -ne 0 ]; then
    echo "!! CRITICAL verification failure -- aborting (scores would be invalid)"
    exit 1
fi

srun python -u experiments/llm_probe_constrained.py \
    --model "$MODEL" --model_tag "$TAG" \
    --episodes 61 --alphas 0.0 1.0 --scorings sum pmi \
    --batch_size 8 \
    --out results/llm_probe_bank61.json

echo ""
echo "Done. Pull back with:  bash scripts/fetch_results.sh"
echo "Then:  python scripts/compare_models.py --pattern 'results/llm_probe_bank61_*.json'"
