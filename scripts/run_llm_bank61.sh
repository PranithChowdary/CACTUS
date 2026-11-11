#!/bin/bash
#SBATCH --job-name=bank61
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
# THE LAST JOB. Re-measures the LLM side on the 61-scenario bank so the LLM and
# RL numbers finally rest on the same n.
#
# Why this is needed and why it is the ONLY thing still outstanding:
# the RL results (results/nl_alpha_shift.json) use all 61 scenarios, but every
# LLM number still comes from the 12-scenario bank. At alpha=0 the policy is a
# deterministic argmax over deterministic scenarios, so all 4 repeats of a
# scenario are byte-identical -- verified 12/12 for all four models. The
# effective n was therefore 12, not the 48 that was reported, and every alpha=0
# rate is a multiple of 1/12 (0.833 = 10/12, 0.333 = 4/12, 0.083 = 1/12).
# Publishing those as n=48 overstates precision by about 2x.
#
# --episodes 61 gives each scenario EXACTLY ONCE (the harness picks
# scenario = i % len(SCENARIOS)), so n_effective = 61 at alpha=0 with no wasted
# duplicate forward passes. At alpha=1 each scenario draws one random branch,
# which is 61 independent episodes -- adding repeats would buy branch coverage
# per scenario, not sample size, and is not worth the GPU hours.
#
# Nothing here changes the action space: the bank grew 12 -> 61 while TOOLS
# stayed at 36 + DONE, so "DONE ranked 35th of 35" stays directly comparable to
# the earlier runs, which are kept as provenance under the old filenames.
#
# Budget, measured from the previous runs' own wall_seconds scaled by 61/48:
#   llama-3.2-1b   ~9 min      qwen2.5-7b  ~29 min      llama-3.1-8b  ~36 min
# plus model loads => ~1.5 h against an 8 h limit. 70B is a separate job
# (scripts/run_llm_bank61_70b.sh) because it needs ~5 h on its own.
#
#   sbatch scripts/run_llm_bank61.sh

# cn4_mangala caps cpus-per-task at 8 and memory at ~94 GB per GPU. Earlier
# jobs in this project ran with 16 CPUs and were accepted, but 8 is the stated
# limit, so these conform rather than rely on that.
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

MODELS=(
  "llama-3.2-1b|meta-llama/Llama-3.2-1B-Instruct"
  "qwen2.5-7b|Qwen/Qwen2.5-7B-Instruct"
  "llama-3.1-8b|meta-llama/Meta-Llama-3.1-8B-Instruct"
)

for entry in "${MODELS[@]}"; do
    TAG="${entry%%|*}"; MODEL="${entry##*|}"
    echo ""
    echo "-------------------------------------------------------------"
    echo "# $TAG   ($MODEL)"
    echo "-------------------------------------------------------------"

    # Re-verified per model: chat templates and BPE differ per family, and a
    # silent tokenization mismatch corrupts every score while looking fine.
    echo "--- verifying scoring maths on this model's tokenizer ---"
    srun python -u experiments/verify_llm_policy.py --model "$MODEL"
    if [ $? -ne 0 ]; then
        echo "!! CRITICAL verification failure for $TAG -- SKIPPING (scores would be invalid)"
        continue
    fi

    srun python -u experiments/llm_probe_constrained.py \
        --model "$MODEL" --model_tag "$TAG" \
        --episodes 61 --alphas 0.0 1.0 --scorings sum pmi \
        --out results/llm_probe_bank61.json
done

echo ""
echo "-------------------------------------------------------------"
echo "# cross-model comparison (61-scenario bank)"
echo "-------------------------------------------------------------"
srun python -u scripts/compare_models.py --pattern 'results/llm_probe_bank61_*.json'

echo ""
echo "Done. Pull back with:  bash scripts/fetch_results.sh"
