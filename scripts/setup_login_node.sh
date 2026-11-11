#!/bin/bash
#
# Run this ON THE LOGIN NODE (kalpana), NOT via sbatch:
#
#   bash scripts/setup_login_node.sh
#
# Compute nodes on cn4_mangala have no outbound internet, so EVERY
# network-dependent step has to happen here, once, before submitting:
#   1. create/update the conda env and install requirements (pip needs PyPI)
#   2. download the sentence-transformers encoder and save a STANDALONE copy
#      into models/embedding/ inside the repo
#
# Step 2 deliberately does not rely on the HuggingFace cache. A saved model
# directory is loaded straight off disk by path, so there is no hub lookup and
# no way for HF_HOME/cache-permission mismatches between the login node and the
# compute node to break the job. env/config.py:resolve_embedding_model() picks
# that directory up automatically when it exists.

set -euo pipefail

source ~/miniconda3/bin/activate
cd /users/staff/pranith/CACTUS/

if [ ! -d "cactus_env" ]; then
    echo "Creating new local conda environment 'cactus_env'..."
    conda create --prefix ./cactus_env python=3.12 -y
fi
conda activate ./cactus_env

echo "========================================="
echo "1/2  Installing requirements (needs internet)"
echo "========================================="
pip install -r requirements.txt

echo "========================================="
echo "2/2  Vendoring the embedding model (needs internet)"
echo "========================================="

# Make sure nothing forces offline mode for THIS step -- we need the download.
unset HF_HUB_OFFLINE || true
unset TRANSFORMERS_OFFLINE || true

PYTHONPATH=. python - <<'PY'
from sentence_transformers import SentenceTransformer
from env import config

target = config.EMBEDDING_MODEL_DIR
print(f"Downloading '{config.EMBEDDING_MODEL}' and saving to: {target}")
model = SentenceTransformer(config.EMBEDDING_MODEL)
model.save(target)
print("Saved. Verifying it loads from disk...")

reloaded = SentenceTransformer(target)
dim = reloaded.get_sentence_embedding_dimension()
vec = reloaded.encode("verification sentence")
print(f"OK: loaded from {target} (dim={dim}, encoded shape={vec.shape})")
PY

echo "========================================="
echo "Setup complete."
echo "The encoder is vendored at models/embedding/ and will be loaded by path,"
echo "so the compute node needs no network and no HF cache."
echo "Now submit:  sbatch scripts/train_ppo_cluster.sh"
echo "========================================="
