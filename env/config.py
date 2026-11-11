"""
config.py - Research Configuration

Centralized configuration for CACTUS experiments.
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === PATHS ===
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
TRAIN_DATA = os.path.join(DATASETS_DIR, "train.json")
VAL_DATA = os.path.join(DATASETS_DIR, "val.json")
TEST_DATA = os.path.join(DATASETS_DIR, "test.json")
VOCAB_PATH = os.path.join(DATASETS_DIR, "vocab.json")

LOGS_DIR = os.path.join(BASE_DIR, "logs")
CHECKPOINT_DIR = os.path.join(LOGS_DIR, "checkpoints")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# === ENVIRONMENT ===
# len(vocab) = 15 tools + DONE = 16 actions. With action masking, a
# trajectory can select at most 15 tools before DONE is the only unmasked
# action left, so MAX_STEPS is a safety truncation, not the real bound -- it
# must be identical between training and evaluation, so it lives here once.
MAX_STEPS = 20
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Cluster compute nodes have no outbound internet, so the encoder cannot be
# fetched from the HF hub at job time. scripts/setup_login_node.sh downloads
# it once on the login node and saves a standalone copy here; when that
# directory exists it is loaded directly from disk -- no hub lookup, and no
# dependence on HF_HOME/cache layout matching between login and compute nodes.
EMBEDDING_MODEL_DIR = os.path.join(BASE_DIR, "models", "embedding", EMBEDDING_MODEL)


def resolve_embedding_model(model_name: str = None) -> str:
    """Return the vendored local model directory if present, else the hub name.

    Entry points should call this rather than passing EMBEDDING_MODEL directly,
    so a job on an offline compute node picks up the local copy automatically.
    """
    if os.path.isdir(EMBEDDING_MODEL_DIR):
        print(f"[config] Using vendored embedding model: {EMBEDDING_MODEL_DIR}")
        return EMBEDDING_MODEL_DIR
    resolved = model_name or EMBEDDING_MODEL
    print(f"[config] No vendored model at {EMBEDDING_MODEL_DIR}; "
          f"falling back to hub name '{resolved}' (requires network or a warm HF cache). "
          f"Run scripts/setup_login_node.sh on the login node to vendor it.")
    return resolved

# === RL TRAINING ===
RL_ALGO = "MaskablePPO"
LEARNING_RATE = 3e-4
GAMMA = 0.99
BATCH_SIZE = 64
NUM_TIMESTEPS = 100000  # Total steps for training, per seed

# === SEEDS ===
# A single-seed result is an anecdote. Every training/eval script should
# loop over SEEDS and report mean +/- std, never a single run.
SEED = 42
SEEDS = [42, 43, 44, 45, 46]
