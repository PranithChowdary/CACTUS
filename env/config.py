"""
config.py

Configuration file for the MetaTool RL project.
Centralizes dataset paths, environment parameters, and training hyperparameters.
"""

import os

# === PATHS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", DATASET_PATH)
LOGS_DIR = os.path.join(BASE_DIR, "utils")

# === ENVIRONMENT ===
MAX_STEPS = 1         # For single-shot predictions (change for multi-step)
OBSERVATION_TYPE = "text"  # Options: "text", "embedding"

# === RL TRAINING ===
RL_ALGO = "PPO"       # Options: "PPO", "DQN", "A2C"
LEARNING_RATE = 3e-4
GAMMA = 0.99
BATCH_SIZE = 32
NUM_EPISODES = 1000

# === REWARD SETTINGS ===
REWARD_CORRECT = 1.0
REWARD_INCORRECT = -0.5

# === SEED ===
SEED = 42

# Create logs directory if missing
os.makedirs(LOGS_DIR, exist_ok=True)
