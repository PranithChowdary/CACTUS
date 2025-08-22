"""
config.py

Configuration file for the MetaTool RL project.
Centralizes dataset paths, environment parameters, and training hyperparameters.
"""

import os

# === PATHS ===
DATASET_PATH = "data/Raw/all_clean_data.csv"
LOGS_DIR = "utils/logs"

# === ENVIRONMENT ===
MAX_STEPS = 1        # For single-shot predictions (change for multi-step)
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

ACTION_LIST = ['Ahrefs', 'Connected Papers', 'Elicit', 'Google Scholar', 'Moz', 'ResearchFinder', 'ResearchHelper', 'ResearchRabbit', 'SEMRush', 'SEOTool', 'SciSpace', 'Semantic Scholar']

# Create logs directory if missing
os.makedirs(LOGS_DIR, exist_ok=True)
print("Configuration loaded successfully.\n")
# print("Configurations:")
# for key, value in globals().items():
#     if key.isupper():
#         print(f"  {key}: {value}")
