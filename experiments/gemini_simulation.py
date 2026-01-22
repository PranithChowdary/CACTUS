
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import os

# --- Configuration ---
LOG_DIR = "logs/gemini_simulation"
DATA_PATH = "data/Processed/editeddataset.csv"
NUM_SIMULATION_STEPS = 100
MODELS_TO_SIMULATE = [
    "Gemini-3B-SFT",
    "Gemini-3B-RLHF",
    "RandomAgent",
    "HeuristicAgent",
    "RLAgent-PPO",
    "ScriptedAgent",
]

# --- Simulation Logic ---

def get_model_performance_profile(model_name):
    """
    Defines the base performance and improvement characteristics for each model.
    Returns a dictionary with base metrics and learning rates.
    """
    profiles = {
        "Gemini-3B-SFT": {"acc": 0.85, "f1": 0.88, "precision": 0.90, "recall": 0.86, "latency": 0.5, "improvement": 0.05},
        "Gemini-3B-RLHF": {"acc": 0.88, "f1": 0.92, "precision": 0.94, "recall": 0.90, "latency": 0.6, "improvement": 0.07},
        "RLAgent-PPO": {"acc": 0.60, "f1": 0.65, "precision": 0.68, "recall": 0.62, "latency": 0.2, "improvement": 0.35},
        "HeuristicAgent": {"acc": 0.45, "f1": 0.50, "precision": 0.55, "recall": 0.48, "latency": 0.1, "improvement": 0.02},
        "ScriptedAgent": {"acc": 0.40, "f1": 0.45, "precision": 0.50, "recall": 0.42, "latency": 0.1, "improvement": 0.01},
        "RandomAgent": {"acc": 0.15, "f1": 0.18, "precision": 0.20, "recall": 0.16, "latency": 0.05, "improvement": 0.0},
    }
    return profiles.get(model_name, {"acc": 0, "f1": 0, "precision": 0, "recall": 0, "latency": 0, "improvement": 0})

def simulate_step(step, total_steps, profile):
    """
    Simulates a single step of model evaluation, showing gradual improvement.
    """
    progress = step / total_steps
    improvement_factor = 1 + (profile["improvement"] * progress)

    # Simulate metrics with some noise
    accuracy = min(0.99, profile["acc"] * improvement_factor + np.random.normal(0, 0.02))
    f1 = min(0.99, profile["f1"] * improvement_factor + np.random.normal(0, 0.02))
    precision = min(0.99, profile["precision"] * improvement_factor + np.random.normal(0, 0.01))
    recall = min(0.99, profile["recall"] * improvement_factor + np.random.normal(0, 0.03))
    latency = max(0.01, profile["latency"] + np.random.normal(0, 0.05) * (1 - progress))

    return {
        "Accuracy": max(0, accuracy),
        "F1-Score": max(0, f1),
        "Precision": max(0, precision),
        "Recall": max(0, recall),
        "Latency (s)": max(0, latency),
    }

def run_simulation():
    """
    Main function to run the simulation and write to TensorBoard.
    """
    print(f"Starting simulation. Logging to '{LOG_DIR}'")

    # --- Setup ---
    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)

    # Load dataset (we'll just use its length for simulation context)
    try:
        dataset = pd.read_csv(DATA_PATH, encoding='latin1')
        print(f"Successfully loaded {len(dataset)} records from '{DATA_PATH}'")
    except FileNotFoundError:
        print(f"Warning: Dataset not found at '{DATA_PATH}'. Running simulation without data context.")
        dataset = []

    # --- Simulation Loop ---
    for step in range(NUM_SIMULATION_STEPS):
        # Group scalars for cleaner TensorBoard graphs
        accuracy_scalars = {}
        f1_scalars = {}
        latency_scalars = {}

        for model_name in MODELS_TO_SIMULATE:
            profile = get_model_performance_profile(model_name)
            metrics = simulate_step(step, NUM_SIMULATION_STEPS, profile)

            # Log individual metrics for each model
            for metric_name, value in metrics.items():
                writer.add_scalar(f"{model_name}/{metric_name}", value, step)

            # Collect scalars for grouped charts
            accuracy_scalars[model_name] = metrics["Accuracy"]
            f1_scalars[model_name] = metrics["F1-Score"]
            latency_scalars[model_name] = metrics["Latency (s)"]

        # Add grouped scalars
        writer.add_scalars("Group/Accuracy", accuracy_scalars, step)
        writer.add_scalars("Group/F1-Score", f1_scalars, step)
        writer.add_scalars("Group/Latency", latency_scalars, step)

        if (step + 1) % 10 == 0:
            print(f"Simulation step {step + 1}/{NUM_SIMULATION_STEPS} complete.")
            time.sleep(0.1) # Simulate work

    writer.close()
    print("\nSimulation finished successfully!")
    print("To view the results, run the following command in your terminal:")
    print(f"tensorboard --logdir {LOG_DIR}")

if __name__ == "__main__":
    run_simulation()
