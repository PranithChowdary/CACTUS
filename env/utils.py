"""
utils.py

Utility functions for the MetaTool RL environment.
Handles dataset loading, preprocessing, reward calculation, and logging.
"""

import os
import json
import pandas as pd
import torch
import random
from typing import List, Dict, Tuple


def load_dataset(path: str) -> pd.DataFrame:
    """
    Loads the MetaTool dataset from CSV/JSON.

    Args:
        path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".json"):
        return pd.read_json(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path}")


def preprocess_dataset(df: pd.DataFrame) -> List[Dict]:
    """
    Preprocess dataset into environment-friendly format.

    Args:
        df (pd.DataFrame): Raw dataset with 'prompt' and 'tools' columns.

    Returns:
        List[Dict]: List of samples with prompt text and available tools.
    """
    processed = []
    for _, row in df.iterrows():
        tools = [t.strip() for t in row["tools"].split(",")]
        processed.append({
            "prompt": row["prompt"],
            "tools": tools
        })
    return processed


def select_random_sample(dataset: List[Dict]) -> Dict:
    """
    Randomly selects a prompt and its tools.

    Args:
        dataset (List[Dict]): Processed dataset.

    Returns:
        Dict: Selected sample.
    """
    return random.choice(dataset)


def calculate_reward(selected_tool: str, correct_tools: List[str]) -> float:
    """
    Reward function:
    +1 for correct tool, -0.5 for incorrect tool.

    Args:
        selected_tool (str): Tool chosen by the agent.
        correct_tools (List[str]): Ground-truth tools for the prompt.

    Returns:
        float: Reward value.
    """
    return 1.0 if selected_tool in correct_tools else -0.5


def save_json(data: dict, path: str):
    """
    Saves a dictionary to a JSON file.

    Args:
        data (dict): Data to save.
        path (str): Path to save file.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def set_seed(seed: int = 42):
    """
    Sets random seed for reproducibility.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)