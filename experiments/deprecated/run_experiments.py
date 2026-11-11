"""
run_experiments.py

Orchestrates the experiments described in DESIGN.md:
- Creates dated log folders under `logs/YYYYMMDD_HHMMSS/<exp_name>/`
- Instantiates environment and agents
- Runs training/eval loop
- Logs scalars to TensorBoard and stores CSV summaries & checkpoints

Usage:
    python -m experiments.run_experiments --config experiments/experiments_config.py
"""

import os
import csv
import argparse
import datetime
import importlib
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# local imports (adjust if your layout differs)
from env.metatool_env import MetaToolEnv
from env import utils, config

# -------------------------
# Experiment definitions
# -------------------------
EXPERIMENTS = [
    {"name": "scripted", "agent_module": "models.scripted_agent", "agent_class": "ScriptedAgent", "params": {}},
    {"name": "random",   "agent_module": "models.random_agent",   "agent_class": "RandomAgent",   "params": {}},
    {"name": "rl_only",  "agent_module": "models.rl_agent",       "agent_class": "RLAgent",       "params": {"use_llm": False, "algo": "PPO"}},
    {"name": "rl_llm",   "agent_module": "models.rl_agent",       "agent_class": "RLAgent",       "params": {"use_llm": True,  "algo": "PPO", "llm_backend": "gpt"}},
    {"name": "ablated",  "agent_module": "models.rl_agent",       "agent_class": "RLAgent",       "params": {"use_llm": True,  "algo": "PPO", "llm_backend": "gpt", "llm_noise": True}},
    # Add further experiments (algos / LLM swaps) similarly
]

# -------------------------
# Helpers
# -------------------------
def make_log_dir(base_logs="logs"):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(base_logs) / ts
    base.mkdir(parents=True, exist_ok=True)
    return base

def load_dataset(path):
    # Use the util loader you already wrote
    return utils.load_dataset(path)

def instantiate_agent(module_name: str, class_name: str, **kwargs):
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls(**kwargs)

# -------------------------
# Main loop
# -------------------------
def run_one_experiment(exp_def: Dict[str, Any], dataset_df: pd.DataFrame, vocab: list, global_logdir: Path, episodes: int):
    exp_name = exp_def["name"]
    agent_module = exp_def["agent_module"]
    agent_class = exp_def["agent_class"]
    params = exp_def.get("params", {})

    exp_logdir = global_logdir / exp_name
    exp_logdir.mkdir(exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(exp_logdir / "tensorboard"))

    # instantiate environment (the env will sample prompts for each episode internally)
    env = MetaToolEnv(dataframe=dataset_df, tool_vocab=vocab, mode="multi-step", max_steps=config.MAX_STEPS, seed=config.SEED)

    # instantiate agent (agents must implement a standard interface described below)
    agent = instantiate_agent(agent_module, agent_class, env=env, logdir=str(exp_logdir), **params)

    # training loop - agents must implement `train(num_episodes, writer)` which returns a list of per-episode dict metrics
    print(f"[EXP START] {exp_name} - episodes={episodes}")
    episode_metrics = agent.train(num_episodes=episodes, writer=writer)

    # Save episode metrics to CSV
    csv_path = exp_logdir / "episode_metrics.csv"
    if episode_metrics:
        keys = sorted(episode_metrics[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=keys)
            writer_csv.writeheader()
            for row in episode_metrics:
                writer_csv.writerow(row)

    # Save agent checkpoint if supported
    try:
        ckpt_path = exp_logdir / "checkpoint.pkl"
        agent.save(ckpt_path)
    except Exception as e:
        print("Agent checkpoint save skipped or failed:", e)

    writer.close()
    print(f"[EXP END] {exp_name} logs at {exp_logdir}")

def run_all(experiments, dataset_df, vocab, total_episodes=500):
    base_logdir = make_log_dir()
    for exp in experiments:
        run_one_experiment(exp, dataset_df, vocab, base_logdir, episodes=total_episodes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=config.DATASET_PATH, help="path to dataset CSV")
    parser.add_argument("--vocab", type=str, default=None, help="path to tool vocab json (optional)")
    parser.add_argument("--episodes", type=int, default=200, help="episodes per experiment")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    # Build a global vocab if not provided
    if args.vocab:
        vocab = utils.load_vocab_json(args.vocab)
    else:
        vocab = utils.build_tool_vocab(df["tools"].tolist())  # implement this util to return list[str]

    run_all(EXPERIMENTS, df, vocab, total_episodes=args.episodes)