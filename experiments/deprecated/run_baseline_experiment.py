
# run_baseline_experiment.py

# Runs baseline agents on the MetaTool datasets and logs the results to TensorBoard.

import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from env.metatool_env import MetaToolEnv
from models.baseline_agents import RandomAgent, HeuristicAgent
import os

def load_and_prepare_data(data_path):
    """Loads a dataset and returns a unified tool vocabulary."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    
    df = pd.read_csv(data_path, encoding='ISO-8859-1')
    
    # Rename columns to be consistent
    if "Query" in df.columns:
        df = df.rename(columns={"Query": "prompt"})
    if "Tool" in df.columns:
        df = df.rename(columns={"Tool": "tools"})
    if "New Tools" in df.columns:
        # Assuming 'New Tools' are also valid tools
        df['tools'] = df.apply(lambda row: f"{row['tools']},{row['New Tools']}" if pd.notna(row['New Tools']) else row['tools'], axis=1)

    if "prompt" not in df.columns or "tools" not in df.columns:
        raise ValueError("Dataset must have 'prompt' and 'tools' columns")
        
    return df

def get_tool_vocab(df_list):
    """Creates a global tool vocabulary from a list of dataframes."""
    tool_vocab = set()
    for df in df_list:
        for tool_string in df["tools"].dropna():
            for tool in tool_string.split(","):
                tool_vocab.add(tool.strip())
    return sorted(list(tool_vocab))

def run_experiment(agent, env, num_episodes, writer):
    """Runs a single experiment and logs results to TensorBoard."""
    total_reward = 0
    for episode in range(num_episodes):
        timestep = env.reset()
        episode_reward = 0
        
        while not timestep.last():
            prompt = timestep.observation["prompt"]
            action = agent.select_action(prompt)
            timestep = env.step(action)
            episode_reward += timestep.reward
        
        total_reward += episode_reward
        writer.add_scalar("Reward/episode", episode_reward, episode)
        
    avg_reward = total_reward / num_episodes
    writer.add_scalar("Reward/average", avg_reward, 0)
    print(f"Agent: {agent.__class__.__name__}, Average Reward: {avg_reward}")

if __name__ == "__main__":
    # Define dataset paths
    processed_data_path = "data/Processed/editeddataset.csv"
    raw_data_path = "data/Raw/all_clean_data.csv"

    # Load datasets
    df_processed = load_and_prepare_data(processed_data_path)
    df_raw = load_and_prepare_data(raw_data_path)
    
    # Create a unified tool vocabulary
    tool_vocab = get_tool_vocab([df_processed, df_raw])
    
    datasets = {
        "processed": df_processed,
        "raw": df_raw
    }
    
    agents = {
        "random": RandomAgent(action_list=tool_vocab),
        "heuristic": HeuristicAgent(action_list=tool_vocab)
    }
    
    num_episodes = 100

    for d_name, d_frame in datasets.items():
        for a_name, agent_instance in agents.items():
            log_dir = f"runs/{d_name}_{a_name}"
            writer = SummaryWriter(log_dir)
            
            print(f"\n--- Running Experiment: Dataset='{d_name}', Agent='{a_name}' ---")
            
            # Create the environment
            env = MetaToolEnv(
                dataframe=d_frame,
                tool_vocab=tool_vocab,
                mode="single-shot" # Using single-shot as per baseline agents
            )
            
            run_experiment(agent_instance, env, num_episodes, writer)
            writer.close()

    print("\nAll experiments complete. Check the 'runs' directory for TensorBoard logs.")