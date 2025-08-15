"""
test_env.py

Basic testing script for the MetaToolEnv environment.
Prevents infinite loops by removing chosen tools from available actions.

Author: CACTUS Authors
Date: 2025-08-08
"""

import pandas as pd
from env.metatool_env import MetaToolEnv



def run_single_episode(env, max_steps=20):
    ts = env.reset()
    print("\n=== New Episode ===")
    print("Prompt:", ts.observation["prompt"])
    print("Available Tools:", ts.observation["available_tools"])

    step_count = 0
    while not ts.last() and step_count < max_steps:
        if not ts.observation["available_tools"]:
            print("No tools left to choose. Ending episode.")
            break

        action = 0  # Always pick first available
        tool_name = ts.observation["available_tools"][action]
        print(f"\nStep {step_count} → Choosing tool index {action} ({tool_name})")

        ts = env.step(action)
        print("Reward:", ts.reward, "Done:", ts.last())
        print("Observation:", ts.observation)

        step_count += 1


if __name__ == "__main__":
    df = pd.DataFrame({
        "prompt": [
            "Find academic papers on reinforcement learning.",
            "Search for recent news about electric vehicles."
        ],
        "tools": [
            "Google Scholar, Arxiv",
            "Google News, Bing News"
        ]
    })

    vocab = [
        "Google Scholar", "Arxiv", "PubMed", "Mendeley", "ResearchGate",
        "Google News", "Bing News", "Yahoo News", "Reddit", "DONE"
    ]

    print("\n##### Testing Single-Shot Mode #####")
    env_single = MetaToolEnv(df, tool_vocab=vocab, mode="single-shot", seed=42)
    run_single_episode(env_single)

    print("\n##### Testing Multi-Step Mode #####")
    env_multi = MetaToolEnv(df, tool_vocab=vocab, mode="multi-step", seed=42)
    run_single_episode(env_multi)