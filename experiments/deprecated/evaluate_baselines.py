import json
import pandas as pd
import numpy as np
from env.metatool_env import MetaToolEnv
from models.baseline_agents import RandomAgent, HeuristicAgent, DenseRetrievalAgent
from env import config
from typing import Dict, List

def evaluate_agent(agent, env, num_episodes=50):
    results = []
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            # Note: Baselines use the raw prompt from info
            prompt = info["prompt"]
            
            # Special case for DenseRetrieval to avoid re-selecting
            if isinstance(agent, DenseRetrievalAgent):
                selected_indices = np.where(obs["selection"] == 1)[0].tolist()
                action = agent.select_action(prompt, exclude_indices=selected_indices)
            else:
                action = agent.select_action(prompt)
                
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
        # Calculate F1 for this episode
        tp = len(set(info["selected_tools"]) & set(info["correct_tools"]))
        precision = tp / max(1, len(info["selected_tools"]))
        recall = tp / max(1, len(info["correct_tools"]))
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            "reward": total_reward,
            "f1": f1,
            "success": 1.0 if set(info["selected_tools"]) == set(info["correct_tools"]) else 0.0
        })
        
    return pd.DataFrame(results).mean().to_dict()

def main():
    # Load test data
    with open(config.TEST_DATA, 'r') as f:
        test_df = pd.DataFrame(json.load(f))
    with open(config.VOCAB_PATH, 'r') as f:
        vocab = json.load(f)
        
    env = MetaToolEnv(test_df, vocab, max_steps=5, mode="multi-step", seed=42)
    
    agents = {
        "Random": RandomAgent(vocab),
        "Heuristic": HeuristicAgent(vocab),
        "DenseRetrieval": DenseRetrievalAgent(vocab, model_name=config.EMBEDDING_MODEL)
    }
    
    summary = {}
    for name, agent in agents.items():
        print(f"Evaluating {name}...")
        metrics = evaluate_agent(agent, env, num_episodes=len(test_df))
        summary[name] = metrics
        print(f"Results for {name}: {metrics}")
        
    # Save results
    results_path = os.path.join(config.BASE_DIR, "results", "baseline_evaluation.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"All baseline evaluations complete. Results saved to {results_path}")

if __name__ == "__main__":
    import os
    main()
