import json
import matplotlib.pyplot as plt
import pandas as pd
import os

def visualize():
    results_path = "results/final_comparison.json"
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Run experiments/evaluate_all.py first.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data).T
    print("Results Summary:")
    print(df)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # F1 Score comparison
    df["f1"].plot(kind="bar", ax=ax1, color=['skyblue', 'orange', 'lightgreen', 'salmon'])
    ax1.set_title("F1 Score Comparison")
    ax1.set_ylabel("F1 Score")
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Success Rate comparison
    df["success"].plot(kind="bar", ax=ax2, color=['skyblue', 'orange', 'lightgreen', 'salmon'])
    ax2.set_title("Success Rate Comparison")
    ax2.set_ylabel("Success Rate")
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = "results/performance_comparison.png"
    plt.savefig(plot_path)
    print(f"Visualization saved to {plot_path}")

if __name__ == "__main__":
    visualize()
