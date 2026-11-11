import argparse
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from env.metatool_env import MetaToolEnv
from env.wrappers import TextEmbeddingWrapper
from models.ppo_agent import PPOAgent
from env import config


def train_one_seed(seed: int, timesteps: int):
    with open(config.TRAIN_DATA, 'r') as f:
        train_df = pd.DataFrame(json.load(f))
    with open(config.VOCAB_PATH, 'r') as f:
        vocab = json.load(f)

    base_env = MetaToolEnv(
        dataframe=train_df,
        tool_vocab=vocab,
        max_steps=config.MAX_STEPS,
        mode="multi-step",
        seed=seed,
    )
    env = TextEmbeddingWrapper(base_env, model_name=config.resolve_embedding_model())

    agent = PPOAgent(
        env=env,
        learning_rate=config.LEARNING_RATE,
        batch_size=config.BATCH_SIZE,
        gamma=config.GAMMA,
        seed=seed,
        tensorboard_log=os.path.join(config.LOGS_DIR, "ppo_tensorboard"),
    )

    agent.train(total_timesteps=timesteps, tb_log_name=f"MaskablePPO_seed{seed}")

    model_path = os.path.join(config.CHECKPOINT_DIR, f"ppo_metatool_seed{seed}")
    agent.save(model_path)
    print(f"[seed={seed}] Training finished. Model saved to {model_path}.zip")


def main():
    parser = argparse.ArgumentParser(description="Train CACTUS (MaskablePPO) on the MetaTool env.")
    parser.add_argument("--seed", type=int, default=config.SEED,
                         help="Single seed to train. Ignored if --all_seeds is passed.")
    parser.add_argument("--all_seeds", action="store_true",
                         help=f"Train one model per seed in config.SEEDS={config.SEEDS} sequentially "
                              f"(prefer a SLURM array job on the cluster -- see scripts/train_ppo_cluster.sh).")
    parser.add_argument("--timesteps", type=int, default=config.NUM_TIMESTEPS)
    args = parser.parse_args()

    seeds = config.SEEDS if args.all_seeds else [args.seed]
    for seed in seeds:
        train_one_seed(seed, args.timesteps)


if __name__ == "__main__":
    main()
