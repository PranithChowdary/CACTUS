import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv


def _mask_fn(env):
    return env.unwrapped.action_masks()


class PPOAgent:
    """
    Wraps sb3_contrib.MaskablePPO. The environment must expose an
    `action_masks()` method (see env/metatool_env.py); ActionMasker wires it
    into the policy so invalid (already-selected) tools get zero probability
    mass instead of merely being reward-penalized.
    """

    def __init__(
        self,
        env,
        learning_rate=3e-4,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        seed=None,
        tensorboard_log="./logs/ppo_tensorboard/",
    ):
        self.raw_env = env
        masked_env = ActionMasker(env, _mask_fn)
        self.vec_env = DummyVecEnv([lambda: masked_env])

        self.model = MaskablePPO(
            policy="MultiInputPolicy",
            env=self.vec_env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
            verbose=verbose,
            seed=seed,
            tensorboard_log=tensorboard_log,
        )
        print(f"MaskablePPO agent initialized (seed={seed}).")

    def train(self, total_timesteps=100000, tb_log_name="MaskablePPO"):
        print(f"Starting training for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name)
        print("Training complete.")

    def save(self, path):
        self.model.save(path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.model = MaskablePPO.load(path, env=self.vec_env)
        print(f"Model loaded from {path}")

    def select_action(self, obs):
        """Predict a single action for one (unbatched) observation, honoring
        the current action mask so a loaded model can never emit an illegal
        (already-selected) action even under deterministic decoding.

        The mask is derived directly from obs["selection"] rather than from
        env state, so this works regardless of which environment instance
        produced `obs`.
        """
        done_idx = self.raw_env.unwrapped.done_idx
        mask = np.asarray(obs["selection"]) == 0
        mask[done_idx] = True
        action, _states = self.model.predict(obs, deterministic=True, action_masks=mask)
        return int(action)
