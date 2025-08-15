"""
metatool_env.py

A custom DeepMind-style environment for training and evaluating AI agents on 
tool selection tasks using the MetaTool dataset. This environment is designed 
to work with `dm_env` and PyTorch (optional), supporting both single-shot 
and multi-step prediction modes.

Author: CACTUS Authors
Date: 2025-08-08
"""

import random
import dm_env
import numpy as np
import pandas as pd
from typing import List, Dict, Optional


class MetaToolEnv(dm_env.Environment):
    """
    A DeepMind-style environment for MetaTool tool-selection tasks.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tool_vocab: List[str],
        mode: str = "single-shot",
        seed: Optional[int] = None
    ):
        """
        Initialize the MetaToolEnv.

        Args:
            dataframe (pd.DataFrame): Dataset with columns ["prompt", "tools"].
                                      "tools" is a comma-separated string of correct tool names.
            tool_vocab (list[str]): Global vocabulary of all tools.
            mode (str): "single-shot" or "multi-step".
            seed (int, optional): Random seed for reproducibility.
        """
        assert "prompt" in dataframe.columns, "DataFrame must contain 'prompt' column"
        assert "tools" in dataframe.columns, "DataFrame must contain 'tools' column"
        assert mode in ["single-shot", "multi-step"], "Invalid mode"

        self.df = dataframe
        self.tool_vocab = tool_vocab  # ✅ Fixed: store a global vocabulary
        self.mode = mode
        self.rng = random.Random(seed)

        self.current_index = None
        self.available_tools = None
        self.correct_tools = None
        self.selected_tools = []
        self.done = False

    def reset(self) -> dm_env.TimeStep:
        """
        Reset environment to start a new episode.

        Returns:
            dm_env.TimeStep: A `FIRST` timestep with observation and zero reward.
        """
        self.done = False
        self.selected_tools = []

        self.current_index = self.rng.randint(0, len(self.df) - 1)
        row = self.df.iloc[self.current_index]
        self.prompt = row["prompt"]
        self.correct_tools = [t.strip() for t in row["tools"].split(",")]

        # Only include correct tools + distractors from vocab
        self.available_tools = sorted(
            list(set(self.correct_tools + self._sample_distractors()))
        )

        return dm_env.restart(self._get_observation())

    def step(self, action: int) -> dm_env.TimeStep:
        """
        Take one step in the environment.

        Args:
            action (int): Index of chosen tool in the global vocabulary.

        Returns:
            dm_env.TimeStep: NEXT or LAST timestep.
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        chosen_tool = self.tool_vocab[action]  # Actions are global-vocab based
        reward = 0.0

        if chosen_tool in self.correct_tools and chosen_tool not in self.selected_tools:
            reward = 1.0
            self.selected_tools.append(chosen_tool)

        if self.mode == "single-shot":
            self.done = True
        else:
            if (
                set(self.selected_tools) == set(self.correct_tools)
                or chosen_tool == "DONE"
            ):
                self.done = True
        
        if self.done:
            return dm_env.termination(reward=reward, observation=self._get_observation())
        else:
            return dm_env.transition(reward=reward, observation=self._get_observation())

    def _get_observation(self) -> Dict[str, List[str]]:
        """
        Construct observation dictionary.

        Returns:
            dict: Contains "prompt" (str) and "available_tools" (list of str).
        """
        return {
            "prompt": self.prompt,
            "available_tools": self.available_tools
        }

    def _sample_distractors(self, n: int = 4) -> List[str]:
        """
        Sample random distractor tools from the global vocabulary.

        Args:
            n (int): Number of distractor tools.

        Returns:
            list[str]: Distractor tool names.
        """
        distractors = list(set(self.tool_vocab) - set(self.correct_tools))
        return self.rng.sample(distractors, min(n, len(distractors)))

    def observation_spec(self):
        """
        Define observation spec.
        """
        return {
            "prompt": dm_env.specs.Array(shape=(), dtype=object, name="prompt"),
            "available_tools": dm_env.specs.Array(shape=(None,), dtype=object, name="available_tools")
        }

    def action_spec(self):
        """
        Define action spec.
        """
        return dm_env.specs.DiscreteArray(
            num_values=len(self.tool_vocab),
            dtype=int,
            name="action"
        )


if __name__ == "__main__":
    # Example standalone test
    df = pd.DataFrame({
        "prompt": ["Find academic papers on reinforcement learning."],
        "tools": ["Google Scholar, Arxiv"]
    })
    vocab = ["Google Scholar", "Arxiv", "PubMed", "Mendeley", "ResearchGate"]
    env = MetaToolEnv(df, tool_vocab=vocab, mode="single-shot", seed=42)
    ts = env.reset()
    print("Observation:", ts.observation)
    ts = env.step(0)
    print("Reward:", ts.reward, "Done:", ts.last())