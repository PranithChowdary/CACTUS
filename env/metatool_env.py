"""
metatool_env.py - Gymnasium Edition

A standardized Gymnasium environment for tool-selection RL research.

Key properties (see docs/DESIGN.md and paper/main.tex Sec 3):
- The action space is the tool vocabulary plus a DONE token.
- Once a tool (right or wrong) has been chosen, it is permanently masked out via
  `action_masks()` so it can never be chosen again. This is the mechanism that
  bounds trajectory length to <= |A| and is what CACTUS's paper calls
  "autoregressive action masking" -- it must be enforced by the *policy*
  (see models/ppo_agent.py using sb3_contrib.MaskablePPO), not just exposed here.
- `reset(options={"index": i})` selects a specific example deterministically,
  which evaluation code uses to walk the test set exactly once instead of
  sampling with replacement.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any


class MetaToolEnv(gym.Env):
    """
    MetaTool environment for reinforcement learning.

    The agent observes the current tool selection and the query, and must
    select the correct subset of tools for the given query, then call DONE.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tool_vocab: List[str],
        max_steps: Optional[int] = None,
        mode: str = "multi-step",
        seed: Optional[int] = None,
        step_penalty: float = 0.05,
        correct_reward: float = 1.0,
        incorrect_reward: float = -0.5,
        exact_match_bonus: float = 2.0,
    ):
        super().__init__()

        # Data and Vocab
        self.df = dataframe.reset_index(drop=True)
        self.tool_vocab = list(tool_vocab)
        if "DONE" not in self.tool_vocab:
            self.tool_vocab.append("DONE")

        self.tool_to_idx = {tool: i for i, tool in enumerate(self.tool_vocab)}
        self.done_idx = self.tool_to_idx["DONE"]
        self.n_actions = len(self.tool_vocab)

        # With masking, a trajectory can select at most (n_actions - 1) real
        # tools before DONE becomes the only unmasked action. max_steps is a
        # safety truncation only -- it should never bind in practice, and
        # MUST be identical between training and evaluation.
        self.max_steps = max_steps if max_steps is not None else self.n_actions
        self.mode = mode

        self.step_penalty = step_penalty
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.exact_match_bonus = exact_match_bonus

        # Spaces
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Dict({
            "selection": spaces.MultiBinary(self.n_actions),
            "prompt_id": spaces.Discrete(len(self.df))
        })

        # State
        self.current_idx = None
        self.correct_tool_indices = set()
        self.selected_tool_indices = set()
        self.step_count = 0

        if seed is not None:
            self.reset(seed=seed)

    def _normalize_tools(self, val) -> List[str]:
        if pd.isna(val):
            return []
        if isinstance(val, str):
            return [t.strip() for t in val.replace(";", ",").split(",") if t.strip()]
        if isinstance(val, list):
            return [str(v).strip() for v in val if str(v).strip()]
        return [str(val).strip()]

    def _get_obs(self) -> Dict[str, Any]:
        selection = np.zeros(self.n_actions, dtype=np.int8)
        for idx in self.selected_tool_indices:
            selection[idx] = 1
        return {
            "selection": selection,
            "prompt_id": self.current_idx
        }

    def _get_info(self) -> Dict[str, Any]:
        row = self.df.iloc[self.current_idx]
        return {
            "prompt": row["Query"],
            "correct_tools": [self.tool_vocab[i] for i in self.correct_tool_indices],
            "selected_tools": [self.tool_vocab[i] for i in self.selected_tool_indices],
            "step_count": self.step_count,
            "index": self.current_idx,
        }

    def action_masks(self) -> np.ndarray:
        """
        Required by sb3_contrib.MaskablePPO (via ActionMasker / get_action_masks).
        True = the action is legal right now. A tool that has already been
        chosen (correct or not) is masked out; DONE is always legal.
        """
        mask = np.ones(self.n_actions, dtype=bool)
        for idx in self.selected_tool_indices:
            mask[idx] = False
        mask[self.done_idx] = True
        return mask

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)

        if options is not None and "index" in options:
            # Deterministic selection, used by evaluation to walk the full
            # test set exactly once instead of sampling with replacement.
            self.current_idx = int(options["index"])
            if not (0 <= self.current_idx < len(self.df)):
                raise IndexError(f"index {self.current_idx} out of range for dataframe of length {len(self.df)}")
        else:
            self.current_idx = int(self.np_random.integers(0, len(self.df)))

        row = self.df.iloc[self.current_idx]

        raw_tools = row["New Tools"] if "New Tools" in row and not pd.isna(row["New Tools"]) else row["Tool"]
        correct_tool_names = self._normalize_tools(raw_tools)
        self.correct_tool_indices = {self.tool_to_idx[t] for t in correct_tool_names if t in self.tool_to_idx}

        self.selected_tool_indices = set()
        self.step_count = 0

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        if not self.action_masks()[action]:
            raise ValueError(
                f"Action {action} ('{self.tool_vocab[action]}') is masked out "
                f"(already selected: {[self.tool_vocab[i] for i in self.selected_tool_indices]}). "
                f"A masked policy should never emit this action -- check the mask wiring."
            )

        self.step_count += 1
        terminated = False
        truncated = False

        if action == self.done_idx:
            terminated = True
            if self.selected_tool_indices == self.correct_tool_indices and len(self.correct_tool_indices) > 0:
                reward = self.exact_match_bonus
            else:
                tp = len(self.selected_tool_indices & self.correct_tool_indices)
                precision = tp / max(1, len(self.selected_tool_indices))
                recall = tp / max(1, len(self.correct_tool_indices))
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                reward = f1
        else:
            # Masking guarantees this tool has never been picked before, so
            # every non-DONE step strictly grows the history -- this is the
            # h_{t+1} = h_t OR e_{a_t} update the paper's Eq. 1 describes.
            self.selected_tool_indices.add(action)
            reward = self.correct_reward if action in self.correct_tool_indices else self.incorrect_reward

        reward -= self.step_penalty

        if self.mode == "single-shot":
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def render(self):
        info = self._get_info()
        print(f"Step: {self.step_count}")
        print(f"Query: {info['prompt']}")
        print(f"Selected: {info['selected_tools']}")
        print(f"Correct: {info['correct_tools']}")
