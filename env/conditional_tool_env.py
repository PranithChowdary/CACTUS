"""
conditional_tool_env.py

A tool-selection environment with *controllable* execution-dependence, built to
test the one hypothesis that decides whether sequential decision-making is
justified for this task at all.

Motivation
----------
On MetaTool, a logistic-regression probe over frozen query embeddings matches
or beats a masked-PPO agent (SEM 0.588 vs 0.573). That is not a tuning failure:
the gold tool set there is a deterministic function of the query, so the
interaction history carries no information the query does not already provide.
The MDP is degenerate and a memoryless classifier is sufficient by construction.

Sequential policies can only pay off when the agent *gains information by
acting*. This environment injects exactly that, and nothing else, so the
mechanism is unambiguous:

  - A fraction `alpha` of queries are CONDITIONAL. For those, the gold set is
    {probe_tool, X} where X depends on a latent bit c that is NOT recoverable
    from the query embedding. Executing `probe_tool` reveals c.
  - The remaining (1 - alpha) are STATIC: the gold set is determined by the
    query alone, exactly as in MetaTool.

A static classifier cannot exceed chance on the conditional subset's branch
choice, however much data or capacity it is given, because the information is
absent from its input. A sequential policy that probes first, observes c, then
commits can in principle reach 100%. Sweeping alpha therefore traces out the
regime boundary rather than asserting one.

Design notes (decisions a reader should be able to check)
---------------------------------------------------------
- The latent bit is drawn per-episode and is independent of the query, so it is
  information-theoretically unavailable to any query-only model. This makes the
  static ceiling a *provable* bound, not an empirical artifact of undertraining.
- The observation exposes tool results through a dedicated `observation`
  channel, separate from the `selection` bitmask. Without this channel the
  sequential policy would be just as blind as the classifier -- so the channel,
  not the RL algorithm, is the thing under test.
- Action masking is retained from MetaToolEnv: selected tools are masked out,
  bounding trajectories to |A| steps.
"""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Observation channel encoding for the revealed latent bit.
OBS_UNKNOWN = 0
OBS_BRANCH_A = 1
OBS_BRANCH_B = 2


class ConditionalToolEnv(gym.Env):
    """Tool selection where a tunable fraction of queries require probing.

    Args:
        n_tools: size of the tool vocabulary (excluding DONE).
        alpha: fraction of episodes that are conditional (0.0 = pure MetaTool-like
            static task, 1.0 = every episode requires a probe to disambiguate).
        embedding_dim: width of the synthetic query embedding.
        n_query_types: number of distinct latent query types. The embedding is a
            noisy draw around a per-type prototype, so a classifier can learn the
            type but never the per-episode latent bit.
        noise: std of the Gaussian noise added to the prototype embedding.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        n_tools: int = 20,
        alpha: float = 0.5,
        # Must exceed 2 * n_query_types (one prototype per type per condition),
        # or arbitrary labelings of the prototype clusters stop being linearly
        # separable and a LINEAR probe cannot reach the analytic ceiling. That
        # would understate the memoryless baseline for representational rather
        # than informational reasons, making the RL comparison unfair -- at
        # dim=32 with 40 types, conditionality was only 81% predictable.
        embedding_dim: int = 96,
        n_query_types: int = 40,
        noise: float = 0.15,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
        step_penalty: float = 0.05,
        correct_reward: float = 1.0,
        incorrect_reward: float = -0.5,
        exact_match_bonus: float = 2.0,
    ):
        super().__init__()
        self.n_tools = n_tools
        self.alpha = float(alpha)
        self.embedding_dim = embedding_dim
        self.n_query_types = n_query_types
        self.noise = noise

        self.n_actions = n_tools + 1
        self.done_idx = n_tools
        self.max_steps = max_steps if max_steps is not None else self.n_actions

        self.step_penalty = step_penalty
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.exact_match_bonus = exact_match_bonus

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Dict({
            "embedding": spaces.Box(-np.inf, np.inf, shape=(embedding_dim,), dtype=np.float32),
            "selection": spaces.MultiBinary(self.n_actions),
            # Result channel: what executing tools has revealed so far, one-hot
            # over {unknown, branch A, branch B}. One-hot rather than a scalar
            # in [0,2]: as a single scalar among 50+ other input dims the signal
            # is trivially ignorable by the policy net, and an agent that
            # ignores it is indistinguishable from a blind one -- which would
            # make the experiment unable to test its own hypothesis.
            "observation": spaces.Box(0, 1, shape=(3,), dtype=np.float32),
        })

        # Fixed task structure, generated once so every agent and every seed
        # faces the same task family (only the episode draws differ).
        rng = np.random.default_rng(12345)
        # TWO prototypes per query type: index 0 = static, 1 = conditional. The
        # embedding therefore signals WHETHER the task needs a probe (learnable
        # in-distribution) without revealing WHICH branch (never learnable).
        # This keeps the ceiling at (1 - alpha) + alpha/2 while making
        # over-gathering an alpha-shift failure rather than an impossible task.
        self.prototypes = rng.normal(
            0, 1, size=(n_query_types, 2, embedding_dim)).astype(np.float32)

        self.probe_tool = rng.integers(0, n_tools, size=n_query_types)
        self.branch_a = rng.integers(0, n_tools, size=n_query_types)
        self.branch_b = rng.integers(0, n_tools, size=n_query_types)
        # Static gold is a PAIR that excludes the probe, so that calling the
        # probe on a static episode is pure waste. In the first version static
        # gold was {probe, static_partner}: the probe was gold in every episode,
        # probing was never wasteful, and over-gathering -- the headline metric
        # of this project -- was unmeasurable by construction.
        self.static_gold_a = rng.integers(0, n_tools, size=n_query_types)
        self.static_gold_b = rng.integers(0, n_tools, size=n_query_types)

        for t in range(n_query_types):
            while self.branch_a[t] == self.probe_tool[t]:
                self.branch_a[t] = rng.integers(0, n_tools)
            while self.branch_b[t] in (self.probe_tool[t], self.branch_a[t]):
                self.branch_b[t] = rng.integers(0, n_tools)
            while self.static_gold_a[t] == self.probe_tool[t]:
                self.static_gold_a[t] = rng.integers(0, n_tools)
            while self.static_gold_b[t] in (self.probe_tool[t], self.static_gold_a[t]):
                self.static_gold_b[t] = rng.integers(0, n_tools)

        assert embedding_dim > 2 * n_query_types, (
            f"embedding_dim={embedding_dim} must exceed 2*n_query_types="
            f"{2 * n_query_types} so a linear probe can attain the analytic "
            f"ceiling; otherwise the memoryless baseline is handicapped by the "
            f"representation rather than by missing information")

        self._validate()
        self._rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    def _validate(self) -> None:
        """Fail loudly on the design violations that previously went unnoticed."""
        for t in range(self.n_query_types):
            assert self.probe_tool[t] not in (self.static_gold_a[t], self.static_gold_b[t]), (
                f"query type {t}: probe is in static gold -- probing would not be "
                f"wasteful at alpha=0 and over-gathering becomes unmeasurable")
            assert self.branch_a[t] != self.branch_b[t], f"query type {t}: branches must differ"
            assert self.probe_tool[t] not in (self.branch_a[t], self.branch_b[t])
            assert self.static_gold_a[t] != self.static_gold_b[t]

    # ------------------------------------------------------------------ state

    def action_masks(self) -> np.ndarray:
        mask = np.ones(self.n_actions, dtype=bool)
        for i in self.selected:
            mask[i] = False
        mask[self.done_idx] = True
        return mask

    def _get_obs(self) -> Dict[str, Any]:
        sel = np.zeros(self.n_actions, dtype=np.int8)
        for i in self.selected:
            sel[i] = 1
        reveal = np.zeros(3, dtype=np.float32)
        reveal[self.revealed] = 1.0
        return {
            "embedding": self.embedding,
            "selection": sel,
            "observation": reveal,
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "query_type": int(self.query_type),
            "is_conditional": bool(self.is_conditional),
            "latent_bit": int(self.latent_bit),
            "gold": sorted(int(x) for x in self.gold),
            "selected": sorted(int(x) for x in self.selected),
            "step_count": self.step_count,
            "revealed": int(self.revealed),
            "probe": int(self.probe_tool[self.query_type]),
            "probe_is_gold": bool(int(self.probe_tool[self.query_type]) in self.gold),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if options is not None and "query_type" in options:
            self.query_type = int(options["query_type"])
            self.is_conditional = bool(options.get("is_conditional",
                                                    self._rng.random() < self.alpha))
            self.latent_bit = int(options.get("latent_bit", self._rng.integers(0, 2)))
        else:
            self.query_type = int(self._rng.integers(0, self.n_query_types))
            self.is_conditional = bool(self._rng.random() < self.alpha)
            self.latent_bit = int(self._rng.integers(0, 2))

        # Prototype 1 for conditional episodes, 0 for static ones: the query
        # itself says whether a probe is warranted, never which branch.
        proto = self.prototypes[self.query_type][1 if self.is_conditional else 0]
        self.embedding = (proto
                          + self._rng.normal(0, self.noise, self.embedding_dim)).astype(np.float32)

        probe = int(self.probe_tool[self.query_type])
        if self.is_conditional:
            branch = int(self.branch_a[self.query_type] if self.latent_bit == 0
                         else self.branch_b[self.query_type])
            self.gold = {probe, branch}
        else:
            # Probe deliberately absent: calling it here is a wasted step and a
            # wrong tool, which is what makes over-gathering measurable.
            self.gold = {int(self.static_gold_a[self.query_type]),
                         int(self.static_gold_b[self.query_type])}

        self.selected = set()
        self.revealed = OBS_UNKNOWN
        self.step_count = 0
        return self._get_obs(), self._get_info()

    # ------------------------------------------------------------------- step

    def step(self, action: int):
        action = int(action)
        if not self.action_masks()[action]:
            raise ValueError(f"Action {action} is masked (already selected: {sorted(self.selected)})")

        self.step_count += 1
        terminated = False
        truncated = False

        if action == self.done_idx:
            terminated = True
            if self.selected == self.gold:
                reward = self.exact_match_bonus
            else:
                tp = len(self.selected & self.gold)
                p = tp / max(1, len(self.selected))
                r = tp / max(1, len(self.gold))
                reward = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        else:
            self.selected.add(action)
            reward = self.correct_reward if action in self.gold else self.incorrect_reward
            # Executing the probe tool reveals the latent bit. This is the only
            # way the information enters the agent's observation.
            if self.is_conditional and action == int(self.probe_tool[self.query_type]):
                self.revealed = OBS_BRANCH_A if self.latent_bit == 0 else OBS_BRANCH_B

        reward -= self.step_penalty

        if self.step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    # ------------------------------------------------------------- analysis

    def static_ceiling(self) -> float:
        """Best achievable exact-match for ANY query-only (memoryless) model.

        Static and conditional episodes use different query prototypes, so a
        perfect classifier can tell them apart and gets every static episode
        right. On conditional episodes the branch is decided by a latent bit
        drawn independently of the query, so the best a query-only model can do
        is guess one branch: 1/2 in expectation. Hence (1 - alpha) + alpha/2.

        Note this is the ceiling for a model that never probes. An agent that
        probes on STATIC episodes scores below it, because the probe is not in
        the static gold set -- that gap is the over-gathering cost.

        This is an information-theoretic bound, not an empirical baseline --
        it holds regardless of capacity, data, or training budget.
        """
        return (1.0 - self.alpha) + self.alpha * 0.5
