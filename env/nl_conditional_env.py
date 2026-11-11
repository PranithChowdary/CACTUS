"""
nl_conditional_env.py

Natural-language tool selection with a tunable fraction `alpha` of tasks that
genuinely require gathering information before acting. Queries, tool
descriptions and tool results are text, so an LLM can be the policy directly.

Design requirements (each one was a bug found in review -- see RESEARCH_PLAN.md)
-------------------------------------------------------------------------------
1. **At alpha=0 the probe is NOT in the gold set.** The headline metric of this
   project is over-gathering: an agent that probes when probing is unnecessary
   wastes a step and forfeits exact match. In the first version the probe was
   gold in *every* episode, which made over-gathering unmeasurable by
   construction and made `unneeded_branch_rate` report 1.000 for behaviour that
   was actually correct.

2. **The query signals whether the task is conditional, never which branch.**
   Static and conditional episodes have different queries, so "should I probe?"
   is learnable in-distribution; "which branch?" is not recoverable from text at
   all. Over-gathering is therefore a *transfer* failure under an alpha-shift,
   not an impossible task. The ceiling for any query-only policy stays
   (1 - alpha) + alpha/2: it gets the static episodes right and must guess the
   branch on conditional ones.

3. **The latent bit is drawn per-episode, independent of the query.** It is
   information-theoretically unavailable to any model that does not execute the
   probe (verified: a classifier predicting it from the query scores ~0.5).

4. **The bank is large enough that greedy decoding does not collapse the sample
   size.** With 7 scenarios every metric was a multiple of 1/7 regardless of how
   many episodes were run.

Episode structure
-----------------
  conditional (probability alpha):
      query   = scenario.cond_query          (signals "it depends")
      gold    = {probe, branch[bit]}
      probing reveals the bit; not probing means guessing between two branches
  static (probability 1 - alpha):
      query   = scenario.static_query        (fully specified)
      gold    = scenario.static_gold         (two tools, PROBE NOT INCLUDED)
      probing is pure waste: a step spent and a wrong tool selected
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

# The scenario bank lives in its own module: it grew from 12 to 61 scenarios to
# fix the n_effective collapse (a deterministic policy makes every repeat of a
# scenario identical, so 48 episodes over 12 scenarios gave n=12 and every rate
# came out as a multiple of 1/12). Re-exported here so that existing
# `from env.nl_conditional_env import NLConditionalEnv, SCENARIOS` keeps working.
from env.nl_scenarios import (  # noqa: F401
    Scenario,
    TOOLS,
    SCENARIOS,
    validate_bank,
    split,
)

validate_bank()


class NLConditionalEnv:
    """Text-observation tool selection with a tunable information-need rate."""

    def __init__(self, alpha: float = 0.5, seed: Optional[int] = None, max_steps: int = 5):
        self.alpha = float(alpha)
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.tools = list(TOOLS.keys())

    def reset(self, options: Optional[Dict] = None):
        options = options or {}
        self.s_idx = int(options.get("scenario", self.rng.integers(0, len(SCENARIOS))))
        self.s = SCENARIOS[self.s_idx]
        self.is_conditional = bool(options.get("is_conditional", self.rng.random() < self.alpha))
        self.bit = int(options.get("latent_bit", self.rng.integers(0, 2)))

        if self.is_conditional:
            self.query = self.s.cond_query
            self.gold = {self.s.probe, self.s.branch[self.bit]}
        else:
            self.query = self.s.static_query
            self.gold = set(self.s.static_gold)

        self.selected: List[str] = []
        self.transcript: List[str] = []
        self.steps = 0
        return self.observation(), self.info()

    def observation(self) -> str:
        """Prompt layout matters more than it looks.

        The first version buried DONE as the last of 36 catalogue entries and
        showed the already-called tools only as an implicit transcript. Against
        LLaMA-3.1-8B that produced DONE in 0/24 episodes: the model filled every
        remaining turn by re-naming tools it had already called, which then
        failed to parse because selected tools are removed from the legal set.
        The parse failures were a symptom of the missing stop, not a separate
        bug. So: the stop action is stated separately from the catalogue, the
        already-called set is made explicit with a do-not-repeat instruction,
        and the remaining budget is shown.
        """
        catalogue = "\n".join(f"  - {n}: {d}" for n, d in TOOLS.items()
                              if n not in self.selected and n != "DONE")
        if self.transcript:
            called = "\n".join(f"  - {line}" for line in self.transcript)
        else:
            called = "  (none yet)"
        remaining = self.max_steps - self.steps
        return (
            "You are a tool-selection agent. Call the tools the request needs, then stop.\n\n"
            f"User request: {self.query}\n\n"
            f"Tools already called -- do NOT call any of these again:\n{called}\n\n"
            f"Tools you may call now:\n{catalogue}\n\n"
            "Reply with EXACTLY ONE of:\n"
            "  - the name of one tool from the list above, or\n"
            "  - the single word DONE, if the request is already fully served.\n"
            f"You have {remaining} turn(s) left. Answer with the word only, nothing else.\n\n"
            "Next:"
        )

    def info(self) -> Dict:
        return {
            "scenario": self.s_idx,
            "is_conditional": self.is_conditional,
            "latent_bit": self.bit,
            "gold": sorted(self.gold),
            "selected": list(self.selected),
            "steps": self.steps,
            "probe": self.s.probe,
            "probe_is_gold": self.s.probe in self.gold,
        }

    def legal(self) -> List[str]:
        return [t for t in self.tools if t not in self.selected]

    def step(self, tool: str):
        tool = tool.strip()
        self.steps += 1
        if tool not in self.legal():
            self.transcript.append(f"[invalid action: {tool!r}]")
            return self.observation(), -0.5, self.steps >= self.max_steps, self.info()

        if tool == "DONE":
            sel = set(self.selected)
            reward = 2.0 if sel == self.gold else self._f1(sel)
            return self.observation(), reward, True, self.info()

        self.selected.append(tool)
        if tool == self.s.probe:
            # The probe reveals the bit only when there is a bit to reveal.
            result = self.s.result[self.bit] if self.is_conditional else "Nothing unusual to report."
        else:
            result = "OK."
        self.transcript.append(f"{tool} -> {result}")
        reward = (1.0 if tool in self.gold else -0.5) - 0.05
        return self.observation(), reward, self.steps >= self.max_steps, self.info()

    def _f1(self, sel):
        tp = len(sel & self.gold)
        p = tp / max(1, len(sel))
        r = tp / max(1, len(self.gold))
        return 2 * p * r / (p + r) if p + r > 0 else 0.0

    def static_ceiling(self) -> float:
        """Best exact-match achievable by any policy that never executes the probe.

        Static episodes are fully specified by the query, so a perfect query-only
        policy gets them all. Conditional episodes hinge on a bit that is absent
        from the query, so the best it can do is guess one branch: 1/2.
        """
        return (1.0 - self.alpha) + self.alpha * 0.5
