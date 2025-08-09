"""
baseline_agents.py
---------
Defines baseline agents for the MetaTool environment.

Includes:
    - RandomAgent: Chooses a random tool from the ACTION_LIST.
    - HeuristicAgent: Matches keywords from prompt to tools.
"""

import random
from env.config import ACTION_LIST


class RandomAgent:
    """
    Baseline agent that randomly selects a tool from the ACTION_LIST.

    Methods:
        select_action(prompt: str) -> int:
            Returns the index of a randomly chosen tool.
    """
    def __init__(self, action_list=None):
        self.action_list = action_list or ACTION_LIST

    def select_action(self, prompt: str) -> int:
        return random.randint(0, len(self.action_list) - 1)


class HeuristicAgent:
    """
    Simple keyword-matching agent.

    For each tool in ACTION_LIST, if its name appears in the prompt,
    it selects that tool; otherwise, it falls back to random choice.
    """
    def __init__(self, action_list=None):
        self.action_list = action_list or ACTION_LIST

    def select_action(self, prompt: str) -> int:
        prompt_lower = prompt.lower()
        for idx, tool in enumerate(self.action_list):
            if tool.lower() in prompt_lower:
                return idx
        return random.randint(0, len(self.action_list) - 1)


if __name__ == "__main__":
    # Quick test
    random_agent = RandomAgent()
    heuristic_agent = HeuristicAgent()

    prompt = "I need to analyze this data with a spreadsheet."
    print(f"[RandomAgent] Chosen action: {random_agent.select_action(prompt)}")
    print(f"[HeuristicAgent] Chosen action: {heuristic_agent.select_action(prompt)}")