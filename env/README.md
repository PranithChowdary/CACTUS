# MetaTool RL Environment

This folder contains the **custom RL environment** for training and evaluating agents on the **MetaTool** (synthetic version) dataset, implemented using [dm\_env](https://github.com/google-deepmind/dm_env) and PyTorch.

The environment simulates **tool selection** tasks based on natural language prompts, where the agent must choose the most appropriate tool(s) from a given set to solve the task.

---

## 📌 Features

* **Dataset-Driven** — Uses the MetaTool dataset (or synthetic variant with real tool names) to generate episodes.
* **Single-Shot & Multi-Step Modes** —

  * *Single-Shot*: Agent predicts the correct tool(s) in one step.
  * *Multi-Step*: Agent selects tools sequentially until the task is solved or a step limit is reached.
* **Flexible Reward Functions** — Reward can be binary (correct/incorrect) or shaped (partial credit for near matches).
* **Observation Design** — Converts prompts into embeddings for agent input (via LLM encoder or other embedding models).
* **dm\_env Compatible** — Easily pluggable with DeepMind’s RL frameworks and wrappers.


---

## Environment Overview:

- Each episode consists of a "prompt" (user query) and a list of available tools.
- The agent's task is to select the correct tool(s) for the given prompt.
- Supports both:
    1. Single-shot mode: agent predicts all tools in one go.
    2. Multi-step mode: agent selects tools step-by-step until done.

---

## Observation Space:

- observation["prompt"]: str
    Natural language task description.
- observation["available_tools"]: list[str]
    List of tool names available for selection.

---

## Action Space:

- Integer index of the chosen tool in the `available_tools` list.
- Special "DONE" index (optional in multi-step mode) to indicate task completion.

---

## Reward Structure:

- +1.0 if correct tool chosen, else 0.0.
- In multi-step mode: partial rewards possible for each correct selection.



---

## 🛠 Environment API

The environment follows the standard `dm_env.Environment` interface:

```python
import dm_env
from env.metatool_env import MetaToolEnv

env = MetaToolEnv(dataset_path="path/to/dataset.csv", mode="single_shot")

# Reset environment
ts = env.reset()

# Step through environment
while not ts.last():
    action = agent.select_action(ts.observation)
    ts = env.step(action)
```
---

## 📊 Modes

* `single_shot`: Agent predicts tools in one decision step.
* `multi_step`: Agent selects tools sequentially, receiving feedback at each step.

---

## 📈 Evaluation Metrics

* **Accuracy** — % of prompts where the agent picks correct tool(s).
* **Steps to Success** — Avg. number of steps in multi-step mode.
* **Generalization** — Performance on unseen prompts/tools.