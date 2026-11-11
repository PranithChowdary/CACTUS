import random
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional

# All three baselines are mask-aware: they never re-select an already-chosen
# tool, and each has an explicit rule for calling DONE (index of DONE is the
# last entry of action_list, matching MetaToolEnv.tool_vocab). Without a stop
# rule a baseline is structurally unable to compete with an agent that can
# choose to stop -- see paper review notes on this.


class RandomAgent:
    def __init__(self, action_list: List[str]):
        self.action_list = action_list
        self.done_idx = action_list.index("DONE")

    def select_action(self, prompt: str, exclude_indices: Optional[List[int]] = None) -> int:
        exclude = set(exclude_indices or [])
        valid = [i for i in range(len(self.action_list)) if i not in exclude]
        return random.choice(valid) if valid else self.done_idx


class HeuristicAgent:
    """Greedily picks any not-yet-selected tool whose name appears verbatim
    in the prompt; stops (DONE) once no further tool name matches."""

    def __init__(self, action_list: List[str]):
        self.action_list = action_list
        self.done_idx = action_list.index("DONE")

    def select_action(self, prompt: str, exclude_indices: Optional[List[int]] = None) -> int:
        exclude = set(exclude_indices or [])
        prompt_lower = prompt.lower()
        for idx, tool in enumerate(self.action_list):
            if idx in exclude or tool == "DONE":
                continue
            if tool.lower() in prompt_lower:
                return idx
        return self.done_idx


class DenseRetrievalAgent:
    """
    A strong IR baseline: embeds the prompt and each tool name, and greedily
    selects the highest-similarity not-yet-selected tool until the best
    remaining similarity falls below `stop_threshold`, at which point it
    calls DONE. The threshold is a real hyperparameter and must be tuned on
    the validation set (see calibrate_threshold) -- an untuned or absent stop
    rule silently cripples this baseline's precision by forcing it to keep
    adding tools past the correct set.
    """

    def __init__(self, action_list: List[str], model_name: str = "all-MiniLM-L6-v2", stop_threshold: float = 0.2):
        self.action_list = action_list
        self.model = SentenceTransformer(model_name)
        self.stop_threshold = stop_threshold

        self.tool_names = [t for t in action_list if t != "DONE"]
        self.tool_embeddings = self.model.encode(self.tool_names)
        self.done_idx = action_list.index("DONE")

        print(f"DenseRetrievalAgent initialized with {len(self.tool_names)} tools, stop_threshold={stop_threshold}.")

    def _similarities(self, prompt: str) -> np.ndarray:
        query_emb = self.model.encode(prompt)
        return np.dot(self.tool_embeddings, query_emb) / (
            np.linalg.norm(self.tool_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )

    def select_action(self, prompt: str, exclude_indices: Optional[List[int]] = None) -> int:
        exclude = set(exclude_indices or [])
        similarities = self._similarities(prompt)

        for idx in exclude:
            if idx < len(similarities):
                similarities[idx] = -np.inf

        best_local_idx = int(np.argmax(similarities))
        best_score = similarities[best_local_idx]

        if best_score < self.stop_threshold:
            return self.done_idx

        tool_name = self.tool_names[best_local_idx]
        return self.action_list.index(tool_name)

    def calibrate_threshold(self, val_examples, correct_tools_fn, candidates=None) -> float:
        """
        Sweep candidate thresholds on the validation set and keep the one
        maximizing mean F1, simulating the full stop-aware selection loop
        (not just one-shot top-k). Mutates self.stop_threshold and returns it.

        val_examples: list of prompt strings
        correct_tools_fn: val_examples[i] -> set of correct tool names
        """
        if candidates is None:
            candidates = np.linspace(0.0, 0.6, 25)

        # Precompute similarities once per example so the sweep is cheap.
        all_sims = [self._similarities(p) for p in val_examples]

        best_thresh, best_f1 = self.stop_threshold, -1.0
        for thresh in candidates:
            f1_sum = 0.0
            for prompt, sims, correct in zip(val_examples, all_sims, [correct_tools_fn(v) for v in val_examples]):
                sims = sims.copy()
                selected = set()
                for _ in range(len(self.tool_names)):
                    masked = sims.copy()
                    for i in selected:
                        masked[i] = -np.inf
                    idx = int(np.argmax(masked))
                    if masked[idx] < thresh:
                        break
                    selected.add(idx)
                selected_names = {self.tool_names[i] for i in selected}
                tp = len(selected_names & correct)
                precision = tp / max(1, len(selected_names))
                recall = tp / max(1, len(correct))
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                f1_sum += f1
            mean_f1 = f1_sum / max(1, len(val_examples))
            if mean_f1 > best_f1:
                best_f1, best_thresh = mean_f1, thresh

        self.stop_threshold = float(best_thresh)
        print(f"DenseRetrievalAgent: calibrated stop_threshold={self.stop_threshold:.3f} (val F1={best_f1:.3f})")
        return self.stop_threshold
