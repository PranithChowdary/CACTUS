"""
nl_gym_env.py

Gymnasium adapter that lets masked PPO act on NLConditionalEnv -- the SAME
natural-language task the constrained LLM policy plays.

Why this exists
---------------
The LLM results and the RL results were measured on two different
environments: the LLM on NLConditionalEnv (natural-language queries, 36 tools)
and PPO on ConditionalToolEnv (synthetic Gaussian prototypes, 20 tools). The
headline claim -- "instruct LLMs cannot terminate; RL learns to" -- was
therefore comparing a text policy on one task against a PPO policy on another.
A reviewer would answer that the PPO agent stops because its reward pays an
exact-match bonus precisely for stopping on a complete set, a signal the LLM
never saw, on a task the LLM never saw. That reduces a capability claim to a
training-signal claim across two toy worlds.

With this adapter both policies face one task: same 61 scenarios, same 36 tools
plus DONE, same gold sets, same action masking, same episode budget. What
differs is only the policy class, which is what the paper is about.

The result channel is TEXT, not a pre-parsed bit
------------------------------------------------
ConditionalToolEnv hands the agent a 3-way one-hot {unknown, branch A, branch
B}: the latent bit, already decoded. Carrying that over would quietly advantage
RL, because the LLM has to read the probe's sentence ("Denver: severe winter
storm warning...") and work out what it implies, while the agent would be
handed the answer.

So by default (`result_mode="embedding"`) the agent receives the MiniLM
embedding of the probe's actual result text -- the same information the LLM
reads, in the same modality, differing only in encoder. `result_mode="onehot"`
reproduces the old pre-decoded channel and exists as an ablation: it separates
"cannot use the revealed bit" from "cannot parse the result sentence".

Efficiency
----------
Every query and every probe result is a fixed string drawn from the bank, so
the full set of texts is finite (2 per scenario, plus 2 results per scenario,
plus one static-probe reply). They are embedded ONCE and cached to
data/nl_embeddings.npz, keyed by a hash of the text set so a bank edit
invalidates it. Training then does dictionary lookups only -- no transformer
call in the 200k-step loop, and no transformers import at all once the cache
exists.
"""

import hashlib
import os
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.nl_conditional_env import NLConditionalEnv
from env.nl_scenarios import SCENARIOS, TOOLS

EMB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "models", "embedding", "all-MiniLM-L6-v2")
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "nl_embeddings.npz")

# Fixed action ordering. TOOLS is insertion-ordered and DONE is its last entry,
# so DONE is the final index -- mirroring ConditionalToolEnv.done_idx.
ACTIONS: List[str] = list(TOOLS.keys())
DONE_IDX = ACTIONS.index("DONE")
assert DONE_IDX == len(ACTIONS) - 1, "DONE must be the last action index"

STATIC_PROBE_REPLY = "Nothing unusual to report."
EMB_DIM = 384

# Result-channel encoding for result_mode="onehot".
OBS_UNKNOWN, OBS_BRANCH_A, OBS_BRANCH_B = 0, 1, 2


def _all_texts() -> List[str]:
    """Every string that can ever appear in an observation channel."""
    t = []
    for s in SCENARIOS:
        t += [s.cond_query, s.static_query, s.result[0], s.result[1]]
    t.append(STATIC_PROBE_REPLY)
    return sorted(set(t))


def _fingerprint(texts: List[str]) -> str:
    h = hashlib.sha256()
    for x in texts:
        h.update(x.encode("utf-8")); h.update(b"\0")
    return h.hexdigest()[:16]


def load_embeddings(verbose: bool = True) -> Dict[str, np.ndarray]:
    """Text -> MiniLM embedding (mean-pooled, L2-normalised), cached on disk.

    The cache is keyed by a fingerprint of the text set, so editing the
    scenario bank forces a rebuild instead of silently serving stale vectors
    for queries that no longer exist.
    """
    texts = _all_texts()
    fp = _fingerprint(texts)
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=False)
        if str(z["fingerprint"]) == fp:
            if verbose:
                print(f"[nl_gym_env] embeddings from cache ({len(texts)} texts)")
            return {t: v for t, v in zip(texts, z["vectors"])}
        elif verbose:
            print("[nl_gym_env] scenario bank changed -- rebuilding embedding cache")

    if verbose:
        print(f"[nl_gym_env] embedding {len(texts)} texts with MiniLM-L6-v2 ...")
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMB_DIR, local_files_only=True)
    mod = AutoModel.from_pretrained(EMB_DIR, local_files_only=True).eval()
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            b = tok(texts[i:i + 32], padding=True, truncation=True,
                    max_length=256, return_tensors="pt")
            h = mod(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            vecs.append(torch.nn.functional.normalize(pooled, p=2, dim=1).numpy())
    V = np.concatenate(vecs).astype(np.float32)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, vectors=V, fingerprint=np.array(fp))
    if verbose:
        print(f"[nl_gym_env] wrote {CACHE}")
    return {t: v for t, v in zip(texts, V)}


class NLGymEnv(gym.Env):
    """Masked-PPO-compatible view of NLConditionalEnv.

    Args:
        alpha: fraction of episodes that are conditional.
        result_mode: "embedding" (default, faithful -- the agent reads the same
            sentence the LLM reads) or "onehot" (pre-decoded latent bit, as an
            ablation).
        scenario_ids: restrict episodes to these scenarios. Used for the
            held-out split; None means the whole bank, which is the headline
            configuration.
    """

    metadata = {"render_modes": []}

    def __init__(self, alpha: float = 0.5, seed: Optional[int] = None,
                 result_mode: str = "embedding", max_steps: int = 5,
                 scenario_ids: Optional[List[int]] = None,
                 embeddings: Optional[Dict[str, np.ndarray]] = None,
                 verbose: bool = False):
        super().__init__()
        if result_mode not in ("embedding", "onehot"):
            raise ValueError(f"result_mode must be embedding|onehot, got {result_mode!r}")
        self.result_mode = result_mode
        self.alpha = float(alpha)
        self.max_steps = max_steps
        self.scenario_ids = (list(range(len(SCENARIOS))) if scenario_ids is None
                             else list(scenario_ids))
        # Shared across envs in a vectorised stack: embedding every text once
        # per worker would be wasteful and is avoidable.
        self.emb = embeddings if embeddings is not None else load_embeddings(verbose)
        self.inner = NLConditionalEnv(alpha=alpha, seed=seed, max_steps=max_steps)
        self._rng = np.random.default_rng(seed)

        self.n_actions = len(ACTIONS)
        self.done_idx = DONE_IDX
        self.action_space = spaces.Discrete(self.n_actions)
        chans = {
            "query": spaces.Box(-np.inf, np.inf, shape=(EMB_DIM,), dtype=np.float32),
            "selection": spaces.MultiBinary(self.n_actions),
        }
        if result_mode == "embedding":
            # Zeros is an unambiguous "nothing revealed": MiniLM outputs are
            # L2-normalised, so a genuine embedding can never be the zero
            # vector. has_result is still passed explicitly rather than left
            # implicit, because a policy net should not have to infer
            # "all-zeros means unknown" from the norm.
            chans["result"] = spaces.Box(-np.inf, np.inf, shape=(EMB_DIM,), dtype=np.float32)
            chans["has_result"] = spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        else:
            chans["result"] = spaces.Box(0, 1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict(chans)

    # ------------------------------------------------------------------ state

    def action_masks(self) -> np.ndarray:
        """Selected tools are illegal; DONE is always legal. Identical rule to
        ConditionalToolEnv and to the legal set the LLM policy is scored over."""
        mask = np.ones(self.n_actions, dtype=bool)
        for t in self.inner.selected:
            mask[ACTIONS.index(t)] = False
        mask[self.done_idx] = True
        return mask

    def _probe_result_text(self) -> Optional[str]:
        """The probe's reply, if the probe has been executed this episode.

        Derived from the inner env's own state rather than from its transcript
        string, so no parsing and no risk of drifting out of sync with it.
        """
        s = self.inner.s
        if s.probe not in self.inner.selected:
            return None
        return s.result[self.inner.bit] if self.inner.is_conditional else STATIC_PROBE_REPLY

    def _get_obs(self) -> Dict[str, np.ndarray]:
        sel = np.zeros(self.n_actions, dtype=np.int8)
        for t in self.inner.selected:
            sel[ACTIONS.index(t)] = 1
        obs = {"query": self.emb[self.inner.query].astype(np.float32), "selection": sel}
        res = self._probe_result_text()
        if self.result_mode == "embedding":
            obs["result"] = (np.zeros(EMB_DIM, dtype=np.float32) if res is None
                             else self.emb[res].astype(np.float32))
            obs["has_result"] = np.array([0.0 if res is None else 1.0], dtype=np.float32)
        else:
            oh = np.zeros(3, dtype=np.float32)
            if res is None or not self.inner.is_conditional:
                oh[OBS_UNKNOWN] = 1.0
            else:
                oh[OBS_BRANCH_A if self.inner.bit == 0 else OBS_BRANCH_B] = 1.0
            obs["result"] = oh
        return obs

    @property
    def n_query_types(self) -> int:
        """Alias so experiments written against ConditionalToolEnv work here."""
        return len(self.scenario_ids)

    def _get_info(self) -> Dict:
        info = dict(self.inner.info())
        # `step_count` and `query_type` mirror ConditionalToolEnv's info keys so
        # that experiments/alpha_shift.py drives either environment unchanged
        # rather than needing a duplicated copy of its gate and report logic.
        info["step_count"] = info["steps"]
        info["query_type"] = info["scenario"]
        info["gold_idx"] = sorted(ACTIONS.index(t) for t in info["gold"])
        info["selected_idx"] = [ACTIONS.index(t) for t in info["selected"]]
        info["probe_idx"] = ACTIONS.index(info["probe"])
        return info

    # ------------------------------------------------------------------ gym API

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.inner.rng = np.random.default_rng(seed)
        options = dict(options or {})
        # Accept ConditionalToolEnv's key name for the same concept.
        if "query_type" in options and "scenario" not in options:
            options["scenario"] = int(options.pop("query_type"))
        options.pop("query_type", None)
        if "scenario" not in options:
            options["scenario"] = int(self._rng.choice(self.scenario_ids))
        if "is_conditional" not in options:
            options["is_conditional"] = bool(self._rng.random() < self.alpha)
        if "latent_bit" not in options:
            options["latent_bit"] = int(self._rng.integers(0, 2))
        self.inner.reset(options=options)
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        action = int(action)
        if not self.action_masks()[action]:
            raise ValueError(
                f"action {action} ({ACTIONS[action]}) is masked; "
                f"already selected {self.inner.selected}")
        # The inner env returns the 4-tuple (obs, reward, done, info) and folds
        # "chose DONE" and "ran out of steps" into one flag. Gymnasium needs
        # them apart: conflating them would make a truncated episode look like a
        # deliberate stop, which is the exact behaviour under study.
        _, reward, _, _ = self.inner.step(ACTIONS[action])
        terminated = action == self.done_idx
        truncated = (not terminated) and self.inner.steps >= self.max_steps
        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def static_ceiling(self) -> float:
        return self.inner.static_ceiling()


def make_vec_compatible(alpha=0.5, seed=None, result_mode="embedding",
                        scenario_ids=None, embeddings=None):
    """Factory for DummyVecEnv/ActionMasker, with the embedding table shared."""
    emb = embeddings if embeddings is not None else load_embeddings(verbose=False)

    def _make():
        return NLGymEnv(alpha=alpha, seed=seed, result_mode=result_mode,
                        scenario_ids=scenario_ids, embeddings=emb)
    return _make
