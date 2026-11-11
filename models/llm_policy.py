"""
llm_policy.py

A constrained LLM policy: instead of free-form generation, the model scores
every legal action and the argmax is taken. This makes the LLM a policy over
exactly the same discrete action set as the masked PPO agent, so the two are
comparable rather than one being handicapped by its interface.

Why constrained decoding is necessary here
------------------------------------------
Under free generation, LLaMA-3.1-8B emitted DONE in 0 of 240 turns. Scoring the
first token showed DONE ranked ~31st of ~35 candidates with a 13.5-nat gap to
the best tool: the stop action carries ~1e-6 of the mass of the best tool. It is
not losing the argmax, it is absent from the distribution. No prompt wording
recovers that, so the action set must be imposed rather than hoped for.

The scoring problem: surface-form competition
---------------------------------------------
Picking argmax_a log p(a | prompt) does NOT measure "which action fits the task".
It also measures how probable the *string* is a priori. "DONE" is 1-2 tokens;
"CarRentalTool" is several. Summed log-probability structurally favours short,
frequent strings; mean log-probability over-corrects and favours long ones.
This is surface-form competition (Holtzman et al., "Surface Form Competition:
Why the Highest Probability Answer Isn't Always Right", EMNLP 2021).

We therefore compute three scores and report all of them, so the choice is an
auditable decision rather than a hidden one:

  sum   : log p(a | prompt)                      -- the true generation
                                                    probability; short-biased
  mean  : log p(a | prompt) / |a|                -- length-normalised
  pmi   : log p(a | prompt) - log p(a | premise) -- domain-conditional PMI.
          Divides out the model's prior over the surface form, leaving how much
          the *task context* raises that action. This is the primary score.

`premise` is a content-free stand-in for the prompt that preserves the answer
format but removes the request and the tool catalogue, so p(a | premise) is the
model's prior over emitting that string in this slot.

Efficiency
----------
All candidates share a prefix, so each step is ONE batched forward pass over
[prompt + candidate] sequences (plus one cached pass for the premise), not one
pass per candidate. Right-padding is safe for scoring under causal attention:
padding sits after every scored position and cannot influence it.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

DEFAULT_PREMISE = (
    "You are a tool-selection agent. Reply with exactly one tool name, "
    "or the single word DONE.\n\nNext:"
)


@dataclass
class ActionScores:
    """Per-candidate scores plus everything needed to audit the decision."""
    sum_logprob: Dict[str, float] = field(default_factory=dict)
    mean_logprob: Dict[str, float] = field(default_factory=dict)
    pmi: Dict[str, float] = field(default_factory=dict)
    prior_logprob: Dict[str, float] = field(default_factory=dict)
    n_tokens: Dict[str, int] = field(default_factory=dict)

    def best(self, scoring: str = "pmi") -> str:
        table = getattr(self, {"pmi": "pmi", "sum": "sum_logprob",
                               "mean": "mean_logprob"}[scoring])
        return max(table, key=table.get)

    def rank_of(self, action: str, scoring: str = "pmi") -> int:
        table = getattr(self, {"pmi": "pmi", "sum": "sum_logprob",
                               "mean": "mean_logprob"}[scoring])
        if action not in table:
            return -1
        return sum(1 for v in table.values() if v > table[action]) + 1

    def distribution(self, scoring: str = "pmi") -> Dict[str, float]:
        """Softmax over the legal actions only -- the masked policy's actual
        distribution, directly comparable to the RL agent's masked softmax."""
        table = getattr(self, {"pmi": "pmi", "sum": "sum_logprob",
                               "mean": "mean_logprob"}[scoring])
        keys = list(table)
        t = torch.tensor([table[k] for k in keys])
        p = torch.softmax(t, dim=0)
        return {k: float(v) for k, v in zip(keys, p)}


class ConstrainedLLMPolicy:
    def __init__(self, model, tokenizer, scoring: str = "pmi",
                 premise: str = DEFAULT_PREMISE, batch_size: int = 48,
                 use_chat_template: bool = True):
        if scoring not in ("pmi", "sum", "mean"):
            raise ValueError(f"scoring must be pmi|sum|mean, got {scoring!r}")
        self.model = model
        self.tok = tokenizer
        self.scoring = scoring
        self.premise = premise
        self.batch_size = batch_size
        self.use_chat_template = use_chat_template
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self._prior_cache: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------- internals

    def _prefix_ids(self, prompt: str) -> List[int]:
        if self.use_chat_template:
            text = self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        return self.tok(text, add_special_tokens=not self.use_chat_template)["input_ids"]

    def _cand_ids(self, candidate: str) -> List[int]:
        # Leading space matters for BPE: the model emits " DONE" after a colon,
        # not "DONE". Scoring the wrong variant silently penalises every
        # candidate by a different amount.
        return self.tok(candidate, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def _score(self, prefix_ids: Sequence[int], candidates: Sequence[str]) -> Dict[str, float]:
        """Sum of token log-probabilities of each candidate given the prefix."""
        device = self.model.device
        pad_id = self.tok.pad_token_id
        out: Dict[str, float] = {}

        for i in range(0, len(candidates), self.batch_size):
            chunk = list(candidates[i:i + self.batch_size])
            cand_ids = [self._cand_ids(c) for c in chunk]
            seqs = [list(prefix_ids) + c for c in cand_ids]
            maxlen = max(len(s) for s in seqs)

            input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for r, s in enumerate(seqs):
                input_ids[r, :len(s)] = torch.tensor(s, dtype=torch.long)
                attn[r, :len(s)] = 1
            input_ids, attn = input_ids.to(device), attn.to(device)

            logits = self.model(input_ids=input_ids, attention_mask=attn).logits

            # Only the positions that PREDICT a candidate token are needed:
            # token j of a candidate sits at sequence position start+j and is
            # predicted from start+j-1. Candidates are 1-5 tokens, so that is a
            # handful of positions out of ~700.
            #
            # Previously this did log_softmax(logits.float()) over the WHOLE
            # [B, T, V] tensor. On Llama-3.1-70B (V=128256, B=35, T~700) that is
            # a 12.6 GB float32 allocation and the job died with
            # "Tried to allocate 10.86 GiB" AFTER passing every verification
            # check. Slicing first makes it ~90 MB and is numerically identical.
            start = len(prefix_ids)
            max_cand = max(len(c) for c in cand_ids)
            sl = logits[:, start - 1:start - 1 + max_cand, :].float()
            logZ = torch.logsumexp(sl, dim=-1)          # [B, max_cand]

            for r, (c, cid) in enumerate(zip(chunk, cand_ids)):
                total = 0.0
                for j, tok_id in enumerate(cid):
                    total += float(sl[r, j, tok_id] - logZ[r, j])
                out[c] = total
            del logits, sl, logZ
        return out

    def _priors(self, candidates: Sequence[str]) -> Dict[str, float]:
        key = "|".join(sorted(candidates))
        if key not in self._prior_cache:
            self._prior_cache[key] = self._score(self._prefix_ids(self.premise), candidates)
        return self._prior_cache[key]

    # ---------------------------------------------------------------- public

    def score_actions(self, prompt: str, candidates: Sequence[str]) -> ActionScores:
        prefix = self._prefix_ids(prompt)
        cond = self._score(prefix, candidates)
        prior = self._priors(candidates)
        s = ActionScores()
        for c in candidates:
            n = max(1, len(self._cand_ids(c)))
            s.sum_logprob[c] = cond[c]
            s.mean_logprob[c] = cond[c] / n
            s.prior_logprob[c] = prior[c]
            s.pmi[c] = cond[c] - prior[c]
            s.n_tokens[c] = n
        return s

    def select(self, prompt: str, candidates: Sequence[str],
               scoring: Optional[str] = None):
        """Returns (chosen_action, ActionScores). Deterministic: pure argmax,
        no sampling, so a rerun reproduces exactly."""
        scoring = scoring or self.scoring
        scores = self.score_actions(prompt, candidates)
        return scores.best(scoring), scores
