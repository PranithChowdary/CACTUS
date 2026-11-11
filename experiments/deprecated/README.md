# Deprecated experiment scripts

These predate the current pipeline and **do not run against the current code**.
Nothing in `scripts/*.sh` calls them. They are kept only for reference — delete
them once you're sure nothing here is worth reviving.

Use instead:

| Task | Current script |
|---|---|
| Train CACTUS (all seeds) | `experiments/train_ppo.py` |
| Evaluate everything + significance test | `experiments/evaluate_all.py` |
| LLaMA-3.1 zero-shot baseline | `experiments/evaluate_llm_baseline.py` |

## Why they were retired

`MetaToolEnv` now enforces action masking: once a tool is selected it is masked
out, and `step()` **raises** if a masked action is submitted. These scripts call
`agent.select_action(prompt)` without passing `exclude_indices`, so the agent can
re-emit an already-selected tool and the episode crashes.

They also carry the methodological bugs that the rewrite fixed:

- `env.reset()` with no index — samples test episodes **with replacement**, so
  agents are compared on different, non-matched subsets of the test set.
- `max_steps=5` hardcoded, which disagreed with the `max_steps=10` used during
  training (`evaluate_baselines.py:51`).
- Single run, single seed — no variance, no confidence intervals, no
  significance test.
- Baselines with no stopping rule, which structurally prevents them from
  competing against an agent that can choose to terminate.

`evaluate_all.py` addresses all of the above.
