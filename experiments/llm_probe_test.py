"""
llm_probe_test.py  --  KILL TEST #1 for the information-gathering research program.

Question: does a prompted LLM ALREADY probe before committing on execution-
dependent tasks? If yes, there is nothing for RL to teach and the program stops.

We measure, at alpha = 1.0 (every episode requires the probe):
  - exact-match             vs the analytic query-only ceiling of 0.500
  - probe-first rate        fraction of episodes whose first call is the probe
  - conditions-on-result    among episodes that probed, fraction that then chose
                            the branch consistent with the result they observed

Interpretation (decided before running):
  probe-first > 0.8 AND exact-match > 0.8   -> the LLM already does it. STOP.
  probe-first > 0.8 AND exact-match ~ 0.5   -> probes but ignores results: a
                                               conditioning problem RL could fix.
  probe-first < 0.5                          -> does not gather information:
                                               the capability RL would need to
                                               teach. Program proceeds.

Also runs alpha = 0.0 to measure OVER-probing: an agent that always probes
even when unnecessary wastes a step on every static query. The field never
measures this, and it is the failure mode SFT-on-optimal-trajectories is
predicted to inherit.

Engineering notes (learned the hard way -- the first run timed out with NO
output despite an hour of work):
  - stdout is line-buffered and every print flushes. A SLURM time-limit kill
    otherwise discards everything in the buffer.
  - results are written to disk after EVERY config, so a killed job still
    leaves the configs it finished.
  - episodes are stepped in lockstep and generation is batched across them,
    instead of one generate() call per step per episode.

Runs zero-shot and few-shot. Requires the LLaMA cache on the cluster:
    sbatch scripts/run_llm_probe_test.sh
"""
import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from env.nl_conditional_env import NLConditionalEnv, SCENARIOS

FEW_SHOT = """Example:
User request: Book me somewhere to stay in Lisbon next week.
Tools called so far and their results:
(none yet)
Next tool: CalendarTool
Tools called so far and their results:
CalendarTool -> You are free Tue-Thu next week.
Next tool: HotelTool
Tools called so far and their results:
CalendarTool -> You are free Tue-Thu next week.
HotelTool -> OK.
Next tool: DONE

"""


def parse_tool(text: str, legal):
    text = text.strip().split("\n")[0].strip().strip("`'\".:")
    for t in legal:
        if text.lower() == t.lower():
            return t
    for t in legal:  # tolerate "Call WeatherTool" style
        if t.lower() in text.lower():
            return t
    return text  # invalid; env will penalize


@torch.no_grad()
def generate_batch(model, tok, prompts, batch_size, max_new_tokens=12, verbose=True):
    """Greedy-decode a list of prompts in batches. Left padding so the
    generated continuation starts at the same column for every row.

    Instrumented per chunk: a run that stalls here produced three hours of
    silence once, because the only progress print was one level up. Never let
    the innermost loop be the unlogged one.
    """
    outputs = []
    n_chunks = (len(prompts) + batch_size - 1) // batch_size
    for ci, i in enumerate(range(0, len(prompts), batch_size)):
        t0 = time.time()
        chunk = prompts[i:i + batch_size]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True)
                 for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        # token_type_ids is not a generate() kwarg on some tokenizer backends.
        enc.pop("token_type_ids", None)
        if verbose:
            print(f"      chunk {ci + 1}/{n_chunks}: {enc['input_ids'].shape[0]}x"
                  f"{enc['input_ids'].shape[1]} tokens -> generating...", flush=True)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        gen = out[:, enc.input_ids.shape[1]:]
        outputs.extend(tok.batch_decode(gen, skip_special_tokens=True))
        if verbose:
            print(f"      chunk {ci + 1}/{n_chunks}: done in {time.time() - t0:.1f}s",
                  flush=True)
    return outputs


def run(model, tok, alpha, n_episodes, few_shot, seed, batch_size):
    # One env per episode so they can be stepped in lockstep. Scenarios are
    # assigned round-robin so every scenario gets equal coverage.
    envs = [NLConditionalEnv(alpha=alpha, seed=seed * 100_000 + i) for i in range(n_episodes)]
    obs = []
    infos = []
    for i, e in enumerate(envs):
        o, inf = e.reset(options={"scenario": i % len(SCENARIOS)})
        obs.append(o)
        infos.append(inf)

    first = [None] * n_episodes
    done = [False] * n_episodes
    t0 = time.time()
    step = 0
    while not all(done):
        active = [i for i in range(n_episodes) if not done[i]]
        prompts = [(FEW_SHOT + obs[i]) if few_shot else obs[i] for i in active]
        raws = generate_batch(model, tok, prompts, batch_size)
        for i, raw in zip(active, raws):
            action = parse_tool(raw, envs[i].legal())
            if first[i] is None:
                first[i] = action
            obs[i], r, d, infos[i] = envs[i].step(action)
            done[i] = d
        step += 1
        print(f"    step {step}: {len(active)} active episodes, "
              f"{time.time() - t0:.0f}s elapsed", flush=True)

    sems, probe_first, cond_ok = [], [], []
    over_probe, wasted_steps, static_sem = [], [], []
    for i in range(n_episodes):
        info = infos[i]
        sel, gold = set(info["selected"]), set(info["gold"])
        sems.append(1.0 if sel == gold else 0.0)
        if info["is_conditional"]:
            probe_first.append(1.0 if first[i] == info["probe"] else 0.0)
            if info["probe"] in info["selected"]:
                cond_ok.append(1.0 if gold <= sel else 0.0)
        else:
            # OVER-GATHERING. On a static episode the probe is NOT in gold, so
            # calling it is pure waste: a step spent and a wrong tool selected.
            # The earlier version asked whether any BRANCH tool was selected,
            # which was meaningless because branch[0] was itself the static
            # partner -- it scored correct behaviour as a failure (always 1.000).
            over_probe.append(1.0 if info["probe"] in sel else 0.0)
            wasted_steps.append(float(len(sel - gold)))
            static_sem.append(1.0 if sel == gold else 0.0)

    def m(x):
        return float(np.mean(x)) if x else float("nan")

    return {
        "alpha": alpha, "n": n_episodes, "few_shot": few_shot,
        "static_ceiling": envs[0].static_ceiling(),
        "exact_match": m(sems),
        "probe_first": m(probe_first),
        "conditions_on_result": m(cond_ok),
        # Headline over-gathering metrics, defined only on static episodes.
        "over_probe_rate": m(over_probe),
        "wasted_tools_per_static_episode": m(wasted_steps),
        "static_exact_match": m(static_sem),
        "wall_seconds": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--episodes", type=int, default=140)  # 20 per scenario
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", default="results/llm_probe_test.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto", local_files_only=True)
    model.eval()
    print(f"model loaded in {time.time() - t0:.0f}s on {model.device}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results = []
    for few_shot in (False, True):
        for alpha in (1.0, 0.0):
            tag = "few-shot " if few_shot else "zero-shot"
            print(f"\n[{tag}] alpha={alpha:.1f}  running {args.episodes} episodes...", flush=True)
            r = run(model, tok, alpha, args.episodes, few_shot, seed=0, batch_size=args.batch_size)
            results.append(r)
            print(f"[{tag}] alpha={alpha:.1f}  ceiling={r['static_ceiling']:.3f}  "
                  f"SEM={r['exact_match']:.3f}  probe_first={r['probe_first']:.3f}  "
                  f"cond_on_result={r['conditions_on_result']:.3f}  "
                  f"over_probe={r['over_probe_rate']:.3f}  "
                  f"wasted/ep={r['wasted_tools_per_static_episode']:.2f}  "
                  f"({r['wall_seconds']:.0f}s)", flush=True)
            # Persist after every config so a time-limit kill keeps what finished.
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)

    a1 = [r for r in results if r["alpha"] == 1.0 and r["few_shot"]][0]
    print("\n" + "=" * 64)
    if a1["probe_first"] > 0.8 and a1["exact_match"] > 0.8:
        print("VERDICT: prompted LLM already gathers information reliably.")
        print("         KILL CRITERION #1 FIRES -- nothing for RL to teach here.")
    elif a1["probe_first"] > 0.8:
        print("VERDICT: LLM probes but does not act on the result.")
        print("         Conditioning failure -- a target RL could plausibly fix.")
    else:
        print("VERDICT: LLM does not gather information before committing.")
        print("         This is the capability the program sets out to teach. PROCEED.")
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
