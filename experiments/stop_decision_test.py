"""
stop_decision_test.py

Isolates "can the model RECOGNISE completion?" from "will it spontaneously
EMIT a stop token?".

Why
---
Across 4 configs x 12 episodes x 5 turns (240 generations), LLaMA-3.1-8B emitted
DONE exactly 0 times -- not once, not even as a substring, and not even with a
few-shot example that ended in DONE. Every episode ran to max_steps. Two very
different explanations predict that same observation:

  (i)  the model cannot tell when the request is already served -- a genuine
       capability gap, and the thing this project would then be about;
  (ii) the open-ended "Next:" prompt after a 35-tool catalogue primes a tool
       name so strongly that DONE never wins the argmax -- a harness artifact,
       in which case every number collected so far is meaningless.

This script separates them by asking the SAME underlying question in three
formats over identical states:

  A. FREE     : the agent prompt, open generation          (what we have been doing)
  B. BINARY   : "Have all required tools been called? YES/NO"
  C. FORCED   : compare the logprob of "DONE" against the logprob of the best
                legal tool name -- does DONE even rank, regardless of decoding?

States are constructed by oracle prefix, so the ground truth is known:
  - COMPLETE   : exactly the gold set has been called -> correct answer is stop
  - INCOMPLETE : one gold tool is missing              -> correct answer is continue

If B is accurate while A never stops, the failure is format and the harness
needs a decomposed stop decision (as CaRT uses). If B is also at chance, the
model cannot recognise completion and that is the finding.

    sbatch scripts/run_stop_decision_test.sh
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from env.nl_conditional_env import NLConditionalEnv, SCENARIOS


def build_state(scenario, conditional, bit, complete):
    """Oracle-prefix an episode to a known state. Returns (env, missing_tool)."""
    env = NLConditionalEnv(alpha=1.0 if conditional else 0.0, seed=0)
    env.reset(options={"scenario": scenario, "is_conditional": conditional, "latent_bit": bit})
    gold = sorted(env.gold)
    if conditional:
        # Probe first so the branch is revealed, mirroring correct play.
        order = [env.s.probe] + [g for g in gold if g != env.s.probe]
    else:
        order = gold
    to_call = order if complete else order[:-1]
    for t in to_call:
        env.step(t)
    missing = None if complete else order[-1]
    return env, missing


@torch.no_grad()
def free_generation(model, tok, prompt, max_new_tokens=12):
    enc = tok(tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      tokenize=False, add_generation_prompt=True),
              return_tensors="pt").to(model.device)
    enc.pop("token_type_ids", None)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def first_token_logprobs(model, tok, prompt, candidates):
    """Logprob of each candidate's FIRST token, at the first generated position.

    Approximate (multi-token names are judged on their first token only), but
    enough to answer the question that matters: does DONE rank anywhere near the
    tool names, or is it effectively excluded before decoding begins?
    """
    enc = tok(tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      tokenize=False, add_generation_prompt=True),
              return_tensors="pt").to(model.device)
    enc.pop("token_type_ids", None)
    logits = model(**enc).logits[0, -1]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    out = {}
    for c in candidates:
        ids = tok.encode(c, add_special_tokens=False)
        if ids:
            out[c] = float(logprobs[ids[0]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--out", default="results/stop_decision_test.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto", local_files_only=True)
    model.eval()
    print(f"model on {model.device}\n", flush=True)

    records = []
    for conditional in (False, True):
        for complete in (True, False):
            tag = f"{'conditional' if conditional else 'static':<11} {'COMPLETE' if complete else 'INCOMPLETE'}"
            a_stop = b_correct = c_done_ranks = 0
            n = 0
            for s_idx in range(len(SCENARIOS)):
                env, missing = build_state(s_idx, conditional, bit=s_idx % 2, complete=complete)
                n += 1

                # --- A. free generation, the agent prompt we have been using
                raw_a = free_generation(model, tok, env.observation())
                said_done = raw_a.upper().startswith("DONE")
                a_stop += said_done

                # --- B. direct binary question over the same state
                called = ", ".join(env.selected) or "(none)"
                q = (f"User request: {env.query}\n\n"
                     f"Tools already called: {called}\n\n"
                     "Has everything the request needs now been done? "
                     "Answer with exactly one word: YES or NO.")
                raw_b = free_generation(model, tok, q, max_new_tokens=4).upper()
                said_yes = raw_b.startswith("YES")
                b_correct += (said_yes == complete)

                # --- C. does DONE even rank against the best legal tool?
                cands = ["DONE"] + [t for t in env.legal() if t != "DONE"]
                lp = first_token_logprobs(model, tok, env.observation(), cands)
                best_tool = max((v, k) for k, v in lp.items() if k != "DONE")
                done_lp = lp.get("DONE", float("-inf"))
                rank = sum(1 for k, v in lp.items() if v > done_lp) + 1
                c_done_ranks += rank

                records.append({
                    "scenario": s_idx, "conditional": conditional, "complete": complete,
                    "gold": sorted(env.gold), "called": list(env.selected), "missing": missing,
                    "A_raw": raw_a, "A_said_done": bool(said_done),
                    "B_raw": raw_b, "B_said_yes": bool(said_yes), "B_correct": bool(said_yes == complete),
                    "C_done_logprob": done_lp, "C_best_tool": best_tool[1],
                    "C_best_tool_logprob": best_tool[0], "C_done_rank": rank,
                })

            print(f"{tag} | A free-gen said DONE: {a_stop:>2}/{n}   "
                  f"B binary correct: {b_correct:>2}/{n}   "
                  f"C mean rank of DONE: {c_done_ranks / n:.1f} of {len(cands)}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)

    comp = [r for r in records if r["complete"]]
    a = sum(r["A_said_done"] for r in comp) / max(1, len(comp))
    b = sum(r["B_correct"] for r in comp) / max(1, len(comp))
    print("\n" + "=" * 68)
    print(f"On COMPLETE states: free-gen stops {a:.0%} of the time, "
          f"binary question correct {b:.0%} of the time.")
    if b > 0.7 and a < 0.2:
        print("VERDICT: the model CAN recognise completion but will not emit it in")
        print("         the open-ended format. This is a HARNESS artifact -- the stop")
        print("         decision must be decomposed (ask explicitly, or constrain")
        print("         decoding). Prior LLM numbers are invalid; rerun after fixing.")
    elif b <= 0.7:
        print("VERDICT: the model cannot reliably recognise completion even when")
        print("         asked directly. That is a genuine capability gap and is the")
        print("         finding -- over-gathering has a mechanism.")
    else:
        print("VERDICT: mixed; inspect results/stop_decision_test.json by hand.")
    print("=" * 68, flush=True)


if __name__ == "__main__":
    main()
