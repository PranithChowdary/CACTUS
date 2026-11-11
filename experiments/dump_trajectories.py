"""
dump_trajectories.py

Prints what the model ACTUALLY said, turn by turn, for a handful of episodes.

Motivation: the first real run of llm_probe_test.py returned exact_match=0.14
at alpha=0, a setting where a logistic probe scores 1.00, with
unneeded_branch_rate=1.00. Aggregates cannot distinguish the candidate causes:

  (a) the model never emits DONE and runs to max_steps, so the selected set is
      always too large and exact match is impossible by construction;
  (b) parse_tool's substring fallback mis-resolves the output to the wrong tool;
  (c) the model genuinely over-predicts, which would be a real finding.

Each produces the same summary numbers and a completely different conclusion,
so the only way forward is to read the transcripts. Prints, per turn: the raw
generation, what it parsed to, whether the parse was exact or a fallback, and
the running selection against the gold set.

    sbatch scripts/run_dump_trajectories.sh
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from env.nl_conditional_env import NLConditionalEnv, SCENARIOS

FEW_SHOT = """Example of the expected format:
User request: Book me somewhere to stay in Lisbon next week and tell my manager.
Next: HotelTool
Next: EmailTool
Next: DONE

Now the real request.

"""


def parse_tool_verbose(text, legal, already_selected=()):
    """Same logic as llm_probe_test.parse_tool, but reports HOW it matched.

    REPEAT is distinguished from UNPARSED: naming an already-called tool is the
    model failing to stop, not the parser failing to understand. Collapsing the
    two hid the real failure in the first run (12/12 "UNPARSED" episodes were
    almost entirely repeats).
    """
    text = text.strip().split("\n")[0].strip().strip("`'\".:")
    for t in legal:
        if text.lower() == t.lower():
            return t, "exact"
    for t in already_selected:
        if text.lower() == t.lower():
            return text, "REPEAT"
    for t in legal:
        if t.lower() in text.lower():
            return t, "substring-fallback"
    return text, "UNPARSED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="0.0 is the diagnostic case: a static task a classifier solves perfectly.")
    ap.add_argument("--episodes", type=int, default=7, help="one per scenario by default")
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--few_shot", action="store_true",
                    help="Prepend a worked example ending in DONE. Few-shot lifted "
                         "probe-first from 0.57 to 0.86 in the earlier probe test.")
    ap.add_argument("--out", default="results/trajectories.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto", local_files_only=True)
    model.eval()
    print(f"model on {model.device}\n", flush=True)

    records = []
    for ep in range(args.episodes):
        env = NLConditionalEnv(alpha=args.alpha, seed=ep)
        obs, info = env.reset(options={"scenario": ep % len(SCENARIOS)})
        s = SCENARIOS[info["scenario"]]
        print("=" * 72, flush=True)
        print(f"EPISODE {ep}  scenario={info['scenario']}  conditional={info['is_conditional']}  "
              f"bit={info['latent_bit']}", flush=True)
        print(f"  query : {env.query}", flush=True)
        print(f"  gold  : {info['gold']}   (probe={s.probe}, probe_is_gold={info['probe_is_gold']}, "
              f"branches={s.branch}, static_gold={s.static_gold})", flush=True)

        turns = []
        done = False
        turn = 0
        while not done:
            prompt = (FEW_SHOT + obs) if args.few_shot else obs
            enc_text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                               tokenize=False, add_generation_prompt=True)
            enc = tok(enc_text, return_tensors="pt").to(model.device)
            enc.pop("token_type_ids", None)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            raw = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
            action, how = parse_tool_verbose(raw, env.legal(), env.selected)
            obs, r, done, info = env.step(action)
            turn += 1
            flag = "" if how == "exact" else f"   <-- {how}"
            print(f"  turn {turn}: raw={raw.strip()!r:40s} -> {action!r:16s} "
                  f"reward={r:+.2f}  selected={info['selected']}{flag}", flush=True)
            turns.append({"turn": turn, "raw": raw, "parsed": action,
                          "parse_mode": how, "reward": r,
                          "selected": list(info["selected"])})

        sel, gold = set(info["selected"]), set(info["gold"])
        verdict = "EXACT MATCH" if sel == gold else (
            "SUPERSET (over-predicted)" if gold < sel else
            "SUBSET (under-predicted)" if sel < gold else "WRONG SET")
        emitted_done = any(t["parsed"] == "DONE" for t in turns)
        over_probed = (not info["is_conditional"]) and (s.probe in sel)
        note = "  <-- OVER-PROBED (probe not needed here)" if over_probed else ""
        print(f"  RESULT: {verdict}   emitted DONE: {emitted_done}   "
              f"turns used: {turn}/{env.max_steps}{note}", flush=True)
        records.append({
            "episode": ep, "scenario": info["scenario"],
            "is_conditional": info["is_conditional"], "latent_bit": info["latent_bit"],
            "gold": sorted(gold), "selected": sorted(sel),
            "verdict": verdict, "emitted_done": emitted_done,
            "over_probed": bool(over_probed),
            "turns_used": turn, "turns": turns,
        })

    print("\n" + "=" * 72, flush=True)
    n = len(records)
    n_done = sum(r["emitted_done"] for r in records)
    n_maxed = sum(r["turns_used"] >= 5 for r in records)
    n_unparsed = sum(any(t["parse_mode"] == "UNPARSED" for t in r["turns"]) for r in records)
    n_repeat = sum(any(t["parse_mode"] == "REPEAT" for t in r["turns"]) for r in records)
    print(f"episodes repeating a called tool: {n_repeat}/{n}")
    n_fallback = sum(any(t["parse_mode"] == "substring-fallback" for t in r["turns"]) for r in records)
    n_over = sum(r.get("over_probed") for r in records)
    n_static = sum(not r["is_conditional"] for r in records)
    print(f"episodes emitting DONE          : {n_done}/{n}")
    if n_static:
        print(f"OVER-PROBED (static episodes)   : {n_over}/{n_static}")
    print(f"episodes hitting max_steps      : {n_maxed}/{n}")
    print(f"episodes with an UNPARSED turn  : {n_unparsed}/{n}")
    print(f"episodes using substring fallback: {n_fallback}/{n}")
    print()
    if n_done < n / 2:
        print("DIAGNOSIS (a): the model rarely emits DONE. Exact match is near-impossible")
        print("  by construction -- fix the stopping affordance (prompt and/or max_steps)")
        print("  before drawing any conclusion about information gathering.")
    elif n_unparsed or n_fallback > n / 2:
        print("DIAGNOSIS (b): parsing is unreliable. Fix parse_tool before trusting metrics.")
    else:
        print("DIAGNOSIS (c): the model stops cleanly and parses cleanly, yet still")
        print("  over-selects. That is a genuine over-prediction finding.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
