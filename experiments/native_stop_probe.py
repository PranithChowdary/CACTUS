"""
native_stop_probe.py

THE DECISIVE CHECK on this project's central claim.

The claim is that instruction-tuned models fail to terminate. The evidence is
that a literal "DONE" string carries ~0.2% of the probability mass at the step
where the task is complete. The obvious objection -- and it is a good one -- is
that these models were never post-trained to end a tool-use episode by naming a
pseudo-tool called DONE. They end turns by emitting ordinary text and a native
end-of-turn token. A low probability on "DONE" is therefore consistent with two
incompatible readings:

  (a) the model cannot recognise that the task is complete, or
  (b) it recognises completion and expresses it through a channel our action
      set does not contain.

This script separates them. At every step it scores, alongside the tools and
"DONE", each model's own NATIVE stop tokens -- eos_token, and for chat models
the end-of-turn marker (<|eot_id|>, <|im_end|>, <end_of_turn>, ...) read off the
tokenizer rather than hardcoded. Nothing else changes: same prompts, same
scenarios, same forward passes, one extra candidate per step. The cost is
negligible next to a full re-run.

How to read the output:
  P(native stop | gold complete) HIGH while P(DONE | gold complete) ~ 0
      -> reading (b). The paper's central claim is an artifact of the action
         set and must be withdrawn or reframed as an interface result.
  BOTH ~ 0
      -> reading (a). The claim is considerably strengthened: the model does
         not signal completion through its own native mechanism either.

Either outcome is publishable. Not running it is the only unpublishable state,
which is why this is the first job to submit.

    sbatch scripts/run_native_stop_probe.sh
"""
import argparse, json, os, sys, time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from env.nl_conditional_env import NLConditionalEnv, SCENARIOS
from models.llm_policy import ConstrainedLLMPolicy

# Chat end-of-turn markers across the families we evaluate. We only USE one if
# the tokenizer actually encodes it as a single known token, so a wrong guess
# is skipped rather than silently scored as ordinary text.
EOT_CANDIDATES = ["<|eot_id|>", "<|im_end|>", "<end_of_turn>", "<|end|>",
                  "<|endoftext|>", "</s>"]


def native_stop_tokens(tok):
    """Every native stop marker this tokenizer genuinely knows, with its id."""
    found = {}
    vocab = tok.get_vocab()
    for t in EOT_CANDIDATES:
        if t in vocab:
            found[t] = vocab[t]
    if tok.eos_token and tok.eos_token not in found:
        found[tok.eos_token] = tok.eos_token_id
    return found


@torch.no_grad()
def score_with_native(policy, prompt, candidates, native):
    """Scores the normal candidates, then the native stop tokens by id.

    The native markers are scored directly from the logits at the first
    generated position rather than through the tokenizer round-trip, because a
    special token re-encoded from its string form can split into pieces and
    would then be scored as several ordinary tokens.
    """
    scores = policy.score_actions(prompt, candidates)
    prefix = policy._prefix_ids(prompt)
    ids = torch.tensor([prefix], device=policy.model.device)
    logits = policy.model(input_ids=ids).logits[0, -1].float()
    logZ = torch.logsumexp(logits, dim=-1)
    nat = {name: float(logits[tid] - logZ) for name, tid in native.items()}
    return scores, nat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--episodes", type=int, default=61)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0])
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--out", default="results/native_stop.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto", local_files_only=True).eval()
    policy = ConstrainedLLMPolicy(model, tok, batch_size=args.batch_size)
    native = native_stop_tokens(tok)
    print(f"model on {model.device}")
    print(f"native stop tokens found: {native}\n", flush=True)
    if not native:
        print("!! no native stop token identified -- the comparison is impossible "
              "for this tokenizer; aborting rather than reporting a misleading zero")
        return 1

    out = []
    for alpha in args.alphas:
        t0 = time.time()
        rows = []
        for i in range(args.episodes):
            env = NLConditionalEnv(alpha=alpha, seed=i)
            obs, info = env.reset(options={"scenario": i % len(SCENARIOS)})
            done = False
            while not done:
                cands = env.legal()
                complete = set(env.selected) == set(env.gold)
                scores, nat = score_with_native(policy, obs, cands, native)
                dist = scores.distribution("sum")
                # Mass on each stop channel, renormalised over the SAME support
                # (legal actions + native markers) so the two are comparable.
                tot = torch.logsumexp(torch.tensor(
                    [scores.sum_logprob[c] for c in cands] + list(nat.values())), dim=0)
                p_done = float(torch.exp(torch.tensor(scores.sum_logprob["DONE"]) - tot))
                p_nat = float(sum(np.exp(v - float(tot)) for v in nat.values()))
                rows.append({"episode": i, "step": env.steps, "gold_complete": bool(complete),
                             "p_done": p_done, "p_native_stop": p_nat,
                             "best_native": max(nat, key=nat.get),
                             "argmax_is_done": scores.best("sum") == "DONE"})
                obs, r, done, info = env.step(scores.best("sum"))
            print(f"  episode {i+1}/{args.episodes} ({time.time()-t0:.0f}s)", flush=True)

        at = [r for r in rows if r["gold_complete"]]
        before = [r for r in rows if not r["gold_complete"]]
        m = lambda xs, k: float(np.mean([x[k] for x in xs])) if xs else float("nan")
        rec = {"model": args.model, "alpha": alpha, "n_steps": len(rows),
               "n_complete_states": len(at),
               "p_done_at_complete": m(at, "p_done"),
               "p_native_at_complete": m(at, "p_native_stop"),
               # The contrast the paper cannot currently make: is either stop
               # channel ELEVATED at completion, or flat across the episode?
               "p_done_before_complete": m(before, "p_done"),
               "p_native_before_complete": m(before, "p_native_stop"),
               "steps": rows, "wall_seconds": time.time() - t0}
        out.append(rec)
        print(f"\n[alpha={alpha}] P(DONE|complete)={rec['p_done_at_complete']:.4f} "
              f"vs before={rec['p_done_before_complete']:.4f}")
        print(f"[alpha={alpha}] P(native|complete)={rec['p_native_at_complete']:.4f} "
              f"vs before={rec['p_native_before_complete']:.4f}\n", flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        base, ext = os.path.splitext(args.out)
        with open(f"{base}_{args.model_tag}{ext}", "w") as f:
            json.dump(out, f, indent=2)

    r = out[0]
    print("=" * 70)
    print("INTERPRETATION")
    if r["p_native_at_complete"] > 10 * max(r["p_done_at_complete"], 1e-9):
        print("  Native stop carries far more mass than DONE. The termination")
        print("  claim is an ACTION-SET ARTIFACT and must be reframed.")
    elif r["p_native_at_complete"] < 0.05:
        print("  Neither channel carries mass at completion. The termination")
        print("  claim is STRENGTHENED: the native mechanism is absent too.")
    else:
        print("  Intermediate. Report both channels and do not over-read either.")
    print(f"  elevation at completion: DONE x{r['p_done_at_complete']/max(r['p_done_before_complete'],1e-9):.1f}, "
          f"native x{r['p_native_at_complete']/max(r['p_native_before_complete'],1e-9):.1f}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
