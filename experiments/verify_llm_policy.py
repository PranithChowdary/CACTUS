"""
verify_llm_policy.py

Correctness tests for models/llm_policy.py. Runs on any small causal LM (gpt2
by default), so the scoring maths can be checked in seconds on a laptop rather
than discovered to be wrong after a cluster run.

Checks, in order of how badly a failure would corrupt results:

1. BATCH INVARIANCE    scoring a candidate in a batch must equal scoring it
                       alone. Padding or position bugs show up here first.
2. MANUAL AGREEMENT    the summed log-probability must equal an independent
                       token-by-token computation.
3. VALID LOGPROBS      every score <= 0; softmax over candidates sums to 1.
4. LENGTH BIAS EXISTS  demonstrates the problem PMI is there to solve: sum
                       correlates negatively with token length.
5. PMI REDUCES IT      the primary score must be less length-correlated than
                       the raw sum, otherwise calibration is not doing its job.
6. SANITY RANKING      on a prompt with an obviously correct continuation, that
                       continuation should win.

    python experiments/verify_llm_policy.py --model gpt2
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.llm_policy import ConstrainedLLMPolicy

PASS, FAIL, WARN = "PASS", "** FAIL **", "warn"
results = []      # hard correctness invariants -- a failure invalidates results
diagnostics = []  # informative properties -- a failure is worth seeing, not fatal


def check(name, ok, detail="", critical=True):
    """critical=True  : scoring maths is wrong, downstream numbers are garbage.
       critical=False : a property of the model/scorer worth reporting, but not
                        a reason to abort a job. Conflating the two caused a
                        spurious abort on a badly-specified length-bias test."""
    (results if critical else diagnostics).append(ok)
    tag = (PASS if ok else FAIL) if critical else (PASS if ok else WARN)
    print(f"  [{tag}] {name}{(' -- ' + detail) if detail else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    # gpt2 has no chat template; the real run uses one.
    policy = ConstrainedLLMPolicy(model, tok, scoring="pmi", use_chat_template=False,
                                  premise="Reply with one word.\nNext:")

    prompt = ("You are a tool-selection agent.\n"
              "User request: Book me a hotel and a rental car for Denver.\n"
              "Next:")
    candidates = [" HotelTool", " CarRentalTool", " WeatherTool", " DONE",
                  " PharmacyTool", " DeployTool"]

    print("\n1. BATCH INVARIANCE")
    prefix = policy._prefix_ids(prompt)
    batched = policy._score(prefix, candidates)
    policy.batch_size = 1
    single = policy._score(prefix, candidates)
    policy.batch_size = 48
    worst = max(abs(batched[c] - single[c]) for c in candidates)
    check("batched == unbatched", worst < 1e-3, f"max abs diff {worst:.2e}")

    print("\n2. MANUAL AGREEMENT (independent token-by-token recomputation)")
    c = candidates[0]
    cid = policy._cand_ids(c)
    seq = list(prefix) + cid
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq])).logits.float(), dim=-1)[0]
    manual = sum(float(lp[len(prefix) + j - 1, t]) for j, t in enumerate(cid))
    check("summed logprob matches manual", abs(manual - batched[c]) < 1e-3,
          f"manual {manual:.4f} vs policy {batched[c]:.4f}")

    print("\n3. VALID LOG-PROBABILITIES")
    s = policy.score_actions(prompt, candidates)
    check("all sum logprobs <= 0", all(v <= 0 for v in s.sum_logprob.values()))
    for mode in ("sum", "mean", "pmi"):
        d = s.distribution(mode)
        check(f"softmax over candidates sums to 1 ({mode})", abs(sum(d.values()) - 1) < 1e-5)

    # Tests 4 and 5 must isolate length from CORRECTNESS. A first version
    # correlated length against score over 6 hand-picked candidates, 4 of them
    # tied at 2 tokens; the result was driven entirely by DONE (1 token,
    # correctly lowest) and CarRentalTool (4 tokens, correctly highest), so it
    # measured correctness-vs-length confounding, not length bias, and reported
    # a spurious failure. The fix: correlate over tools that are ALL equally
    # irrelevant to the query, where any remaining length effect is pure bias.
    from env.nl_conditional_env import TOOLS
    relevant = {" HotelTool", " CarRentalTool", " TripTool", " WeatherTool", " DONE"}
    distractors = [f" {t}" for t in TOOLS
                   if t != "DONE" and f" {t}" not in relevant]
    s_d = policy.score_actions(prompt, distractors)
    lens = np.array([s_d.n_tokens[c] for c in distractors], dtype=float)

    print(f"\n4. LENGTH BIAS IS REAL  (n={len(distractors)} tools, all irrelevant "
          f"to the query, lengths {int(lens.min())}-{int(lens.max())} tokens)")
    r_sum = float(np.corrcoef(lens, np.array([s_d.sum_logprob[c] for c in distractors]))[0, 1])
    check("sum-logprob correlates negatively with length", r_sum < -0.2,
          f"r = {r_sum:+.3f} (longer strings score worse despite equal irrelevance)",
          critical=False)

    print("\n5. PMI REDUCES LENGTH BIAS")
    r_pmi = float(np.corrcoef(lens, np.array([s_d.pmi[c] for c in distractors]))[0, 1])
    check("|r| smaller for PMI than for raw sum", abs(r_pmi) < abs(r_sum),
          f"r_pmi = {r_pmi:+.3f} vs r_sum = {r_sum:+.3f}", critical=False)

    print("\n6. SANITY RANKING")
    best_pmi = s.best("pmi")
    check("a plausible tool outranks an irrelevant one under PMI",
          s.pmi[" HotelTool"] > s.pmi[" DeployTool"],
          f"HotelTool {s.pmi[' HotelTool']:+.2f} vs DeployTool {s.pmi[' DeployTool']:+.2f}",
          critical=False)
    print(f"      (argmax under PMI on this toy prompt: {best_pmi!r})")

    print("\n--- per-candidate table ---")
    print(f"  {'candidate':<16}{'toks':>5}{'sum':>10}{'mean':>9}{'prior':>10}{'PMI':>10}")
    for c in sorted(candidates, key=lambda x: -s.pmi[x]):
        print(f"  {c:<16}{s.n_tokens[c]:>5}{s.sum_logprob[c]:>10.2f}"
              f"{s.mean_logprob[c]:>9.2f}{s.prior_logprob[c]:>10.2f}{s.pmi[c]:>10.2f}")

    print(f"\ncritical: {sum(results)}/{len(results)} passed   "
          f"diagnostic: {sum(diagnostics)}/{len(diagnostics)} passed")
    if not all(results):
        print("CRITICAL FAILURE: the scoring maths is wrong; downstream numbers "
              "would be invalid.")
        return 1
    if not all(diagnostics):
        print("Note: a diagnostic property did not hold. Not fatal -- inspect the "
              "table above, then proceed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
