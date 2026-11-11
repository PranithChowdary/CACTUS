"""
check_scenario_bank.py

The gate that decides whether the expanded NL bank can carry the LLM-vs-RL
comparison at all. It must pass BEFORE any gym adapter or training job is
written, because each failure mode below would silently invalidate the
comparison rather than weaken it.

What has to be true, and why
---------------------------
1. CONDITIONALITY IS PREDICTABLE from the query embedding.
   "Should I gather?" must be learnable in-distribution. If it is not, an agent
   that probes on static tasks is facing an impossible discrimination, and
   over-gathering stops being a policy failure -- it becomes the task's fault.
   This is design requirement 2 in RESEARCH_PLAN.md.

2. THE LATENT BRANCH BIT IS **NOT** PREDICTABLE from the query embedding.
   If it leaks, the whole premise collapses: a memoryless classifier could beat
   the (1-alpha) + alpha/2 ceiling without ever probing, and the environment
   would no longer require information gathering. Must sit at chance.

3. SCENARIO IDENTITY IS LINEARLY SEPARABLE in the embedding.
   This is the representation-adequacy check, and it is the one that has bitten
   this project before: at embedding_dim=32 with 40 query types, conditionality
   was only 81% linearly predictable, which handicapped the memoryless baseline
   for REPRESENTATIONAL rather than informational reasons and made the RL
   comparison unfair. A linear policy must be able to represent the query ->
   gold mapping, or PPO's failure would not be about sequential decisions.

4. THE STATIC GOLD SET IS REPRESENTABLE BY THE POLICY CLASS.
   Two different quantities were conflated here on the first pass, and gating
   on the wrong one produced a spurious FAIL:

     (a) what a MEMORYLESS LINEAR BASELINE scores (one-vs-rest logistic,
         top-2). This is a baseline to compare the agent against. It came out
         at 0.705, which is a perfectly good baseline value -- not a defect.
     (b) whether the POLICY CLASS can represent query -> gold at all. PPO's
         MultiInputPolicy is a multi-class softmax behind a 2x64 MLP, not an
         independent per-label linear model, so (a) systematically understates
         (b): one-vs-rest decision functions are not calibrated against each
         other, and tools appearing in only one or two scenarios get buried.

   Only (b) is a gate. If (b) fails, PPO is handicapped by the representation
   and the comparison is unfair -- the exact failure that occurred at
   embedding_dim=32 in the synthetic env. (a) is reported, never gated.

5. CONDITIONALITY MUST NOT REDUCE TO ONE KEYWORD.
   If every conditional query is separable on "depending on", then "the query
   signals whether the task is conditional" is a claim about a token, not about
   meaning, and a reviewer will say the LLM is pattern-matching.

   The first implementation of this check DELETED the marker spans, and was
   invalid: it cut conditional queries from 11.2 to 6.3 mean words against
   11.6 for static ones, and left truncation artifacts ("My connection in
   Frankfurt is tight..", "Close out the month on the company card,."). The
   classifier then separated the classes on LENGTH and broken punctuation, and
   accuracy went UP (0.804 -> 0.852), which read as "the signal is semantic"
   when it measured nothing of the kind.

   The marker span is now REPLACED by a length-comparable neutral filler that
   carries no information about information-need, keeping both classes
   well-formed and length-matched. Length parity is asserted and printed, so
   this cannot silently break again.

   Expect accuracy to stay high, and for a legitimate reason: static queries
   NAME the actions they need ("Book me a hotel and a rental car") while
   conditional ones do not. That absence is what "underspecified" means and is
   exactly the semantic cue the design intends. The markers are redundant with
   it.

fit vs. generalize -- both are reported, and they answer different questions
---------------------------------------------------------------------------
Query text is FIXED per (scenario, conditional) pair, so embeddings are
deterministic and an agent trained on all scenarios only has to MEMORIZE 2N
points. For that design, linear *fit* is the relevant quantity (check 3).
Cross-validated accuracy answers a different question -- whether the mapping
transfers to a scenario never seen -- which matters only under the held-out
split in env/nl_scenarios.split(). Reporting one as the other would overstate
or understate the bank depending on which design is used, so both are printed.

    python experiments/check_scenario_bank.py
"""
import argparse
import json
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from env.nl_scenarios import SCENARIOS, TOOLS, split

EMB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "models", "embedding", "all-MiniLM-L6-v2")

# Gates. Numbers chosen before running, so a marginal result cannot be talked up.
GATE_COND_FIT = 0.95     # conditionality must be representable
GATE_SCEN_FIT = 0.95     # scenario identity must be representable
GATE_GOLD_FIT = 0.90     # static gold must be representable BY THE POLICY CLASS
GATE_BIT_MAX = 0.60      # branch bit must stay near chance (0.5)

# The lexical markers that make a query obviously conditional. Stripping these
# tests whether the signal is semantic or a single-token shortcut.
MARKERS = [
    r"\bdepending on [^.,]*", r"\bas appropriate\b", r"\bwhatever [^.,]*",
    r"\bhowever [^.,]*", r"\baccording to [^.,]*", r"\bbased on [^.,]*",
    r"\bas the [^.,]* (?:requires|dictates|allows|indicates|warrants|shows|permits|justifies|calls for)\b",
    r"\bact on [^.,]*", r"\bper the [^.,]*", r"\bin line with [^.,]*",
    r"\bthe (?:right|appropriate) (?:step|action) for [^.,]*",
    r"\bwhat(?:'s| is) warranted\b", r"\bas (?:its|their) [^.,]* requires\b",
    r"\bthat (?:fit|fits) the situation\b", r"\bif (?:the situation )?calls? for it\b",
    r"\bwhere it stands\b", r"\bas the terms require\b", r"\bthe conditions call for\b",
    r"\bin the (?:way|manner|form|format) [^.,]*", r"\bto the degree [^.,]*",
    r"\bor (?:don't|hold it)[^.,]*", r"\band act accordingly\b",
    r"\bresponds? to what you find\b", r"\brespond as [^.,]*",
    r"\bdo what [^.,]*", r"\btake the [^.,]*", r"\bhandle it [^.,]*",
    r"\bproceed (?:as|on) [^.,]*", r"\bwhichever [^.,]*",
]


# Neutral fillers: comparable length to the marker spans they replace, and
# uninformative about whether a check is needed. Cycled deterministically.
FILLERS = [
    "when you get a chance",
    "as we discussed earlier",
    "before the end of the day",
    "and let me know once it is done",
    "sometime this week please",
]


def strip_markers(text: str, neutralise: bool = True, i: int = 0) -> str:
    """Remove the conditional marker phrase.

    neutralise=True replaces it with a length-comparable neutral filler, which
    is the only valid form of this ablation: plain deletion halves the sentence
    and leaves broken punctuation, so the classes become separable on length
    and artifacts rather than on meaning.
    """
    out, n = text, 0
    for m in MARKERS:
        out, k = re.subn(m, "<<M>>" if neutralise else "", out, flags=re.IGNORECASE)
        n += k
    if neutralise:
        # Only the FIRST match site becomes filler; any others just vanish.
        parts = out.split("<<M>>")
        out = parts[0] + FILLERS[i % len(FILLERS)] + "".join(parts[1:]) if len(parts) > 1 else parts[0]
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    out = re.sub(r"([.,;:])\1+", r"\1", out)        # "tight.." -> "tight."
    out = re.sub(r",\s*\.", ".", out)               # "card,." -> "card."
    # Capitalise a filler that landed at the start of a sentence: "SKU 2210.
    # when you get a chance." leaves a lowercase-after-period cue, which is a
    # surface artifact of the same kind the deletion version was rejected for.
    out = re.sub(r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out)
    out = out.strip()
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    if out and out[-1] not in ".!?":
        out += "."
    return out


@torch.no_grad()
def embed(texts, batch_size=32):
    """MiniLM mean-pooled, L2-normalised -- i.e. exactly what
    sentence-transformers does for this checkpoint (modules.json:
    Transformer -> Pooling(mean) -> Normalize), without the dependency."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMB_DIR, local_files_only=True)
    mod = AutoModel.from_pretrained(EMB_DIR, local_files_only=True).eval()
    out = []
    for i in range(0, len(texts), batch_size):
        b = tok(texts[i:i + batch_size], padding=True, truncation=True,
                max_length=256, return_tensors="pt")
        h = mod(**b).last_hidden_state                      # [B, T, H]
        mask = b["attention_mask"].unsqueeze(-1).float()     # [B, T, 1]
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out.append(torch.nn.functional.normalize(pooled, p=2, dim=1))
    return torch.cat(out).numpy()


def fit_acc(X, y, C=10.0):
    """Training accuracy = can a LINEAR policy represent this mapping at all."""
    clf = LogisticRegression(C=C, max_iter=5000).fit(X, y)
    return float(clf.score(X, y))


def cv_acc(X, y, C=10.0, folds=5):
    """Cross-validated accuracy = does the mapping transfer to unseen queries."""
    y = np.asarray(y)
    counts = np.bincount(y) if y.dtype.kind in "iu" else None
    if counts is not None and counts.min() < 2:
        return float("nan")
    k = min(folds, int(counts.min())) if counts is not None else folds
    if k < 2:
        return float("nan")
    accs = []
    for tr, te in StratifiedKFold(n_splits=k, shuffle=True, random_state=0).split(X, y):
        clf = LogisticRegression(C=C, max_iter=5000).fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


def topk_exact_match(X, Y, k=2, cv=False):
    """Memoryless multilabel top-k exact-set-match -- the empirical ceiling for
    a query-only model. Given the target cardinality, which if anything favours
    it over the agent, since the agent must also learn when to stop."""
    def score(Xtr, Ytr, Xte, Yte):
        clf = OneVsRestClassifier(LogisticRegression(C=10.0, max_iter=5000)).fit(Xtr, Ytr)
        S = clf.decision_function(Xte)
        hit = 0
        for r in range(S.shape[0]):
            pred = set(np.argsort(-S[r])[:k].tolist())
            gold = set(np.where(Yte[r] == 1)[0].tolist())
            hit += int(pred == gold)
        return hit / S.shape[0]
    if not cv:
        return score(X, Y, X, Y)
    accs = []
    idx = np.arange(len(X))
    rng = np.random.default_rng(0); rng.shuffle(idx)
    for f in np.array_split(idx, 5):
        tr = np.setdiff1d(idx, f)
        accs.append(score(X[tr], Y[tr], X[f], Y[f]))
    return float(np.mean(accs))


def gold_mlp_fit(X, Y, k=2, seed=0):
    """Can the POLICY CLASS represent query -> gold set?

    Matches how MaskablePPO actually parameterises its head: one shared 2x64
    MLP trunk and a softmax over all tools, scored jointly, rather than 36
    independent linear discriminants. Fit (not CV) is the right measure -- the
    question is representational capacity, and with a fixed query per scenario
    an agent trained on all scenarios only has to memorise them.
    """
    y = np.array([tuple(np.where(r == 1)[0]) for r in Y])
    classes = {t: i for i, t in enumerate(sorted(set(map(tuple, y))))}
    yc = np.array([classes[tuple(t)] for t in y])
    clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=4000,
                        random_state=seed).fit(X, yc)
    return float(clf.score(X, yc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/scenario_bank_check.json")
    args = ap.parse_args()

    n = len(SCENARIOS)
    tool_idx = {t: i for i, t in enumerate(k for k in TOOLS if k != "DONE")}

    cond = [s.cond_query for s in SCENARIOS]
    stat = [s.static_query for s in SCENARIOS]
    cond_stripped = [strip_markers(s.cond_query, neutralise=True, i=i)
                     for i, s in enumerate(SCENARIOS)]
    # Self-validating: if the ablation ever stops being length-matched it is
    # measuring length again, so refuse to report it as a semantic result.
    wl = lambda xs: float(np.mean([len(x.split()) for x in xs]))
    len_cond, len_stat, len_abl = wl(cond), wl(stat), wl(cond_stripped)

    print(f"bank: {n} scenarios, {len(tool_idx)} tools + DONE")
    print("embedding with MiniLM-L6-v2 (mean pool + L2 norm)...")
    E = embed(cond + stat + cond_stripped)
    Ec, Es, Ecs = E[:n], E[n:2 * n], E[2 * n:]

    R = {"n_scenarios": n}

    # ---- 1. conditionality -------------------------------------------------
    X1 = np.vstack([Ec, Es]); y1 = np.array([1] * n + [0] * n)
    R["cond_fit"] = fit_acc(X1, y1); R["cond_cv"] = cv_acc(X1, y1)

    # ---- 2. latent bit must NOT leak --------------------------------------
    # The conditional query text is identical for bit=0 and bit=1, so the bit is
    # unavailable BY CONSTRUCTION. Measured anyway: a future edit that made the
    # query hint at the branch would break the ceiling argument silently.
    Xb = np.vstack([Ec, Ec]); yb = np.array([0] * n + [1] * n)
    R["bit_fit"] = fit_acc(Xb, yb); R["bit_cv"] = cv_acc(Xb, yb)

    # ---- 3. scenario identity (representation adequacy) -------------------
    X3 = np.vstack([Ec, Es]); y3 = np.array(list(range(n)) * 2)
    R["scen_fit"] = fit_acc(X3, y3); R["scen_cv"] = cv_acc(X3, y3)

    # ---- 4. static gold recoverable by a memoryless top-2 model -----------
    Y4 = np.zeros((n, len(tool_idx)), dtype=int)
    for i, s in enumerate(SCENARIOS):
        for t in s.static_gold:
            Y4[i, tool_idx[t]] = 1
    R["gold_linear_baseline_fit"] = topk_exact_match(Es, Y4, k=2, cv=False)
    R["gold_linear_baseline_cv"] = topk_exact_match(Es, Y4, k=2, cv=True)
    R["gold_policyclass_fit"] = gold_mlp_fit(Es, Y4)
    R["gold_policyclass_cv"] = float("nan")

    # ---- 5. lexical-shortcut ablation -------------------------------------
    X5 = np.vstack([Ecs, Es]); y5 = np.array([1] * n + [0] * n)
    R["cond_fit_stripped"] = fit_acc(X5, y5); R["cond_cv_stripped"] = cv_acc(X5, y5)

    print("\n" + "=" * 78)
    print("SCENARIO BANK CHECK")
    print("=" * 78)
    print(f"  {'check':<42}{'fit':>9}{'cv':>9}{'gate':>9}{'':>6}")
    rows = [
        ("1. conditionality predictable (need HIGH)", "cond_fit", "cond_cv", GATE_COND_FIT, "ge"),
        ("3. scenario identity separable (need HIGH)", "scen_fit", "scen_cv", GATE_SCEN_FIT, "ge"),
        ("4. static gold representable by policy class", "gold_policyclass_fit", "gold_policyclass_cv", GATE_GOLD_FIT, "ge"),
        ("2. latent bit leaks? (need ~0.50, LOW)", "bit_fit", "bit_cv", GATE_BIT_MAX, "le"),
    ]
    ok = True
    for label, fk, ck, gate, direction in rows:
        v = R[fk]
        # The bit check is about GENERALIZATION: a linear model can always
        # memorise two identical points into opposite labels, so `fit` is
        # meaningless there and cv is the number that matters.
        v = R[ck] if direction == "le" else v
        good = (v >= gate) if direction == "ge" else (v <= gate)
        ok &= good
        cvs = "   nan" if np.isnan(R[ck]) else f"{R[ck]:.3f}"
        print(f"  {label:<42}{R[fk]:>9.3f}{cvs:>9}{gate:>9.2f}"
              f"{'  PASS' if good else '  FAIL':>6}")
    print(f"  {'-- memoryless linear baseline (not a gate)':<42}"
          f"{R['gold_linear_baseline_fit']:>9.3f}{R['gold_linear_baseline_cv']:>9.3f}{'--':>9}{'':>6}")
    print(f"  {'5. conditionality, markers STRIPPED':<42}"
          f"{R['cond_fit_stripped']:>9.3f}{R['cond_cv_stripped']:>9.3f}{'--':>9}{'':>6}")

    print("\n" + "-" * 78)
    print(f"  length parity (mean words): conditional {len_cond:.1f} | "
          f"static {len_stat:.1f} | ablated {len_abl:.1f}")
    parity_ok = abs(len_abl - len_stat) <= 2.5
    if not parity_ok:
        print("  !! ABLATION INVALID: the ablated class is no longer length-matched")
        print("  !! to the static class, so check 5 is measuring sentence length,")
        print("  !! not meaning. Do not report it. Widen FILLERS and re-run.")
    R["ablation_length_parity_ok"] = bool(parity_ok)
    R["mean_words"] = {"cond": len_cond, "static": len_stat, "ablated": len_abl}
    drop = R["cond_cv"] - R["cond_cv_stripped"]
    print(f"  lexical-shortcut drop (cv): {R['cond_cv']:.3f} -> "
          f"{R['cond_cv_stripped']:.3f}  (delta {-drop:+.3f})")
    if parity_ok and R["cond_cv_stripped"] >= 0.80:
        print("  The signal survives marker replacement at matched length: it is")
        print("  carried by meaning, not by one keyword. The operative cue is that")
        print("  static queries NAME their actions and conditional ones do not --")
        print("  which is what underspecification is, and is the intended design.")
    elif not parity_ok:
        print("  (interpretation withheld -- see the parity warning above)")
    else:
        print("  DISCLOSE: most of the conditional signal is carried by the marker")
        print("  phrases. State this in the paper -- the LLM may be matching")
        print("  phrasing rather than reasoning about information need.")

    tr, te = split()
    print(f"\n  held-out split available: {len(tr)} train / {len(te)} test")
    print(f"  (cv columns above are the relevant ones if that split is used)")

    print("\n" + "=" * 78)
    print(f"VERDICT: {'PASS -- bank can carry the comparison' if ok else 'FAIL -- fix the bank before building the adapter'}")
    print("=" * 78)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
