"""
evaluate_supervised.py

Evaluates a memoryless supervised probe on the real MetaTool benchmark, under
the SAME protocol as every other agent: one deterministic pass over all 51 test
queries, per-example records written to results/raw/Supervised.json so the
paired significance tests in scripts/recompute_stats.py can pick it up.

Why this is a first-class method and not an afterthought
-------------------------------------------------------
Run ad hoc, a logistic regression over the same frozen all-MiniLM-L6-v2
embeddings, trained on the same 397 examples, matched or beat the full
MaskablePPO pipeline (SEM 0.588 vs 0.573 with a validation-tuned threshold;
0.745 with top-k). If that holds under the shared protocol, the MDP formulation
is unnecessary for this benchmark -- the gold set is a deterministic function
of the query, so the interaction history carries no information the query does
not already provide. Any claim that sequential decision-making helps has to
clear this bar first, and the result belongs in the paper either way.

Two selection rules, because they answer different questions:
  - threshold : cardinality chosen by the model, tuned on VALIDATION only.
                The apples-to-apples comparison against an agent that must
                also decide when to stop.
  - top-k     : cardinality supplied externally (k=2). An upper bound on what
                a classifier can do when the target size is known. Reported
                separately and never used as the headline, since the RL agent
                is not given k.

The probe is deterministic given fixed data (lbfgs on a fixed design matrix),
so it is a single run with zero seed variance, exactly like DenseRetrieval.
Uncertainty is quantified by the paired bootstrap over test examples in
recompute_stats.py, not by reruns.

    python experiments/evaluate_supervised.py
"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from sentence_transformers import SentenceTransformer

from env import config
from models.supervised_baseline import SupervisedToolSelector


def load_split(path, vocab, encoder):
    with open(path) as f:
        rows = json.load(f)
    queries = [r["Query"] for r in rows]
    X = encoder.encode(queries, show_progress_bar=False)
    Y = np.zeros((len(rows), len(vocab)))
    for i, r in enumerate(rows):
        for t in r["Tool"].split(","):
            t = t.strip()
            if t in vocab:
                Y[i, vocab.index(t)] = 1
    return X, Y, queries


def records_from(preds, Y):
    """Per-example records in the same schema evaluate_all.py writes, so
    recompute_stats.py can treat this like any other agent."""
    out = []
    for i, (p, row) in enumerate(zip(preds, Y)):
        gold = set(np.where(row == 1)[0].tolist())
        tp = len(p & gold)
        prec = tp / max(1, len(p))
        rec = tp / max(1, len(gold))
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        out.append({
            "index": i,
            "f1": f1,
            "success": 1.0 if p == gold else 0.0,
            "steps": float(len(p) + 1),  # +1 for the implicit stop, comparable to TL
            "reward": 0.0,               # not an episodic agent; kept for schema parity
        })
    return out


def main():
    with open(config.VOCAB_PATH) as f:
        vocab = [t for t in json.load(f) if t != "DONE"]

    model_path = config.resolve_embedding_model()
    encoder = SentenceTransformer(model_path)

    Xtr, Ytr, _ = load_split(config.TRAIN_DATA, vocab, encoder)
    Xva, Yva, _ = load_split(config.VAL_DATA, vocab, encoder)
    Xte, Yte, _ = load_split(config.TEST_DATA, vocab, encoder)
    print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}  |vocab|={len(vocab)}")

    clf = SupervisedToolSelector(n_tools=len(vocab)).fit(Xtr, Ytr)

    # Threshold tuned on VALIDATION only -- never on test.
    thresh = clf.calibrate_threshold(Xva, Yva)
    print(f"calibrated threshold on validation: {thresh:.3f}")

    preds_thresh = clf.predict(Xte)
    preds_topk = clf.predict(Xte, k=2)

    rec_thresh = records_from(preds_thresh, Yte)
    rec_topk = records_from(preds_topk, Yte)

    os.makedirs("results/raw", exist_ok=True)
    # Headline method: cardinality decided by the model.
    with open("results/raw/Supervised.json", "w") as f:
        json.dump([rec_thresh], f, indent=2)
    # Reported separately; k is information the RL agent is not given.
    with open("results/raw/Supervised-topk.json", "w") as f:
        json.dump([rec_topk], f, indent=2)

    for name, rec in [("Supervised (threshold, val-tuned)", rec_thresh),
                      ("Supervised (top-k=2, k given)", rec_topk)]:
        sem = np.mean([r["success"] for r in rec])
        f1 = np.mean([r["f1"] for r in rec])
        size = np.mean([r["steps"] - 1 for r in rec])
        print(f"{name:36s} SEM={sem:.3f}  F1={f1:.3f}  mean |pred|={size:.2f}")

    print("\nFor reference, from results/final_comparison.json:")
    with open("results/final_comparison.json") as f:
        fc = json.load(f)
    for k in ["PPO", "DenseRetrieval", "LLaMA-3.1-8B-Instruct"]:
        if k in fc:
            print(f"  {k:36s} SEM={fc[k]['success_mean']:.3f}  F1={fc[k]['f1_mean']:.3f}")

    print("\nwrote results/raw/Supervised.json and results/raw/Supervised-topk.json")
    print("Re-run scripts/recompute_stats.py to fold these into the significance tests.")


if __name__ == "__main__":
    main()
