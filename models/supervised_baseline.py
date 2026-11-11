"""
supervised_baseline.py

The baseline that decides whether any of the RL machinery is justified: a
memoryless classifier over the query representation alone.

On MetaTool this matches or beats masked PPO (SEM 0.588 vs 0.573 with a
validation-tuned threshold; 0.745 with top-k), which is why it belongs in the
paper as a first-class method rather than an afterthought. Any claim that
sequential decision-making helps has to clear this bar first.

Two selection rules are provided because they answer different questions:
  - top_k: cardinality supplied externally. Upper bound on what a classifier
    can do when the target-set size is known.
  - threshold: cardinality decided by the model, tuned on validation. The
    apples-to-apples comparison against an agent that must choose when to stop.
"""

from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


class SupervisedToolSelector:
    def __init__(self, n_tools: int, C: float = 10.0, max_iter: int = 2000):
        self.n_tools = n_tools
        self.clf = OneVsRestClassifier(LogisticRegression(max_iter=max_iter, C=C))
        self.threshold = 0.5
        self._fitted = False

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "SupervisedToolSelector":
        """X: (n, d) query features. Y: (n, n_tools) multi-hot gold sets."""
        # A tool absent from every training row makes OneVsRest fail; keep the
        # column set stable by tracking which labels actually vary.
        self._active = np.where(Y.sum(axis=0) > 0)[0]
        self.clf.fit(X, Y[:, self._active])
        self._fitted = True
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() first")
        proba = self.clf.predict_proba(X)
        full = np.zeros((X.shape[0], self.n_tools))
        full[:, self._active] = proba
        return full

    def predict(self, X: np.ndarray, k: Optional[int] = None):
        s = self.scores(X)
        out = []
        for row in s:
            if k is not None:
                out.append(set(np.argsort(-row)[:k].tolist()))
            else:
                pred = set(np.where(row >= self.threshold)[0].tolist())
                if not pred:
                    pred = {int(np.argmax(row))}
                out.append(pred)
        return out

    def calibrate_threshold(self, X_val: np.ndarray, Y_val: np.ndarray,
                            candidates: Optional[np.ndarray] = None) -> float:
        """Pick the threshold maximizing validation exact-match. Mirrors the
        stopping-rule calibration given to the retrieval baseline, so neither
        side gets tuning the other did not."""
        if candidates is None:
            candidates = np.linspace(0.05, 0.9, 60)
        gold = [set(np.where(row == 1)[0].tolist()) for row in Y_val]
        s = self.scores(X_val)
        best_t, best = self.threshold, -1.0
        for t in candidates:
            hits = 0
            for row, g in zip(s, gold):
                pred = set(np.where(row >= t)[0].tolist())
                if not pred:
                    pred = {int(np.argmax(row))}
                hits += (pred == g)
            acc = hits / max(1, len(gold))
            if acc > best:
                best, best_t = acc, float(t)
        self.threshold = best_t
        return best_t


def exact_match(preds, golds) -> float:
    return float(np.mean([1.0 if p == g else 0.0 for p, g in zip(preds, golds)]))


def macro_f1(preds, golds) -> float:
    out = []
    for p, g in zip(preds, golds):
        tp = len(p & g)
        prec = tp / max(1, len(p))
        rec = tp / max(1, len(g))
        out.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(out))
