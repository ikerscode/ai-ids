"""Supervised classifier for known attack families."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .schema import BENIGN


class SupervisedDetector:
    """HistGradientBoosting over log-transformed features.

    Classes are re-weighted (capped) so rare families such as slow exfiltration
    are not drowned out by tens of thousands of benign flows.
    """

    def __init__(self, seed: int = 0, max_iter: int = 150):
        self.model = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=31,
                                                    l2_regularization=1.0, early_stopping=False,
                                                    random_state=seed)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SupervisedDetector":
        classes, counts = np.unique(y, return_counts=True)
        weight = {c: min(len(y) / (len(classes) * n), 50.0) for c, n in zip(classes, counts)}
        self.model.fit(x, y, sample_weight=np.array([weight[v] for v in y]))
        self.classes_ = list(self.model.classes_)
        self.benign_idx_ = self.classes_.index(BENIGN) if BENIGN in self.classes_ else None
        return self

    def predict(self, x: np.ndarray) -> dict:
        """Return p_attack, most likely attack family and its probability."""
        proba = self.model.predict_proba(x)
        attack = proba.copy()
        if self.benign_idx_ is not None:
            p_benign = proba[:, self.benign_idx_]
            attack[:, self.benign_idx_] = -1
        else:
            p_benign = np.zeros(len(x))
        best = attack.argmax(axis=1)
        return {"proba": proba, "p_attack": 1 - p_benign,
                "family": np.array(self.classes_, dtype=object)[best],
                "family_conf": attack[np.arange(len(x)), best]}
