"""Feature transform shared by every model.

Byte/packet/rate features are heavy-tailed (a backup moves 10^9 bytes, a DNS
query 80), so they are log1p-transformed first. The scaled version (z-scores
using benign training statistics) feeds the autoencoder and IsolationForest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import FEATURES, NON_LOG_FEATURES

LOG_MASK = np.array([f not in NON_LOG_FEATURES for f in FEATURES])
CLIP = 8.0


class FeatureTransform:
    def fit(self, benign: pd.DataFrame) -> "FeatureTransform":
        x = self.transform(benign)
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ < 1e-6] = 1.0
        raw = benign[FEATURES].to_numpy(float)
        self.raw_median_ = np.median(raw, axis=0)
        self.log_median_ = np.median(x, axis=0)
        # Medians per destination-port class (web, dns, smb...) so explanations can say
        # "3x above the benign DNS median" instead of comparing DNS with web traffic.
        pc = benign["dst_port_class"].to_numpy().astype(int)
        self.class_median_ = {int(c): np.median(raw[pc == c], axis=0) for c in np.unique(pc) if (pc == c).sum() >= 30}
        return self

    def reference(self, port_cls: int) -> np.ndarray:
        return getattr(self, "class_median_", {}).get(int(port_cls), self.raw_median_)

    @staticmethod
    def transform(df: pd.DataFrame) -> np.ndarray:
        x = df[FEATURES].to_numpy(float).copy()
        x[:, LOG_MASK] = np.log1p(np.clip(x[:, LOG_MASK], 0, None))
        return x

    def scale(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean_) / self.std_, -CLIP, CLIP)

    def scaled(self, df: pd.DataFrame) -> np.ndarray:
        return self.scale(self.transform(df))
