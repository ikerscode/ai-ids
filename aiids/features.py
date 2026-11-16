"""Feature transform shared by every model.

Byte/packet/rate features are heavy-tailed (a backup moves 10^9 bytes, a DNS
query 80), so they are log1p-transformed first. The scaled version (z-scores
using benign training statistics) feeds the autoencoder and IsolationForest.

The transform owns the list of feature columns a bundle was trained on, so a
model trained with the time-window features and one trained without them can
live side by side (bundles saved before this option default to FEATURES).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import FEATURES, NON_LOG_FEATURES
from .windows import WINDOW_NON_LOG

LOG_MASK = np.array([f not in NON_LOG_FEATURES for f in FEATURES])
CLIP = 8.0


def feature_set(with_windows: bool) -> list[str]:
    from .windows import WINDOW_FEATURES
    return list(FEATURES) + (list(WINDOW_FEATURES) if with_windows else [])


class FeatureTransform:
    def __init__(self, features: list[str] | None = None):
        self.features_ = list(features or FEATURES)

    @property
    def features(self) -> list[str]:
        return getattr(self, "features_", FEATURES)

    @property
    def log_mask(self) -> np.ndarray:
        skip = NON_LOG_FEATURES | WINDOW_NON_LOG
        return np.array([f not in skip for f in self.features])

    def fit(self, benign: pd.DataFrame) -> "FeatureTransform":
        x = self.transform(benign)
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ < 1e-6] = 1.0
        raw = benign[self.features].to_numpy(float)
        self.raw_median_ = np.median(raw, axis=0)
        self.log_median_ = np.median(x, axis=0)
        # Medians per destination-port class (web, dns, smb...) so explanations can say
        # "3x above the benign DNS median" instead of comparing DNS with web traffic.
        pc = benign["dst_port_class"].to_numpy().astype(int)
        self.class_median_ = {int(c): np.median(raw[pc == c], axis=0) for c in np.unique(pc) if (pc == c).sum() >= 30}
        return self

    def reference(self, port_cls: int) -> np.ndarray:
        return getattr(self, "class_median_", {}).get(int(port_cls), self.raw_median_)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.features].to_numpy(float).copy()
        m = self.log_mask
        x[:, m] = np.log1p(np.clip(x[:, m], 0, None))
        return x

    def scale(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean_) / self.std_, -CLIP, CLIP)

    def scaled(self, df: pd.DataFrame) -> np.ndarray:
        return self.scale(self.transform(df))
