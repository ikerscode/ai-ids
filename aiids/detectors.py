"""Unsupervised novelty detectors, all trained on benign traffic only.

* IForestDetector   - IsolationForest on scaled features (global outliers)
* AutoencoderDetector - MLPRegressor squeezed through a small bottleneck; flows
  it cannot reconstruct are unlike anything benign it saw
* EntityBaseline    - per-host behaviour profile (ports, peers, time of day,
  volumes) scored with surprisal and robust z-scores
* QuantileNormaliser - maps each detector's raw score onto a common scale
  (how far into the benign tail it is) so they can be averaged
"""
from __future__ import annotations

import math
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

from . import config
from .schema import FEATURES, is_internal


class IForestDetector:
    def __init__(self, seed: int = 0):
        self.model = IsolationForest(n_estimators=150, max_samples=512, random_state=seed, n_jobs=config.N_JOBS)

    def fit(self, z: np.ndarray) -> "IForestDetector":
        self.model.fit(z)
        return self

    def score(self, z: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(z)


class AutoencoderDetector:
    def __init__(self, seed: int = 0, n_features: int | None = None):
        n = n_features or len(FEATURES)
        self.model = MLPRegressor(hidden_layer_sizes=(24, 8, 24), activation="tanh", alpha=1e-4,
                                  learning_rate_init=3e-3, max_iter=120, batch_size=256,
                                  early_stopping=True, n_iter_no_change=8, random_state=seed)
        self.n = n

    def fit(self, z: np.ndarray) -> "AutoencoderDetector":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(z, z)
        err = self.feature_errors(z, normalise=False)
        # Per-feature reference error: some features are always harder to rebuild.
        self.ref_err_ = np.maximum(np.median(err, axis=0), 1e-3)
        return self

    def feature_errors(self, z: np.ndarray, normalise: bool = True) -> np.ndarray:
        err = (self.model.predict(z) - z) ** 2
        return err / self.ref_err_ if normalise else err

    def score(self, z: np.ndarray) -> np.ndarray:
        return np.log1p(self.feature_errors(z)).mean(axis=1)


# Volume-type features compared against each host's own median/MAD (log space).
VOLUME_FEATURES = ["fwd_bytes", "bwd_bytes", "duration", "fwd_pkts", "bwd_pkts", "pkts_per_s"]
RATE_FEATURES = ["ctx_src_flows_60s", "ctx_src_dports_60s", "ctx_src_dsts_60s", "ctx_dst_flows_60s"]
BASELINE_COMPONENTS = ["port", "peer", "service", "hour"] + VOLUME_FEATURES + RATE_FEATURES
GLOBAL = "__global__"


def entity_of(src_ip: str, dst_ip: str) -> str:
    """Which profile judges this flow: the internal client, else the internal server it hit."""
    if is_internal(src_ip):
        return f"src:{src_ip}"
    if is_internal(dst_ip):
        return f"in:{dst_ip}"
    return GLOBAL


def port_key(port: int) -> int:
    return int(port) if int(port) < 49152 else 49152


class _Profile:
    def __init__(self):
        self.n = 0
        self.ports: Counter = Counter()
        self.peers: Counter = Counter()
        self.services: Counter = Counter()
        self.hours = np.zeros(24)
        self.med: dict[str, float] = {}
        self.mad: dict[str, float] = {}

    def absorb(self, peers, ports, hours) -> None:
        self.n += len(peers)
        self.peers.update(peers)
        self.ports.update(ports)
        self.services.update(zip(peers, ports))
        np.add.at(self.hours, hours, 1)

    @staticmethod
    def surprisal(counter: Counter, keys, n: int) -> np.ndarray:
        """-log P(key) with Good-Turing mass for never-seen keys.

        P(unseen) = (#keys seen once) / n. A chatty workstation that visits a new
        website every few minutes has a high unseen mass, so novelty is cheap;
        a database server that only ever talks to three peers has almost none,
        so a brand new peer is very surprising. Seen keys are never scored as
        more surprising than unseen ones.
        """
        n = max(n, 1)
        singles = sum(1 for v in counter.values() if v == 1)
        p_unseen = max(singles / n, 0.5 / n)
        out = np.empty(len(keys))
        for i, k in enumerate(keys):
            c = counter.get(k, 0)
            out[i] = -math.log(max(c / n, p_unseen) if c else p_unseen)
        return np.minimum(out, 12.0)


class EntityBaseline:
    """Per-entity behavioural baselines (internal hosts as clients and as servers)."""

    MIN_FLOWS = 30

    def fit(self, df: pd.DataFrame) -> "EntityBaseline":
        self.profiles: dict[str, _Profile] = {}
        ent = self._entities(df)
        logs = self._log_volumes(df)
        groups = list(pd.Series(np.arange(len(df))).groupby(ent.to_numpy()))
        groups.append((GLOBAL, pd.Series(np.arange(len(df)))))
        for name, idx in groups:
            idx = idx.to_numpy()
            if name != GLOBAL and len(idx) < self.MIN_FLOWS:
                continue
            prof = _Profile()
            sub = df.iloc[idx]
            prof.absorb(*self._keys(sub))
            for col in VOLUME_FEATURES + RATE_FEATURES:
                v = logs[col][idx]
                med = float(np.median(v))
                prof.med[col] = med
                prof.mad[col] = float(np.median(np.abs(v - med))) * 1.4826
            self.profiles[name] = prof
        return self

    @staticmethod
    def _entities(df: pd.DataFrame) -> pd.Series:
        return pd.Series([entity_of(s, d) for s, d in zip(df["src_ip"], df["dst_ip"])], index=df.index)

    @staticmethod
    def _log_volumes(df: pd.DataFrame) -> dict[str, np.ndarray]:
        return {c: np.log1p(df[c].to_numpy(float)) for c in VOLUME_FEATURES + RATE_FEATURES}

    @staticmethod
    def _keys(df: pd.DataFrame):
        ent_is_src = [is_internal(s) for s in df["src_ip"]]
        peers = [d if e else s for s, d, e in zip(df["src_ip"], df["dst_ip"], ent_is_src)]
        ports = [port_key(p) for p in df["dst_port"]]
        hours = ((df["ts"].to_numpy(float) // 3600) % 24).astype(int)
        return peers, ports, hours

    def profile_for(self, entity: str) -> _Profile:
        return self.profiles.get(entity) or self.profiles[GLOBAL]

    def components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-flow component scores (surprisals and robust z-scores)."""
        out = pd.DataFrame(0.0, index=np.arange(len(df)), columns=BASELINE_COMPONENTS)
        if df.empty:
            return out
        ent = self._entities(df).to_numpy()
        logs = self._log_volumes(df)
        peers, ports, hours = self._keys(df)
        peers, ports = np.array(peers, dtype=object), np.array(ports)
        for name in pd.unique(ent):
            idx = np.flatnonzero(ent == name)
            prof = self.profile_for(name)
            gprof = self.profiles[GLOBAL]
            p, q = list(peers[idx]), list(ports[idx])
            out.iloc[idx, 0] = prof.surprisal(prof.ports, q, prof.n)
            out.iloc[idx, 1] = prof.surprisal(prof.peers, p, prof.n)
            out.iloc[idx, 2] = prof.surprisal(prof.services, list(zip(p, q)), prof.n)
            hp = (prof.hours[hours[idx]] + 1) / (prof.n + 24)
            out.iloc[idx, 3] = -np.log(hp) + math.log(1 / 24)  # 0 = as likely as a uniform hour
            for j, col in enumerate(VOLUME_FEATURES + RATE_FEATURES, start=4):
                med = prof.med.get(col, gprof.med[col])
                mad = max(prof.mad.get(col, gprof.mad[col]), 0.15)
                z = (logs[col][idx] - med) / mad
                out.iloc[idx, j] = np.clip(np.abs(z) if col in VOLUME_FEATURES else z, 0, 20)
        return out

    @staticmethod
    def combine(comp: pd.DataFrame) -> np.ndarray:
        novelty = comp["port"] + comp["peer"] + 0.5 * comp["service"] + 0.5 * comp["hour"].clip(lower=0)
        vol = np.sort(comp[VOLUME_FEATURES].to_numpy(), axis=1)[:, -2:].sum(axis=1)
        rate = comp[RATE_FEATURES].to_numpy().max(axis=1)
        return (novelty + 0.5 * vol + 0.5 * rate).to_numpy()

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.combine(self.components(df))

    def absorb(self, df: pd.DataFrame) -> None:
        """Adaptive update: fold flows judged benign into the profiles' counters."""
        if df.empty:
            return
        ent = self._entities(df).to_numpy()
        peers, ports, hours = self._keys(df)
        peers, ports = np.array(peers, dtype=object), np.array(ports)
        for name in pd.unique(ent):
            prof = self.profiles.get(name)
            if prof is None:
                continue
            idx = np.flatnonzero(ent == name)
            prof.absorb(list(peers[idx]), list(ports[idx]), hours[idx])


class QuantileNormaliser:
    """Map a raw score to 'benign tail surprise': -log10 of the share of benign scores above it.

    0 = typical benign, 2 = beyond the benign 99th percentile, 4 = beyond 99.99%.
    Scores past the largest benign score keep growing slowly so ranking is preserved.
    """

    def fit(self, scores: np.ndarray) -> "QuantileNormaliser":
        self.sorted_ = np.sort(np.asarray(scores, float))
        n = len(self.sorted_)
        self.med_ = float(self.sorted_[n // 2])
        self.max_ = float(self.sorted_[-1])
        return self

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, float)
        n = len(self.sorted_)
        above = n - np.searchsorted(self.sorted_, s, side="right")
        z = -np.log10((above + 0.5) / (n + 1))
        spread = max(self.max_ - self.med_, 1e-9)
        extra = np.log1p(np.clip((s - self.max_) / spread, 0, None))
        return z + extra
