"""Per-flow decision logic, including every false-positive reduction stage.

Stages are cumulative so the evaluation can switch them on one at a time:

    naive          anomaly ensemble above its training 95th percentile, or supervised argmax != benign
    calibrated     thresholds chosen on benign validation traffic for a target FPR
    agreement      an anomaly alert also needs >= 2 of 3 detectors to vote
    host_adaptive  per-host thresholds + baselines that keep learning from traffic judged benign
    feedback       analyst suppression rules (and retraining with analyst-confirmed false positives)
    aggregated     flow alerts deduplicated into alert groups (done in alerts.py)
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from . import config
from .detectors import EntityBaseline
from .models import DETECTORS, ModelBundle

STAGES = ["naive", "calibrated", "agreement", "host_adaptive", "feedback", "aggregated"]
UNKNOWN = "unknown_anomaly"


def rule_matches(rule: dict, df: pd.DataFrame, classes: np.ndarray) -> np.ndarray:
    """Vectorised check of one suppression rule. '*' or missing = wildcard."""
    m = np.ones(len(df), dtype=bool)
    for field, col in (("src_ip", "src_ip"), ("dst_ip", "dst_ip")):
        v = rule.get(field, "*")
        if v not in ("*", "", None):
            m &= df[col].astype(str).to_numpy() == str(v)
    port = rule.get("dst_port", "*")
    if port not in ("*", "", None):
        m &= df["dst_port"].to_numpy() == int(port)
    cls = rule.get("alert_class", "*")
    if cls not in ("*", "", None):
        m &= classes == cls
    return m


class Engine:
    def __init__(self, bundle: ModelBundle, rules: list[dict] | None = None, stage: str = "feedback",
                 adaptive: bool = True):
        self.bundle = bundle
        self.rules = list(rules or [])
        self.level = STAGES.index(stage)
        self.adaptive = adaptive and self.level >= 3
        # The adaptive baseline drifts with traffic, so work on a private copy.
        self.baseline: EntityBaseline = copy.deepcopy(bundle.baseline) if self.adaptive else bundle.baseline

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a chunk of flows (assumed in time order) and decide which ones alert."""
        n = len(df)
        if n == 0:
            return pd.DataFrame()
        th = self.bundle.thresholds
        a = self.bundle.anomaly_scores(df, self.baseline)
        s = self.bundle.supervised_scores(df)
        ens = a["ens"]
        votes = sum((a["norm"][k] >= th["vote"][k]).astype(int) for k in DETECTORS)
        entities = EntityBaseline._entities(df).to_numpy()

        if self.level == 0:
            tau = np.full(n, th["tau_ens_naive"])
            tau_sup = 0.5
        else:
            tau = np.full(n, th["tau_ens"])
            tau_sup = th["tau_sup"]
        if self.level >= 3:
            tau = np.array([self.bundle.entity_tau.get(e, t) for e, t in zip(entities, tau)])

        known = s["p_attack"] >= tau_sup
        anomalous = ens >= tau
        if self.level >= 2:
            anomalous &= votes >= config.MIN_VOTES
        alert = known | anomalous

        family = np.asarray(s["family"], dtype=object)
        sure = s["family_conf"] >= config.MIN_SUPERVISED_CONF
        # Open-set rule: anomalous but the classifier does not recognise it -> unknown anomaly.
        cls = np.where(known | (anomalous & sure), family, UNKNOWN)
        cls = np.where(alert, cls, "")
        anom_conf = np.clip(0.5 + 0.15 * (ens - tau), 0.5, 0.99)
        conf = np.where(known, s["p_attack"], 0.0)
        conf = np.maximum(conf, np.where(anomalous, anom_conf, 0.0))

        vote_factor = np.where(votes >= config.MIN_VOTES, 1.0, 0.6) if self.level >= 2 else 1.0
        score = np.maximum(s["p_attack"] / tau_sup, ens / tau * vote_factor) / 2

        suppressed = np.zeros(n, dtype=bool)
        rule_id = np.array([""] * n, dtype=object)
        if self.level >= 4:
            for rule in self.rules:
                m = rule_matches(rule, df, cls) & alert & ~suppressed
                suppressed |= m
                rule_id[m] = str(rule.get("id", ""))
            score = np.where(suppressed, score * 0.1, score)
        alert &= ~suppressed

        out = pd.DataFrame({
            "entity": entities, "z_iforest": a["norm"]["iforest"], "z_autoencoder": a["norm"]["autoencoder"],
            "z_baseline": a["norm"]["baseline"], "ens": ens, "votes": votes, "tau": tau,
            "p_attack": s["p_attack"], "sup_family": family, "sup_conf": s["family_conf"],
            "known": known, "anomalous": anomalous, "alert": alert,
            "alert_class": np.where(alert, cls, ""), "confidence": np.where(alert, conf, 0.0),
            "suppressed": suppressed, "rule_id": rule_id, "score": score,
        }, index=df.index)
        if self.adaptive:
            self.baseline.absorb(df[~(alert | suppressed)])
        return out

    def detect_stream(self, df: pd.DataFrame, chunk_seconds: float = 3600.0) -> pd.DataFrame:
        """Run detect() over time-ordered chunks so the adaptive baseline evolves like it would live."""
        if df.empty:
            return pd.DataFrame()
        df = df.sort_values("ts", kind="stable")
        bins = ((df["ts"] - df["ts"].min()) // chunk_seconds).to_numpy()
        parts = [self.detect(df[bins == b]) for b in np.unique(bins)]
        return pd.concat(parts).loc[df.index]
