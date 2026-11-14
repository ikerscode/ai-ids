"""Explanations for alerts: which features drove the decision, in plain sentences.

Four evidence sources are combined, each normalised so its contributions sum to 1:

* autoencoder   per-feature reconstruction error (relative to that feature's benign error)
* baseline      per-host surprisal / robust z-score components
* supervised    local perturbation: replace one feature with its benign median and
                measure how much the attack probability drops
* iforest       the same perturbation applied to the IsolationForest score

The combined weights drive the bar chart; the sentences compare the flow's value
with this host's own median when a host profile exists, else the benign median.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

from .attack import map_family, map_unknown
from .detectors import BASELINE_COMPONENTS, RATE_FEATURES, VOLUME_FEATURES, EntityBaseline, port_key
from .engine import UNKNOWN
from .models import ModelBundle
from .schema import FEATURES, PORT_CLASS_NAMES, is_internal

LABELS = {
    "duration": "flow duration (s)", "fwd_pkts": "packets sent", "bwd_pkts": "packets received",
    "fwd_bytes": "bytes sent", "bwd_bytes": "bytes received", "fwd_len_mean": "mean sent packet size",
    "fwd_len_std": "sent packet size spread", "fwd_len_max": "largest sent packet",
    "bwd_len_mean": "mean received packet size", "bwd_len_std": "received packet size spread",
    "bwd_len_max": "largest received packet", "pkt_len_mean": "mean packet size", "pkt_len_std": "packet size spread",
    "iat_mean": "mean inter-arrival time", "iat_std": "inter-arrival jitter", "iat_min": "shortest packet gap",
    "iat_max": "longest packet gap", "fwd_iat_mean": "sent packet gap", "bwd_iat_mean": "received packet gap",
    "syn_cnt": "SYN flags", "fin_cnt": "FIN flags", "rst_cnt": "RST flags", "psh_cnt": "PSH flags",
    "ack_cnt": "ACK flags", "urg_cnt": "URG flags", "bytes_per_s": "bytes/s", "pkts_per_s": "packets/s",
    "down_up_ratio": "download/upload ratio", "dst_port_class": "destination port class", "proto": "protocol",
    "ctx_src_flows_60s": "flows from this source in the last 60 s",
    "ctx_src_dports_60s": "distinct ports this source hit in 60 s",
    "ctx_src_dsts_60s": "distinct hosts this source hit in 60 s",
    "ctx_dst_flows_60s": "flows to this destination in 60 s",
    "ctx_pair_flows_1h": "repeat connections to this peer in 1 h",
    "ctx_pair_iat_cv": "regularity of repeat connections (CV)",
    "dst_port": "destination port", "peer": "peer address", "peer_port": "peer + port combination",
    "hour": "time of day",
}
COMPONENT_FEATURE = {"port": "dst_port", "peer": "peer", "service": "peer_port", "hour": "hour"}
SOURCE_WEIGHT = {"autoencoder": 1.0, "baseline": 1.0, "iforest": 0.7, "supervised": 1.0}


def _fmt(v: float) -> str:
    v = float(v)
    if abs(v) >= 1e6:
        return f"{v / 1e6:.1f}M"
    if abs(v) >= 1e4:
        return f"{v / 1e3:.0f}k"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    if abs(v) >= 1:
        return f"{v:.1f}"
    return f"{v:.3f}"


def _shares(m: np.ndarray) -> np.ndarray:
    m = np.clip(np.nan_to_num(m), 0, None)
    s = m.sum(axis=1, keepdims=True)
    return np.divide(m, s, out=np.zeros_like(m), where=s > 0)


class Explainer:
    def __init__(self, bundle: ModelBundle, baseline: EntityBaseline | None = None):
        self.b = bundle
        self.baseline = baseline or bundle.baseline

    def _perturb(self, x: np.ndarray, fn, fill: np.ndarray) -> np.ndarray:
        """Score drop when each feature is replaced by its benign median (one batch call)."""
        n, f = x.shape
        base = fn(x)
        big = np.repeat(x[:, None, :], f, axis=1)
        idx = np.arange(f)
        big[:, idx, idx] = fill[idx]
        pert = fn(big.reshape(n * f, f)).reshape(n, f)
        return base[:, None] - pert

    def explain(self, df: pd.DataFrame, res: pd.DataFrame) -> list[dict]:
        """df: flows (raw schema), res: matching Engine.detect rows. Returns one dict per row."""
        if df.empty:
            return []
        df = df.reset_index(drop=True)
        res = res.reset_index(drop=True)
        x = self.b.transform.transform(df)
        z = self.b.transform.scale(x)
        ae = _shares(np.log1p(self.b.autoencoder.feature_errors(z)))
        ifo = _shares(self._perturb(z, self.b.iforest.score, self.b.transform.scale(self.b.transform.log_median_[None, :])[0]))
        comps = self.baseline.components(df)
        bl = _shares(comps[BASELINE_COMPONENTS].to_numpy())
        sup = np.zeros_like(ae)
        if self.b.supervised is not None:
            sup = _shares(self._perturb(x, lambda m: self.b.supervised.predict(m)["p_attack"], self.b.transform.log_median_))

        names = list(FEATURES) + [COMPONENT_FEATURE[c] for c in ("port", "peer", "service", "hour")]
        col = {n: i for i, n in enumerate(names)}
        out = []
        for i in range(len(df)):
            r = res.iloc[i]
            known = r["alert_class"] not in ("", UNKNOWN)
            w = {"autoencoder": SOURCE_WEIGHT["autoencoder"] * min(1.0, r["z_autoencoder"] / 2),
                 "iforest": SOURCE_WEIGHT["iforest"] * min(1.0, r["z_iforest"] / 2),
                 "baseline": SOURCE_WEIGHT["baseline"] * min(1.0, r["z_baseline"] / 2),
                 "supervised": SOURCE_WEIGHT["supervised"] * (1.0 if known else 0.3) * float(r["p_attack"])}
            per_source = {s: np.zeros(len(names)) for s in w}
            per_source["autoencoder"][: len(FEATURES)] = ae[i]
            per_source["iforest"][: len(FEATURES)] = ifo[i]
            per_source["supervised"][: len(FEATURES)] = sup[i]
            for j, c in enumerate(BASELINE_COMPONENTS):
                per_source["baseline"][col[COMPONENT_FEATURE.get(c, c)]] += bl[i, j]
            total = sum(w[s] * per_source[s] for s in w)
            # A feature whose value is ordinary for this kind of service is weak evidence
            # even if a model leaned on it; push it down so the text stays meaningful.
            ref = self.b.transform.reference(df.iloc[i]["dst_port_class"])
            v = df.iloc[i][FEATURES].to_numpy(float)
            ordinary = np.where((ref == 0) & (v == 0), True,
                                (v >= 0.67 * ref) & (v <= 1.5 * ref) & (ref > 0))
            ordinary[FEATURES.index("dst_port_class")] = False
            total[: len(FEATURES)] *= np.where(ordinary, 0.05, 1.0)
            total = total / total.sum() if total.sum() > 0 else total
            top = np.argsort(-total)[:6]
            flow = df.iloc[i].to_dict()
            contribs = []
            for j in top:
                if total[j] <= 0.01:
                    continue
                feat = names[j]
                text, value, ref = self._sentence(feat, flow, comps.iloc[i])
                contribs.append({"feature": feat, "label": LABELS.get(feat, feat), "weight": round(float(total[j]), 3),
                                 "value": value, "reference": ref, "text": text,
                                 "sources": {s: round(float(w[s] * per_source[s][j]), 3) for s in w}})
            cls = r["alert_class"]
            if known:
                attack = map_family(cls)
            else:
                attack = map_unknown(flow, [c["feature"] for c in contribs[:4]])
            who = flow.get("src_host") or flow.get("src_ip")
            to = flow.get("dst_host") or flow.get("dst_ip")
            lead = f"{cls.replace('_', ' ')} from {who} to {to}:{int(flow.get('dst_port', 0))}"
            reasons = "; ".join(c["text"] for c in contribs[:3])
            out.append({
                "summary": f"{lead}. {reasons}." if reasons else lead,
                "contributions": contribs,
                "attack": attack,
                "detectors": {k: (round(float(r[k]), 3) if not isinstance(r[k], str) else r[k])
                              for k in ("z_iforest", "z_autoencoder", "z_baseline", "ens", "tau", "votes",
                                        "p_attack", "sup_family", "sup_conf")},
            })
        return out

    # ------------------------------------------------------------------ sentences
    def _sentence(self, feat: str, flow: dict, comp: pd.Series):
        ent = EntityBaseline._entities(pd.DataFrame([flow])).iloc[0]
        prof = self.baseline.profile_for(ent)
        own = ent in self.baseline.profiles
        if is_internal(flow.get("src_ip", "")):
            host = flow.get("src_host") or flow.get("src_ip")
        else:
            host = flow.get("dst_host") or flow.get("dst_ip")
        whose = "this host's" if own else "the benign"
        if feat == "dst_port":
            port = int(flow["dst_port"])
            c = prof.ports.get(port_key(port), 0)
            return (f"port {port} {'never seen' if c == 0 else f'used only {c} times'} for {host} during the baseline",
                    port, c)
        if feat in ("peer", "peer_port"):
            peer = flow["dst_ip"] if is_internal(flow["src_ip"]) else flow["src_ip"]
            c = prof.peers.get(peer, 0)
            what = f"{peer}:{int(flow['dst_port'])}" if feat == "peer_port" else peer
            return (f"{what} is {'a new peer' if c == 0 else f'a rare peer ({c} flows)'} for {host}", what, c)
        if feat == "hour":
            h = int((flow["ts"] // 3600) % 24)
            share = prof.hours[h] / max(prof.n, 1) * 100
            stamp = dt.datetime.fromtimestamp(flow["ts"], dt.timezone.utc).strftime("%H:%M")
            return (f"activity at {stamp} UTC; only {share:.1f}% of {host}'s baseline traffic is in this hour", stamp,
                    round(share, 2))
        v = float(flow.get(feat, 0.0))
        if feat in VOLUME_FEATURES + RATE_FEATURES and feat in prof.med:
            ref = math.expm1(prof.med[feat])
        else:
            ref = float(self.b.transform.reference(flow["dst_port_class"])[FEATURES.index(feat)])
            whose = f"the benign {PORT_CLASS_NAMES[int(flow['dst_port_class'])]}"
        label = LABELS.get(feat, feat)
        if feat == "dst_port_class":
            return (f"destination port class '{PORT_CLASS_NAMES[int(v)]}'", PORT_CLASS_NAMES[int(v)],
                    PORT_CLASS_NAMES[int(ref)] if int(ref) < len(PORT_CLASS_NAMES) else ref)
        if feat == "ctx_pair_iat_cv":
            return (f"repeat connections are very regular (interval CV {v:.2f}, benign median {ref:.2f})"
                    if v < ref else f"interval CV {v:.2f} vs benign median {ref:.2f}", round(v, 3), round(ref, 3))
        if ref > 0 and v / ref >= 1.5:
            return f"{label} {v / ref:.0f}x above {whose} median ({_fmt(v)} vs {_fmt(ref)})", round(v, 3), round(ref, 3)
        if v > 0 and ref > 0 and v / ref <= 0.67:
            return f"{label} {ref / v:.0f}x below {whose} median ({_fmt(v)} vs {_fmt(ref)})", round(v, 3), round(ref, 3)
        if ref == 0 and v > 0:
            return f"{label} is {_fmt(v)} where {whose} median is 0", round(v, 3), 0.0
        if v == 0 and ref > 0:
            return f"{label} is 0 where {whose} median is {_fmt(ref)}", 0.0, round(ref, 3)
        return f"{label} = {_fmt(v)} (median {_fmt(ref)}), unusual in combination", round(v, 3), round(ref, 3)
