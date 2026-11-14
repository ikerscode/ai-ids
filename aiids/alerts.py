"""Alert deduplication/aggregation and the common alert record.

Every detection ("hit") from flows, host logs or external tools is grouped by a
class-specific key; hits closer than AGG_GAP_SECONDS merge into one alert. A
port scan of 600 ports becomes one alert with count=600 instead of 600 alerts.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from . import config
from .attack import STAGES, map_family
from .engine import UNKNOWN


# Low-and-slow activity (beacons, chunked exfil) repeats every few minutes; give it a longer merge gap.
SLOW_CLASSES = {"c2_beacon", "slow_exfil", UNKNOWN}


def agg_key(h: dict) -> tuple:
    cls = h["alert_class"]
    if cls in ("portscan", "dns_tunnel"):
        return (cls, h.get("src_ip", ""))
    if cls == "syn_flood":
        return (cls, h.get("dst_ip", ""), h.get("dst_port", 0))
    return (cls, h.get("src_ip", ""), h.get("dst_ip", ""), h.get("dst_host", ""))


def severity(stage: str, cls: str, conf: float, count: int) -> str:
    if conf < 0.6:
        return "low"
    if cls == UNKNOWN or stage == "recon":
        sev = "medium"
    else:
        sev = "high"
    if sev == "high" and conf >= 0.95 and count >= 20:
        sev = "critical"
    return sev


class Aggregator:
    def __init__(self, gap: float = config.AGG_GAP_SECONDS):
        self.gap = gap
        self.groups: dict[tuple, dict] = {}  # currently open group per key
        self.history: list[dict] = []          # every group ever created (batch use)

    def add(self, hits: list[dict]) -> list[dict]:
        """Fold hits (time-ordered) into groups. Returns the groups touched."""
        touched = {}
        for h in hits:
            key = agg_key(h)
            g = self.groups.get(key)
            gap = self.gap * 6 if h["alert_class"] in SLOW_CLASSES else self.gap
            if g is None or h["ts"] - g["end_ts"] > gap:
                g = {"key": key, "ts": h["ts"], "end_ts": h["ts"], "count": 0, "labels": Counter(),
                     "campaigns": Counter(), "rows": [], "dsts": set(), "ports": set(), "best": None,
                     "conf": 0.0, "db_id": None, "new": True, "dirty_expl": True}
                self.groups[key] = g
                self.history.append(g)
            g["end_ts"] = max(g["end_ts"], h["ts"])
            g["count"] += 1
            g["labels"][h.get("label", "")] += 1
            if h.get("campaign"):
                g["campaigns"][h["campaign"]] += 1
            if h.get("row") is not None and len(g["rows"]) < 500:
                g["rows"].append(int(h["row"]))
            g["dsts"].add(h.get("dst_ip", ""))
            g["ports"].add(h.get("dst_port", 0))
            g["conf"] = max(g["conf"], float(h.get("confidence", 0)))
            if g["best"] is None or h["score"] > g["best"]["score"]:
                g["best"] = h
                g["dirty_expl"] = True
            touched[id(g)] = g
        return list(touched.values())

    def expire(self, now: float) -> None:
        for k in [k for k, g in self.groups.items() if now - g["end_ts"] > self.gap * 8]:
            del self.groups[k]


def group_to_alert(g: dict, source: str, dataset: str = "") -> dict:
    """Turn an aggregation group into the normalised alert record stored and correlated."""
    h = g["best"]
    expl = h.get("explanation") or {}
    attack = expl.get("attack") or map_family(h["alert_class"])
    stage = attack.get("stage", "recon")
    truth, n_truth = g["labels"].most_common(1)[0] if g["labels"] else ("", 0)
    benign = g["labels"].get("benign", 0)
    flow = h.get("flow", {})
    summary = expl.get("summary", h["alert_class"])
    if g["count"] > 1:
        extra = f" ({g['count']} events"
        if len(g["ports"]) > 1:
            extra += f", {len(g['ports'])} ports"
        if len(g["dsts"]) > 1:
            extra += f", {len(g['dsts'])} destinations"
        summary += extra + ")"
    return {
        "ts": float(g["ts"]), "end_ts": float(g["end_ts"]), "source": source, "sensor": h.get("sensor", ""),
        "src_ip": h.get("src_ip", ""), "dst_ip": h.get("dst_ip", ""), "src_host": h.get("src_host", ""),
        "dst_host": h.get("dst_host", ""), "user": h.get("user", ""), "dst_port": int(h.get("dst_port", 0) or 0),
        "alert_class": h["alert_class"], "technique_id": attack.get("technique_id", ""),
        "technique": attack.get("technique", ""), "stage": stage if stage in STAGES else "recon",
        "rationale": attack.get("rationale", ""), "confidence": round(float(g["conf"]), 3),
        "severity": severity(stage, h["alert_class"], g["conf"], g["count"]), "score": round(float(h["score"]), 3),
        "count": int(g["count"]), "summary": summary, "explanation": expl, "raw": flow,
        "dataset": dataset, "rows": g["rows"][:200], "truth": truth,
        "truth_attack_frac": round(1 - benign / max(sum(g["labels"].values()), 1), 3),
        "campaign": g["campaigns"].most_common(1)[0][0] if g["campaigns"] else "",
        "rule_id": "", "status": "open",
    }


def flow_hits(df: pd.DataFrame, res: pd.DataFrame, source_rows: np.ndarray | None = None) -> list[dict]:
    """Alerted flows -> hit dicts for the aggregator."""
    m = res["alert"].to_numpy()
    if not m.any():
        return []
    sub, r = df[m], res[m]
    rows = source_rows[m] if source_rows is not None else sub.index.to_numpy()
    cols = ["ts", "src_ip", "dst_ip", "src_host", "dst_host", "dst_port", "sensor", "label", "campaign"]
    recs = sub[cols].to_dict("records")
    hits = []
    for rec, row, (_, rr), (_, flow) in zip(recs, rows, r.iterrows(), sub.iterrows()):
        rec.update({"alert_class": rr["alert_class"], "score": float(rr["score"]), "confidence": float(rr["confidence"]),
                    "row": int(row), "flow": flow.to_dict(), "res": rr.to_dict()})
        hits.append(rec)
    return hits
