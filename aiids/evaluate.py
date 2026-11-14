"""Evaluation: metrics, FP-reduction ablation and leave-one-attack-family-out.

All numbers are computed on the held-out test split. Thresholds come from the
val split (benign only), analyst feedback from the feedback split, so nothing
is tuned on the data it is scored on.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import config
from .alerts import Aggregator, flow_hits, group_to_alert
from .engine import STAGES, UNKNOWN, Engine
from .models import ModelBundle, train_bundle
from .schema import BENIGN, is_known_ip
from .supervised import SupervisedDetector

STAGE_TEXT = {
    "naive": "No FP reduction (training 95th pct / argmax)",
    "calibrated": "+ thresholds calibrated on benign val",
    "agreement": "+ ensemble agreement (2 of 3 detectors)",
    "host_adaptive": "+ per-host adaptive baselines/thresholds",
    "feedback": "+ analyst feedback (rules + retrain)",
    "aggregated": "+ alert dedup/aggregation",
}


def _attack_mask(df: pd.DataFrame) -> np.ndarray:
    return (df["label"] != BENIGN).to_numpy()


def flow_metrics(df: pd.DataFrame, res: pd.DataFrame) -> dict:
    y = _attack_mask(df)
    a = res["alert"].to_numpy()
    tp, fp = int((a & y).sum()), int((a & ~y).sum())
    fn, tn = int((~a & y).sum()), int((~a & ~y).sum())
    hours = max((df["ts"].max() - df["ts"].min()) / 3600, 1e-9)
    return {
        "precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4),
        "fpr": round(fp / max(fp + tn, 1), 5),
        "pr_auc": round(float(average_precision_score(y, res["score"].to_numpy())), 4) if y.any() and (~y).any() else None,
        "alerts_per_hour": round(int(a.sum()) / hours, 2), "false_alerts_per_hour": round(fp / hours, 2),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "step_recall": step_recall(df, a),
    }


def step_recall(df: pd.DataFrame, alert: np.ndarray) -> float:
    """Share of ground-truth attack steps (campaign x family) with at least one alerted flow."""
    y = _attack_mask(df)
    if not y.any():
        return 0.0
    steps = df.loc[y, ["campaign", "label"]].assign(hit=alert[y])
    g = steps.groupby(["campaign", "label"])["hit"].any()
    return round(float(g.mean()), 4)


def aggregate_metrics(df: pd.DataFrame, res: pd.DataFrame) -> dict:
    """Alert-level numbers after deduplication (what an analyst actually sees)."""
    alerts = all_alerts(df, res)
    hours = max((df["ts"].max() - df["ts"].min()) / 3600, 1e-9)
    false = [a for a in alerts if a["truth_attack_frac"] < 0.5]
    return {"alerts": len(alerts), "alerts_per_hour": round(len(alerts) / hours, 2),
            "false_alerts_per_hour": round(len(false) / hours, 2),
            "alert_precision": round(1 - len(false) / max(len(alerts), 1), 4)}


def all_alerts(df: pd.DataFrame, res: pd.DataFrame, source: str = "network") -> list[dict]:
    agg = Aggregator()
    agg.add(sorted(flow_hits(df, res), key=lambda h: h["ts"]))
    return [group_to_alert(g, source) for g in agg.history]


# --------------------------------------------------------------------------- feedback oracle
def analyst_oracle(df: pd.DataFrame, res: pd.DataFrame, max_rules: int = 5, min_count: int = 3):
    """Simulate an analyst triaging one day of alerts.

    False positives are grouped by (source, class); the biggest groups become
    suppression rules (widened to '*' when they span several destinations or
    ports). Every false-positive flow is also handed back as extra benign data.
    """
    fp = df[res["alert"].to_numpy() & ~_attack_mask(df)].assign(alert_class=res.loc[res["alert"], "alert_class"])
    rules = []
    if len(fp):
        for (src, cls), g in sorted(fp.groupby(["src_ip", "alert_class"]), key=lambda kv: -len(kv[1]))[:max_rules]:
            if len(g) < min_count or not is_known_ip(src):
                continue  # no rule for an unknown source: it would suppress the class for everyone
            rules.append({"id": f"auto{len(rules) + 1}", "src_ip": src,
                          "dst_ip": g["dst_ip"].iloc[0] if g["dst_ip"].nunique() == 1 else "*",
                          "dst_port": int(g["dst_port"].iloc[0]) if g["dst_port"].nunique() == 1 else "*",
                          "alert_class": cls, "reason": f"analyst: {len(g)} false positives on feedback day"})
    return rules, fp.drop(columns=["alert_class"])


def with_supervised(bundle: ModelBundle, train: pd.DataFrame, val: pd.DataFrame, exclude: tuple) -> ModelBundle:
    """Copy of the bundle whose supervised layer never saw the excluded families."""
    b = copy.copy(bundle)
    rows = train[~train["label"].isin(set(exclude) | {"unlabelled", ""})]
    b.supervised = SupervisedDetector(seed=0, max_iter=80).fit(bundle.transform.transform(rows), rows["label"].to_numpy())
    vb = val[val["label"] == BENIGN]
    p = b.supervised.predict(bundle.transform.transform(vb))["p_attack"]
    b.thresholds = {**bundle.thresholds, "tau_sup": float(max(config.MIN_SUPERVISED_CONF,
                                                              np.quantile(p, 1 - config.TARGET_FPR_SUPERVISED)))}
    return b


def leave_one_family_out(bundle: ModelBundle, train, val, test, rules, log=print) -> list[dict]:
    rows = []
    full = Engine(bundle, rules, "feedback").detect_stream(test)
    families = sorted(set(train["label"]) - {BENIGN, "unlabelled", ""})
    for fam in families:
        t0 = time.time()
        b = with_supervised(bundle, train, val, (fam,))
        res = Engine(b, rules, "feedback").detect_stream(test)
        m = (test["label"] == fam).to_numpy()
        if not m.any():
            continue
        r = res[m]
        benign = ~_attack_mask(test)
        rows.append({
            "held_out": fam, "test_flows": int(m.sum()),
            "recall_when_trained": round(float(full.loc[m, "alert"].mean()), 4),
            "recall_held_out": round(float(r["alert"].mean()), 4),
            "via_supervised_other_class": round(float(r["known"].mean()), 4),
            "via_anomaly_layer": round(float(r["anomalous"].mean()), 4),
            "labelled_unknown_anomaly": round(float((r["alert_class"] == UNKNOWN).mean()), 4),
            "steps_detected": f"{int(test[m].assign(h=r['alert'].to_numpy()).groupby('campaign')['h'].any().sum())}"
                              f"/{test.loc[m, 'campaign'].nunique()}",
            "benign_fpr": round(float(res.loc[benign, "alert"].mean()), 5),
        })
        log(f"  LOFO {fam}: held-out recall {rows[-1]['recall_held_out']:.3f} "
            f"(anomaly layer {rows[-1]['via_anomaly_layer']:.3f}) [{time.time() - t0:.1f}s]")
    return rows


def confusion(test: pd.DataFrame, res: pd.DataFrame) -> dict:
    pred = np.where(res["alert"], res["alert_class"], BENIGN)
    labels = sorted(set(test["label"]))
    cols = labels + sorted(set(pred) - set(labels))
    m = pd.crosstab(pd.Series(test["label"].to_numpy(), name="true"), pd.Series(pred, name="pred"))
    m = m.reindex(index=labels, columns=cols, fill_value=0)
    return {"rows": labels, "cols": cols, "matrix": m.to_numpy().tolist()}


def per_family(test: pd.DataFrame, res: pd.DataFrame) -> list[dict]:
    out = []
    for fam, idx in test.groupby("label").indices.items():
        r = res.iloc[idx]
        out.append({"label": fam, "flows": int(len(idx)), "alerted": round(float(r["alert"].mean()), 4),
                    "correct_class": round(float((r["alert_class"] == fam).mean()), 4) if fam != BENIGN else None,
                    "unknown_anomaly": round(float((r["alert_class"] == UNKNOWN).mean()), 4)})
    return out


# --------------------------------------------------------------------------- full evaluation
def run_evaluation(flows: pd.DataFrame, bundle: ModelBundle | None = None, lofo: bool = True, log=print) -> dict:
    t0 = time.time()
    split = {s: flows[flows["split"] == s] for s in ("train", "val", "feedback", "test")}
    train, val, fb, test = split["train"], split["val"], split["feedback"], split["test"]
    test = test[test["label"] != "unlabelled"]
    if bundle is None:
        bundle = train_bundle(train, val, log=log)
    ablation, results = [], {}
    for stage in STAGES[:4]:
        res = Engine(bundle, [], stage).detect_stream(test)
        results[stage] = res
        ablation.append({"stage": stage, "description": STAGE_TEXT[stage], **flow_metrics(test, res),
                         **{f"agg_{k}": v for k, v in aggregate_metrics(test, res).items()}})
        log(f"  {stage:14s} P={ablation[-1]['precision']:.3f} R={ablation[-1]['recall']:.3f} "
            f"FPR={ablation[-1]['fpr']:.4f} alerts/h={ablation[-1]['alerts_per_hour']}")
    # Analyst feedback learned on the feedback day, applied to the test day.
    fb_res = Engine(bundle, [], "host_adaptive").detect_stream(fb) if len(fb) else pd.DataFrame()
    rules, extra = analyst_oracle(fb, fb_res) if len(fb) else ([], fb)
    log(f"  analyst oracle: {len(rules)} rules, {len(extra)} false-positive flows returned as benign")
    fb_bundle = train_bundle(train, val, extra_benign=extra, log=lambda *a: None) if len(extra) else bundle
    res = Engine(fb_bundle, rules, "feedback").detect_stream(test)
    results["feedback"] = res
    agg = aggregate_metrics(test, res)
    fm = flow_metrics(test, res)
    ablation.append({"stage": "feedback", "description": STAGE_TEXT["feedback"], **fm,
                     **{f"agg_{k}": v for k, v in agg.items()}})
    ablation.append({"stage": "aggregated", "description": STAGE_TEXT["aggregated"], **fm,
                     "alerts_per_hour": agg["alerts_per_hour"], "false_alerts_per_hour": agg["false_alerts_per_hour"],
                     **{f"agg_{k}": v for k, v in agg.items()}})
    log(f"  feedback       P={fm['precision']:.3f} R={fm['recall']:.3f} FPR={fm['fpr']:.4f} "
        f"alerts/h={fm['alerts_per_hour']} -> aggregated alerts/h={agg['alerts_per_hour']}")
    out = {
        "evaluated_at": time.time(), "test_flows": int(len(test)),
        "test_hours": round((test["ts"].max() - test["ts"].min()) / 3600, 2),
        "test_attack_flows": int(_attack_mask(test).sum()),
        "ablation": ablation, "rules_learned": rules, "extra_benign_flows": int(len(extra)),
        "final": {**fm, **agg}, "confusion": confusion(test, res), "per_family": per_family(test, res),
        "thresholds": bundle.thresholds,
    }
    if lofo:
        out["lofo"] = leave_one_family_out(fb_bundle, train, val, test, rules, log)
    out["seconds"] = round(time.time() - t0, 1)
    return out
