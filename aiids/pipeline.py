"""High-level operations shared by the CLI, the API and the streamer."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config, datasets
from .alerts import Aggregator, flow_hits, group_to_alert
from .engine import Engine
from .evaluate import confusion, flow_metrics, per_family, run_evaluation
from .explain import Explainer
from .hostdetect import HostDetector
from .models import ModelBundle, load_bundle, save_bundle, train_bundle
from .pcap import pcap_to_flows
from .schema import BENIGN
from .simulator import simulate
from .store import Store, dumps


def write_json(name: str, obj) -> None:
    config.ensure_dirs()
    (config.MODEL_DIR / name).write_text(dumps(obj))


def read_json(name: str):
    p = config.MODEL_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def simulate_dataset(name: str = "sim", days: int = 4, seed: int = 7, scale: float = 1.0, log=print) -> dict:
    t0 = time.time()
    flows, events, meta = simulate(days=days, seed=seed, scale=scale)
    meta["source"] = f"simulator seed={seed} scale={scale}"
    name = datasets.save_dataset(name, flows, events, meta)
    log(f"simulated {len(flows)} flows and {len(events)} host events over {days} days -> dataset '{name}' "
        f"({time.time() - t0:.1f}s)")
    return {"name": name, **datasets.stats(flows, events)}


def rules_for_engine(store: Store | None) -> list[dict]:
    return store.rules() if store else []


def feedback_flows(store: Store | None, dataset: str, flows: pd.DataFrame):
    """Flows the analyst labelled in the HUD: FP -> extra benign, TP of a known class -> extra positives."""
    if store is None:
        return None, None
    lab = [r for r in store.labelled_flows() if r["dataset"] == dataset]
    if not lab:
        return None, None
    fp_rows = [r["row"] for r in lab if r["verdict"] == "fp" and r["row"] < len(flows)]
    tp = [(r["row"], r["alert_class"]) for r in lab if r["verdict"] == "tp" and r["row"] < len(flows)]
    extra_benign = flows.iloc[fp_rows].assign(label=BENIGN) if fp_rows else None
    extra_pos = None
    known = set(flows["label"]) - {BENIGN}
    tp = [(row, cls) for row, cls in tp if cls in known]
    if tp:
        extra_pos = flows.iloc[[r for r, _ in tp]].copy()
        extra_pos["label"] = [c for _, c in tp]
    return extra_benign, extra_pos


def train_models(dataset: str = "sim", store: Store | None = None, log=print) -> dict:
    t0 = time.time()
    flows, events, meta = datasets.load_dataset(dataset)
    train = flows[flows["split"] == "train"]
    val = flows[flows["split"] == "val"]
    extra_benign, extra_pos = feedback_flows(store, dataset, flows)
    if extra_pos is not None:
        train = pd.concat([train, extra_pos], ignore_index=True)
    if extra_benign is not None:
        log(f"adding {len(extra_benign)} analyst-confirmed false-positive flows to the benign set")
    bundle = train_bundle(train, val, extra_benign=extra_benign, log=log)
    bundle.meta["dataset"] = dataset
    save_bundle(bundle)
    host = HostDetector().fit(events[events["split"] == "train"]) if len(events) else None
    joblib.dump(host, config.MODEL_DIR / "hostdet.joblib")
    test = flows[(flows["split"] == "test") & (flows["label"] != "unlabelled")]
    metrics = {"dataset": dataset, "trained_at": time.time(), "train_seconds": round(time.time() - t0, 1),
               "meta": bundle.meta, "thresholds": bundle.thresholds, "entity_thresholds": bundle.entity_tau}
    if len(test):
        res = Engine(bundle, rules_for_engine(store), "feedback").detect_stream(test)
        metrics.update({"test": flow_metrics(test, res), "per_family": per_family(test, res),
                        "confusion": confusion(test, res)})
        log(f"test split: precision {metrics['test']['precision']:.3f} recall {metrics['test']['recall']:.3f} "
            f"FPR {metrics['test']['fpr']:.4f} PR-AUC {metrics['test']['pr_auc']}")
    write_json("metrics.json", metrics)
    log(f"training done in {time.time() - t0:.1f}s, models saved to {config.MODEL_DIR}")
    return metrics


def evaluate_models(dataset: str = "sim", lofo: bool = True, log=print) -> dict:
    flows, _, _ = datasets.load_dataset(dataset)
    bundle = load_bundle()
    if bundle is not None and bundle.meta.get("dataset") != dataset:
        bundle = None
    out = run_evaluation(flows, bundle=bundle, lofo=lofo, log=log)
    out["dataset"] = dataset
    write_json("eval.json", out)
    return out


def load_host_detector() -> HostDetector | None:
    p = config.MODEL_DIR / "hostdet.joblib"
    return joblib.load(p) if p.exists() else None


def explain_groups(groups: list[dict], bundle: ModelBundle, baseline=None) -> None:
    """Attach explanations to the representative flow of each network alert group."""
    todo = [g for g in groups if g["best"].get("res") is not None and g.get("dirty_expl", True)]
    if not todo:
        return
    flows = pd.DataFrame([g["best"]["flow"] for g in todo])
    res = pd.DataFrame([g["best"]["res"] for g in todo])
    for g, e in zip(todo, Explainer(bundle, baseline).explain(flows, res)):
        g["best"]["explanation"] = e
        g["dirty_expl"] = False


def detect_dataset(dataset: str, split: str = "test", store: Store | None = None, log=print,
                   source: str | None = None) -> dict:
    """Batch-detect one split of a dataset and store the aggregated alerts."""
    bundle = load_bundle()
    if bundle is None:
        raise RuntimeError("no trained model: run train first")
    store = store or Store()
    flows, events, _ = datasets.load_dataset(dataset)
    df = flows if split == "all" else flows[flows["split"] == split]
    rules = store.rules()
    engine = Engine(bundle, rules, "feedback")
    res = engine.detect_stream(df)
    agg = Aggregator()
    agg.add(sorted(flow_hits(df, res), key=lambda h: h["ts"]))
    explain_groups(agg.history, bundle, engine.baseline)
    alerts = [group_to_alert(g, source or "network", dataset) for g in agg.history]
    host = load_host_detector()
    ev = events if split == "all" else events[events["split"] == split] if len(events) else events
    if host is not None and len(ev):
        host.reset()
        hagg = Aggregator()
        hagg.add(host.process(ev))
        alerts += [group_to_alert(g, "hostlog", dataset) for g in hagg.history]
    alerts.sort(key=lambda a: a["ts"])
    store.insert_alerts(alerts)
    hits = pd.Series(res.loc[res["suppressed"], "rule_id"]).value_counts().to_dict() if len(res) else {}
    store.bump_rule_hits({k: v for k, v in hits.items() if str(k).isdigit()})
    log(f"{dataset}/{split}: {int(res['alert'].sum()) if len(res) else 0} flow alerts -> {len(alerts)} stored alerts "
        f"({int(res['suppressed'].sum()) if len(res) else 0} suppressed by rules)")
    return {"flows": int(len(df)), "flow_alerts": int(res["alert"].sum()) if len(res) else 0,
            "alerts": len(alerts), "suppressed": int(res["suppressed"].sum()) if len(res) else 0}


def ingest_pcap(path: str | Path, name: str | None = None, store: Store | None = None, engine: str = "auto",
                log=print) -> dict:
    path = Path(path)
    flows = pcap_to_flows(path, engine=engine, sensor=f"pcap:{path.stem[:24]}")
    flows["split"] = "test"
    name = datasets.save_dataset(name or f"pcap-{path.stem}", flows, None,
                                 {"kind": "pcap", "source": path.name, "engine": engine})
    log(f"{path.name}: {len(flows)} flows -> dataset '{name}'")
    out = {"dataset": name, "flows": int(len(flows))}
    if len(flows) and load_bundle() is not None:
        out.update(detect_dataset(name, "all", store, log, source="pcap"))
    return out


def ingest_csv(path: str | Path, name: str | None = None, max_rows: int = 200_000, log=print) -> dict:
    path = Path(path)
    flows, meta = datasets.load_csv(path, max_rows=max_rows)
    name = datasets.save_dataset(name or f"csv-{path.stem}", flows, None, meta)
    log(f"{path.name}: {len(flows)} flows ({meta['format']}) -> dataset '{name}'; labels: {meta['labels']}")
    return {"dataset": name, "flows": int(len(flows)), "format": meta["format"], "labels": meta["labels"]}
