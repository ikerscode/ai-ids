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


def train_models(dataset: str = "sim", store: Store | None = None, log=print, windows: bool | None = None) -> dict:
    t0 = time.time()
    flows, events, meta = datasets.load_dataset(dataset)
    train = flows[flows["split"] == "train"]
    val = flows[flows["split"] == "val"]
    extra_benign, extra_pos = feedback_flows(store, dataset, flows)
    if extra_pos is not None:
        train = pd.concat([train, extra_pos], ignore_index=True)
    if extra_benign is not None:
        log(f"adding {len(extra_benign)} analyst-confirmed false-positive flows to the benign set")
    bundle = train_bundle(train, val, extra_benign=extra_benign, log=log, windows=windows)
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


def evaluate_models(dataset: str = "sim", lofo: bool = True, log=print, windows: bool | None = None,
                    save_as: str = "eval.json") -> dict:
    flows, _, _ = datasets.load_dataset(dataset)
    bundle = load_bundle()
    if bundle is not None and bundle.meta.get("dataset") != dataset:
        bundle = None
    if bundle is not None and windows is not None and bool(bundle.meta.get("window_features")) != windows:
        bundle = None  # saved model was trained with the other feature set: train a fresh one
    out = run_evaluation(flows, bundle=bundle, lofo=lofo, log=log, windows=windows)
    out["dataset"] = dataset
    write_json(save_as, out)
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


# --------------------------------------------------------------------------- CIC-IDS2017 TrafficLabelling
def ingest_cic_tl(directory: str | Path, name: str = "cic17tl", log=print) -> dict:
    """Build three datasets from the TrafficLabelling CSVs.

    <name>        sampled flows, time-respecting split (train/val = Monday and pre-attack mornings)
    <name>-rand   the same flows with a random stratified split, for comparison with older results
    <name>-days   every benign flow of the Tuesday-Friday test period plus the sampled test attacks,
                  for alert-rate and incident runs at the real benign volume
    """
    t0 = time.time()
    flows, meta, days = datasets.load_cic_tl(directory, log=log, keep_test_days=True)
    meta.update({"kind": "csv", "source": str(Path(directory).name), "labels": flows["label"].value_counts().to_dict()})
    out = {"time": datasets.save_dataset(name, flows, None, meta)}
    rnd = datasets.assign_splits(flows)
    out["random"] = datasets.save_dataset(f"{name}-rand", rnd, None, {**meta, "split_kind": "random",
                                                                    "split_rule": "random stratified 45/20/10/25"})
    out["days"] = datasets.save_dataset(f"{name}-days", days, None, {**meta, "split_kind": "time-full-benign",
                                                                   "labels": days["label"].value_counts().to_dict()})
    log(f"datasets {out} written ({time.time() - t0:.0f}s)")
    return out


def incident_report(dataset: str, split: str = "test", bundle: ModelBundle | None = None, log=print) -> dict:
    """Detect one split, aggregate, explain, correlate, and check which true attack campaigns became incidents.

    Runs in memory (nothing is written to the HUD's alert store).
    """
    from .correlate import Correlator
    from .evaluate import active_hours
    t0 = time.time()
    bundle = bundle or load_bundle()
    if bundle is None:
        raise RuntimeError("no trained model: run train first")
    flows, _, _ = datasets.load_dataset(dataset)
    df = flows if split == "all" else flows[flows["split"] == split]
    engine = Engine(bundle, [], "feedback")
    res = engine.detect_stream(df)
    agg = Aggregator()
    agg.add(sorted(flow_hits(df, res), key=lambda h: h["ts"]))
    explain_groups(agg.history, bundle, engine.baseline)
    alerts = [group_to_alert(g, "network", dataset) for g in agg.history]
    for i, a in enumerate(sorted(alerts, key=lambda a: a["ts"]), start=1):
        a["id"] = i
    incidents = Correlator().correlate(alerts)
    by_id = {a["id"]: a for a in alerts}
    benign = (df["label"] == BENIGN).to_numpy()
    clean = (df["label_note"].astype(str) == "").to_numpy() if "label_note" in df else np.ones(len(df), bool)
    summary = []
    for inc in incidents:
        members = [by_id[i] for i in inc["alert_ids"]]
        attack_alerts = sum(1 for a in members if a["truth_attack_frac"] >= 0.5)
        summary.append({"id": inc["id"], "severity": inc["severity"], "score": inc["score"],
                        "day": time.strftime("%a", time.gmtime(inc["first_ts"])),
                        "start": time.strftime("%H:%M", time.gmtime(inc["first_ts"])),
                        "end": time.strftime("%H:%M", time.gmtime(inc["last_ts"])),
                        "n_alerts": inc["n_alerts"], "attack_alerts": attack_alerts,
                        "stages": list(inc["stages"]), "title": inc["title"], "campaigns": inc["campaigns"],
                        "classes": dict(pd.Series([a["alert_class"] for a in members]).value_counts())})
    campaigns = []
    for camp in sorted(c for c in df["campaign"].unique() if c):
        m = (df["campaign"] == camp).to_numpy()
        hits = [s for s in summary if camp in s["campaigns"]]
        best = max(hits, key=lambda s: s["score"]) if hits else None
        campaigns.append({"campaign": camp, "flows": int(m.sum()), "flows_alerted": round(float(res.loc[m, "alert"].mean()), 4),
                          "alerts": sum(s["campaigns"][camp] for s in hits), "incidents": len(hits),
                          "best_incident": best["id"] if best else "", "best_severity": best["severity"] if best else "",
                          "best_stages": best["stages"] if best else []})
    false_inc = [s for s in summary if s["attack_alerts"] == 0]
    # A "false" alert whose flows the loader tagged as probable mislabelled attack traffic.
    note = flows["label_note"].astype(str) if "label_note" in flows else None
    tagged = 0
    if note is not None:
        for a in alerts:
            if a["truth_attack_frac"] < 0.5 and a["rows"]:
                tagged += int((note.loc[a["rows"]] != "").mean() >= 0.5)
    hours = active_hours(df["ts"])
    out = {"dataset": dataset, "split": split, "flows": int(len(df)), "benign_flows": int(benign.sum()),
           "hours": round(hours, 1), "flow_alerts": int(res["alert"].sum()),
           "benign_flow_fpr": round(float(res.loc[benign, "alert"].mean()), 5) if benign.any() else None,
           "benign_flow_fpr_clean": round(float(res.loc[benign & clean, "alert"].mean()), 5) if benign.any() else None,
           "tagged_benign_flows": int((benign & ~clean).sum()),
           "alerts": len(alerts), "false_alerts": sum(1 for a in alerts if a["truth_attack_frac"] < 0.5),
           "false_alerts_per_hour": round(sum(1 for a in alerts if a["truth_attack_frac"] < 0.5) / hours, 2),
           "false_alerts_on_tagged_flows": tagged,
           "incidents": len(incidents), "false_incidents": len(false_inc),
           "false_incidents_by_severity": dict(pd.Series([s["severity"] for s in false_inc]).value_counts()),
           "campaigns": campaigns, "top_incidents": summary[:25], "window_features": bundle.meta.get("window_features"),
           "seconds": round(time.time() - t0, 1)}
    log(f"{dataset}/{split}: {out['alerts']} alerts ({out['false_alerts']} false), {out['incidents']} incidents "
        f"({out['false_incidents']} with no attack alert) in {out['seconds']}s")
    for c in campaigns:
        log(f"  {c['campaign']:42s} flows alerted {c['flows_alerted']:.3f}  incidents {c['incidents']}  "
            f"best {c['best_incident']} {c['best_severity']} {c['best_stages']}")
    return out


BENCH_COLUMNS = ["stage", "precision", "recall", "fpr", "pr_auc", "precision_w", "fpr_w", "false_alerts_per_hour_w",
                 "precision_clean", "fpr_clean", "precision_clean_w", "fpr_clean_w", "tagged_benign",
                 "agg_alerts_per_hour", "agg_alert_precision"]


def benchmark_cic(name: str = "cic17tl", lofo: bool = True, incidents: bool = True, log=print) -> dict:
    """Window-feature ablation on CIC-IDS2017 for both splits; writes data/benchmarks/<name>.json."""
    from .evaluate import run_evaluation
    out = {"dataset": name, "started": time.time(), "runs": [], "incidents": []}
    path = config.DATA_DIR / "benchmarks" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    for split_kind, ds in (("time", name), ("random", f"{name}-rand")):
        flows, _, _ = datasets.load_dataset(ds)
        for windows in (False, True):
            t0 = time.time()
            log(f"== {split_kind} split, window features {'on' if windows else 'off'}")
            train = flows[flows["split"] == "train"]
            bundle = train_bundle(train, flows[flows["split"] == "val"], log=log, windows=windows)
            bundle.meta["dataset"] = ds
            ev = run_evaluation(flows, bundle=bundle, lofo=lofo, log=log, windows=windows)
            run = {"split": split_kind, "dataset": ds, "window_features": windows, "n_features": len(bundle.features),
                   "test_flows": ev["test_flows"], "test_attack_flows": ev["test_attack_flows"],
                   "ablation": [{k: r.get(k) for k in BENCH_COLUMNS} for r in ev["ablation"]],
                   "lofo": ev.get("lofo", []), "per_family": ev["per_family"], "rules_learned": ev["rules_learned"],
                   "seconds": round(time.time() - t0, 1)}
            out["runs"].append(run)
            if incidents and split_kind == "time" and datasets.dataset_dir(f"{name}-days").exists():
                rep = incident_report(f"{name}-days", "test", bundle=bundle, log=log)
                rep["window_features"] = windows
                out["incidents"].append(rep)
            path.write_text(dumps(out))  # partial results survive an interrupted run
            del bundle
    out["seconds"] = round(time.time() - out["started"], 1)
    path.write_text(dumps(out))
    log(f"benchmark written to {path}")
    return out


def list_benchmarks() -> list[dict]:
    d = config.DATA_DIR / "benchmarks"
    if not d.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
