"""Command line: python -m aiids {simulate,train,evaluate,stream,detect,ingest-pcap,ingest-csv,serve}."""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import config


def _table(rows: list[dict], cols: list[str]) -> str:
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    out = [line, "-" * len(line)]
    out += ["  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols) for r in rows]
    return "\n".join(out)


def cmd_simulate(a):
    from .pipeline import simulate_dataset
    out = simulate_dataset(a.name, a.days, a.seed, a.scale)
    print(json.dumps({k: out[k] for k in ("name", "flows", "events", "by_label")}, indent=1))


def cmd_train(a):
    from .pipeline import train_models
    from .store import Store
    train_models(a.dataset, Store())


def cmd_evaluate(a):
    from .pipeline import evaluate_models
    out = evaluate_models(a.dataset, lofo=not a.no_lofo)
    print("\nFP-reduction ablation (test split, %d flows, %.1f h):" % (out["test_flows"], out["test_hours"]))
    print(_table(out["ablation"], ["stage", "precision", "recall", "fpr", "pr_auc", "alerts_per_hour",
                                   "false_alerts_per_hour", "step_recall"]))
    if out.get("lofo"):
        print("\nLeave-one-attack-family-out:")
        print(_table(out["lofo"], ["held_out", "test_flows", "recall_when_trained", "recall_held_out",
                                   "via_anomaly_layer", "labelled_unknown_anomaly", "steps_detected", "benign_fpr"]))
    print(f"\nsaved to {config.MODEL_DIR / 'eval.json'} ({out['seconds']}s)")


def cmd_detect(a):
    from .pipeline import detect_dataset
    print(json.dumps(detect_dataset(a.dataset, a.split), indent=1))


def cmd_stream(a):
    from .store import Store
    from .stream import Streamer
    store = Store()
    s = Streamer(store)
    st = s.start(a.dataset, a.split, a.speed)
    print(f"replaying {a.dataset}/{a.split}: {st['flows_total']} flows at {a.speed:.0f}x (Ctrl-C to stop)")
    deadline = time.time() + a.seconds if a.seconds else None
    seen = max([x["id"] for x in store.list_alerts(limit=1)] or [0])
    try:
        while s.status()["running"] and (deadline is None or time.time() < deadline):
            time.sleep(1)
            for al in reversed(store.list_alerts(limit=50, since_id=seen)):
                seen = max(seen, al["id"])
                when = time.strftime("%H:%M:%S", time.gmtime(al["ts"]))
                print(f"[{when}] {al['severity']:8s} {al['alert_class']:18s} {al['technique_id']:10s} {al['summary'][:110]}")
    except KeyboardInterrupt:
        pass
    s.stop()
    st = s.status()
    print(f"stopped: {st['flows_done']}/{st['flows_total']} flows, {st['flow_alerts']} flow alerts, "
          f"{st['alerts_new']} new alerts{', error: ' + st['error'] if st['error'] else ''}")


def cmd_ingest_pcap(a):
    from .pipeline import ingest_pcap
    print(json.dumps(ingest_pcap(a.file, a.name, engine=a.engine), indent=1))


def cmd_ingest_csv(a):
    from .pipeline import ingest_csv
    print(json.dumps(ingest_csv(a.file, a.name, a.max_rows), indent=1, default=str))


def cmd_serve(a):
    import uvicorn
    uvicorn.run("aiids.api:app", host=a.host, port=a.port, log_level="warning")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m aiids", description="ML intrusion detection lab")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate", help="generate a labelled enterprise dataset")
    s.add_argument("--name", default="sim")
    s.add_argument("--days", type=int, default=4)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--scale", type=float, default=1.0, help="benign volume multiplier")
    s.set_defaults(fn=cmd_simulate)
    for name, fn, help_ in (("train", cmd_train, "train and calibrate all models"),
                            ("evaluate", cmd_evaluate, "ablation + leave-one-family-out evaluation")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--dataset", default="sim")
        if name == "evaluate":
            s.add_argument("--no-lofo", action="store_true")
        s.set_defaults(fn=fn)
    s = sub.add_parser("detect", help="batch-detect a dataset split and store alerts")
    s.add_argument("--dataset", default="sim")
    s.add_argument("--split", default="test")
    s.set_defaults(fn=cmd_detect)
    s = sub.add_parser("stream", help="replay a dataset through the pipeline in near real time")
    s.add_argument("--dataset", default="sim")
    s.add_argument("--split", default="test")
    s.add_argument("--speed", type=float, default=600.0, help="simulated seconds per wall second")
    s.add_argument("--seconds", type=float, default=0, help="stop after this many wall seconds")
    s.set_defaults(fn=cmd_stream)
    s = sub.add_parser("ingest-pcap", help="pcap -> flows dataset (+ detection if a model exists)")
    s.add_argument("file")
    s.add_argument("--name")
    s.add_argument("--engine", choices=["auto", "tshark", "scapy"], default="auto")
    s.set_defaults(fn=cmd_ingest_pcap)
    s = sub.add_parser("ingest-csv", help="CIC-IDS2017 / UNSW-NB15 CSV -> flows dataset")
    s.add_argument("file")
    s.add_argument("--name")
    s.add_argument("--max-rows", type=int, default=200_000)
    s.set_defaults(fn=cmd_ingest_csv)
    s = sub.add_parser("serve", help="start the HUD/API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=config.PORT)
    s.set_defaults(fn=cmd_serve)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
