"""Streaming mode: replay a dataset through the pipeline in (accelerated) real time.

A background thread advances a simulated clock by `speed` seconds per wall
second, feeds the flows and host events that "happened" in that slice to the
engine and host detector, folds hits into open alert groups and writes new or
updated alerts to the store, so the HUD sees them arrive.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import pandas as pd

from . import datasets
from .alerts import Aggregator, flow_hits, group_to_alert
from .engine import Engine
from .models import load_bundle
from .pipeline import explain_groups, load_host_detector
from .store import Store

TICK = 0.5  # wall seconds between steps
BUCKET = 600.0  # rate chart resolution in simulated seconds


class Streamer:
    def __init__(self, store: Store):
        self.store = store
        self.thread: threading.Thread | None = None
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()
        self.state = {"running": False, "dataset": "", "split": "", "speed": 600.0, "clock": 0.0, "start": 0.0,
                      "end": 0.0, "flows_done": 0, "flows_total": 0, "flow_alerts": 0, "alerts_new": 0,
                      "alerts_updated": 0, "error": ""}
        self.rate: deque = deque(maxlen=288)

    def status(self) -> dict:
        with self.lock:
            s = dict(self.state)
            s["rate"] = list(self.rate)
        s["progress"] = round((s["clock"] - s["start"]) / max(s["end"] - s["start"], 1), 3) if s["end"] else 0
        return s

    def set_speed(self, speed: float) -> None:
        with self.lock:
            self.state["speed"] = float(min(max(speed, 1.0), 20000.0))

    def start(self, dataset: str, split: str = "test", speed: float = 600.0) -> dict:
        self.stop()
        bundle = load_bundle()
        if bundle is None:
            raise RuntimeError("no trained model: train first")
        flows, events, _ = datasets.load_dataset(dataset)
        if split != "all":
            flows = flows[flows["split"] == split]
            events = events[events["split"] == split] if len(events) else events
        if flows.empty:
            raise ValueError(f"dataset {dataset!r} has no flows in split {split!r}")
        flows = flows.sort_values("ts")
        self.stop_flag.clear()
        with self.lock:
            self.state.update({"running": True, "dataset": dataset, "split": split, "start": float(flows["ts"].min()),
                               "end": float(flows["ts"].max()), "clock": float(flows["ts"].min()), "flows_done": 0,
                               "flows_total": int(len(flows)), "flow_alerts": 0, "alerts_new": 0,
                               "alerts_updated": 0, "error": ""})
            self.state["speed"] = float(min(max(speed, 1.0), 20000.0))
            self.rate.clear()
        self.thread = threading.Thread(target=self._run, args=(bundle, flows, events, dataset), daemon=True)
        self.thread.start()
        return self.status()

    def stop(self) -> None:
        self.stop_flag.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        with self.lock:
            self.state["running"] = False

    # ------------------------------------------------------------------ worker
    def _run(self, bundle, flows: pd.DataFrame, events: pd.DataFrame, dataset: str) -> None:
        try:
            engine = Engine(bundle, self.store.rules(), "feedback")
            host = load_host_detector()
            if host is not None:
                host.reset()
            net_agg, host_agg = Aggregator(), Aggregator()
            ts = flows["ts"].to_numpy()
            ev_ts = events["ts"].to_numpy() if len(events) else None
            i = j = 0
            clock = self.state["start"]
            while not self.stop_flag.is_set() and clock <= self.state["end"]:
                time.sleep(TICK)
                clock += TICK * self.state["speed"]
                k = int(ts.searchsorted(clock, side="right"))
                chunk = flows.iloc[i:k]
                i = k
                hits = []
                if len(chunk):
                    res = engine.detect(chunk)
                    hits = flow_hits(chunk, res)
                touched = net_agg.add(hits)
                explain_groups(touched, bundle, engine.baseline)
                self._persist(touched, "network", dataset)
                if host is not None and ev_ts is not None:
                    m = int(ev_ts.searchsorted(clock, side="right"))
                    if m > j:
                        self._persist(host_agg.add(host.process(events.iloc[j:m])), "hostlog", dataset)
                        j = m
                net_agg.expire(clock)
                host_agg.expire(clock)
                with self.lock:
                    self.state["clock"] = clock
                    self.state["flows_done"] = i
                    self.state["flow_alerts"] += len(hits)
                    bucket = clock - clock % BUCKET
                    if self.rate and self.rate[-1][0] == bucket:
                        b = self.rate[-1]
                        self.rate[-1] = (bucket, b[1] + len(chunk), b[2] + len(hits))
                    else:
                        self.rate.append((bucket, len(chunk), len(hits)))
        except Exception as exc:  # surface the problem in the HUD instead of dying silently
            with self.lock:
                self.state["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            with self.lock:
                self.state["running"] = False

    def _persist(self, groups: list[dict], source: str, dataset: str) -> None:
        for g in groups:
            alert = group_to_alert(g, source, dataset)
            if g["db_id"] is None:
                g["db_id"] = self.store.insert_alerts([alert])[0]
                with self.lock:
                    self.state["alerts_new"] += 1
            else:
                self.store.update_alert(g["db_id"], alert)
                with self.lock:
                    self.state["alerts_updated"] += 1
