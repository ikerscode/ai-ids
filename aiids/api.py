"""FastAPI backend for the HUD. Binds to 127.0.0.1 via run.sh."""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from .localguard import LocalGuard
from . import NAME, __version__, config, datasets, pipeline
from .correlate import Correlator
from .ingest import ExternalAlert, schema_doc, to_alert
from .models import load_bundle
from .schema import is_known_ip
from .store import Store
from .stream import Streamer

STATIC = config.ROOT / "static"
MAX_UPLOAD = 300 * 1024 * 1024

app = FastAPI(title="ai-ids", version=__version__)

app.add_middleware(LocalGuard)  # localhost-only: host allowlist, CSRF block, CSP
store = Store()
streamer = Streamer(store)
_incident_cache: dict = {"version": -1, "incidents": []}


# --------------------------------------------------------------------------- background jobs
class Jobs:
    """One heavy job at a time (simulate/train/evaluate/detect/upload), log captured for the HUD."""

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def busy(self) -> dict | None:
        return next((j for j in self.jobs.values() if j["status"] == "running"), None)

    def submit(self, kind: str, fn) -> dict:
        with self.lock:
            if (b := self.busy()) is not None:
                raise HTTPException(409, f"job '{b['kind']}' is still running")
            job = {"id": uuid.uuid4().hex[:8], "kind": kind, "status": "running", "started": time.time(),
                   "log": [], "result": None, "error": ""}
            self.jobs[job["id"]] = job

        def log(*parts):
            job["log"].append(" ".join(str(p) for p in parts))

        def run():
            try:
                job["result"] = fn(log)
                job["status"] = "done"
            except Exception as exc:
                job["status"] = "error"
                job["error"] = f"{type(exc).__name__}: {exc}"
                job["log"].append(traceback.format_exc(limit=3))
            job["finished"] = time.time()

        threading.Thread(target=run, daemon=True).start()
        return job


jobs = Jobs()


def _active_dataset() -> str:
    return store.get_setting("dataset", "sim")


# --------------------------------------------------------------------------- basics
@app.get("/api/health")
def health():
    return {"status": "ok", "name": NAME, "version": __version__}


@app.get("/api/status")
def status():
    bundle = load_bundle()
    return {"dataset": _active_dataset(), "datasets": datasets.list_datasets(), "trained": bundle is not None,
            "model": bundle.meta if bundle else None, "alerts": store.alert_counts(), "stream": streamer.status(),
            "job": jobs.busy(), "rules": len(store.rules())}


class SimulateReq(BaseModel):
    name: str = "sim"
    days: int = Field(4, ge=2, le=7)
    seed: int = 7
    scale: float = Field(1.0, gt=0.01, le=2.0)


@app.post("/api/simulate")
def simulate(req: SimulateReq):
    def fn(log):
        out = pipeline.simulate_dataset(req.name, req.days, req.seed, req.scale, log=log)
        store.set_setting("dataset", out["name"])
        return {"name": out["name"], "flows": out["flows"], "events": out["events"]}
    return jobs.submit("simulate", fn)


@app.get("/api/datasets")
def list_datasets():
    return {"active": _active_dataset(), "datasets": datasets.list_datasets()}


@app.get("/api/datasets/{name}")
def dataset_stats(name: str):
    try:
        flows, events, meta = datasets.load_dataset(name)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, str(exc))
    meta.pop("assets", None)
    return {"meta": meta, "stats": datasets.stats(flows, events)}


class NameReq(BaseModel):
    dataset: str = ""
    split: str = "test"
    lofo: bool = True


@app.post("/api/datasets/active")
def set_active(req: NameReq):
    datasets.load_dataset(req.dataset)
    store.set_setting("dataset", datasets.safe_name(req.dataset))
    return {"active": req.dataset}


@app.post("/api/upload")
async def upload(request: Request, kind: str, filename: str = "upload"):
    """Raw-body upload (no multipart dependency): ?kind=pcap|csv&filename=x.pcap"""
    if kind not in ("pcap", "csv"):
        raise HTTPException(400, "kind must be pcap or csv")
    config.ensure_dirs()
    stem = datasets.safe_name(Path(filename).stem)
    dest = config.UPLOAD_DIR / f"{stem}.{'pcap' if kind == 'pcap' else 'csv'}"
    size = 0
    with open(dest, "wb") as fh:
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_UPLOAD:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "file too large (300 MB max)")
            fh.write(chunk)
    if size == 0:
        raise HTTPException(400, "empty upload")

    def fn(log):
        if kind == "pcap":
            out = pipeline.ingest_pcap(dest, f"pcap-{stem}", store, log=log)
        else:
            out = pipeline.ingest_csv(dest, f"csv-{stem}", log=log)
        store.set_setting("dataset", out["dataset"])
        return out
    return jobs.submit(f"upload-{kind}", fn)


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    if job_id not in jobs.jobs:
        raise HTTPException(404, "no such job")
    return jobs.jobs[job_id]


# --------------------------------------------------------------------------- models
@app.post("/api/train")
def train(req: NameReq):
    ds = req.dataset or _active_dataset()
    return jobs.submit("train", lambda log: pipeline.train_models(ds, store, log=log))


@app.post("/api/evaluate")
def evaluate(req: NameReq):
    ds = req.dataset or _active_dataset()
    return jobs.submit("evaluate", lambda log: {"ablation": pipeline.evaluate_models(ds, req.lofo, log=log)["ablation"]})


@app.get("/api/models")
def models():
    bundle = load_bundle()
    return {"trained": bundle is not None, "meta": bundle.meta if bundle else None,
            "thresholds": bundle.thresholds if bundle else None,
            "entity_thresholds": bundle.entity_tau if bundle else None,
            "metrics": pipeline.read_json("metrics.json"), "evaluation": pipeline.read_json("eval.json")}


@app.get("/api/benchmarks")
def benchmarks():
    """Public-dataset benchmark results (python -m aiids benchmark), flattened for tables."""
    summary, lofo, incidents = [], [], []
    for b in pipeline.list_benchmarks():
        for run in b.get("runs", []):
            fb = next((r for r in run["ablation"] if r["stage"] == "feedback"), run["ablation"][-1] if run["ablation"] else {})
            held = [r["via_anomaly_layer"] for r in run.get("lofo", [])]
            summary.append({"dataset": run["dataset"], "split": run["split"], "windows": run["window_features"],
                            "precision": fb.get("precision"), "recall": fb.get("recall"), "fpr": fb.get("fpr"),
                            "precision_w": fb.get("precision_w"),
                            "mean_unseen_anomaly_recall": round(sum(held) / len(held), 4) if held else None})
            for r in run.get("lofo", []):
                lofo.append({"split": run["split"], "windows": run["window_features"], **r})
        for rep in b.get("incidents", []):
            for c in rep.get("campaigns", []):
                incidents.append({"windows": rep.get("window_features"), **c})
    return {"summary": summary, "lofo": lofo, "campaigns": incidents}


@app.post("/api/detect")
def detect(req: NameReq):
    ds = req.dataset or _active_dataset()
    return jobs.submit("detect", lambda log: pipeline.detect_dataset(ds, req.split, store, log=log))


# --------------------------------------------------------------------------- alerts + feedback
@app.get("/api/alerts")
def alerts(limit: int = 300, since_id: int = 0, cls: str = "", severity: str = "", source: str = "",
           status: str = "", q: str = ""):
    return store.list_alerts(min(limit, 2000), since_id, cls, severity, source, status, q)


@app.get("/api/alerts/{alert_id}")
def alert(alert_id: int):
    a = store.get_alert(alert_id)
    if a is None:
        raise HTTPException(404, "no such alert")
    return a


class FeedbackReq(BaseModel):
    verdict: str = Field(..., pattern="^(fp|tp)$")
    suppress: str = Field("none", pattern="^(none|exact|source)$")
    note: str = Field("", max_length=300)


@app.post("/api/alerts/{alert_id}/feedback")
def feedback(alert_id: int, req: FeedbackReq):
    a = store.get_alert(alert_id)
    if a is None:
        raise HTTPException(404, "no such alert")
    store.add_feedback(alert_id, req.verdict, req.note)
    store.set_status(alert_id, req.verdict)
    if a.get("dataset") and a.get("rows"):
        store.add_labelled_flows(alert_id, a["dataset"], a["rows"], req.verdict, a["alert_class"])
    rule_id, note = None, None
    if req.verdict == "fp" and req.suppress != "none" and not is_known_ip(a.get("src_ip")):
        note = "no suppression rule: alert has no usable source IP, so a rule would hide this class for every source"
    elif req.verdict == "fp" and req.suppress != "none":
        rule = {"src_ip": a["src_ip"], "alert_class": a["alert_class"],
                "reason": f"analyst FP on alert {alert_id}" + (f": {req.note}" if req.note else "")}
        if req.suppress == "exact":
            rule.update({"dst_ip": a["dst_ip"] or "*", "dst_port": a["dst_port"] or "*"})
        rule_id = store.add_rule(rule)
    return {"ok": True, "rule_id": rule_id, "note": note, "labelled_flows": len(a.get("rows") or [])}


@app.post("/api/alerts/clear")
def clear_alerts():
    streamer.stop()
    store.clear_alerts()
    return {"ok": True}


@app.get("/api/rules")
def rules():
    return {"rules": store.rules(), "feedback": store.feedback()[:100]}


@app.delete("/api/rules/{rule_id}")
def delete_rule(rule_id: int):
    store.delete_rule(rule_id)
    return {"ok": True}


# --------------------------------------------------------------------------- incidents
def _incidents() -> list[dict]:
    if _incident_cache["version"] != store.version:
        _incident_cache["incidents"] = Correlator().correlate(store.all_alerts_light())
        _incident_cache["version"] = store.version
    return _incident_cache["incidents"]


@app.get("/api/incidents")
def incidents(limit: int = 200):
    out = []
    for inc in _incidents()[:limit]:
        out.append({k: v for k, v in inc.items() if k not in ("graph", "timeline", "alert_ids")})
    return out


@app.get("/api/incidents/{inc_id}")
def incident(inc_id: str):
    for inc in _incidents():
        if inc["id"] == inc_id:
            return inc
    raise HTTPException(404, "no such incident")


# --------------------------------------------------------------------------- external ingest
@app.get("/api/ingest/schema")
def ingest_schema():
    return schema_doc()


@app.post("/api/ingest")
async def ingest(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "body must be JSON")
    items = body.get("alerts") if isinstance(body, dict) and "alerts" in body else [body]
    if not isinstance(items, list) or not 1 <= len(items) <= 500:
        raise HTTPException(400, "send one alert object or {\"alerts\": [1..500 objects]}")
    parsed, errors = [], []
    for i, item in enumerate(items):
        try:
            parsed.append(ExternalAlert.model_validate(item))
        except ValidationError as exc:
            errors.append({"index": i, "errors": [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]})
    if errors:
        return JSONResponse({"accepted": 0, "errors": errors}, status_code=422)
    for p in parsed:
        store.add_external(p.source, p.model_dump())
    ids = store.insert_alerts([to_alert(p) for p in parsed])
    linked = []
    for inc in _incidents():
        hit = set(ids) & set(inc["alert_ids"])
        if hit:
            linked.append({"incident": inc["id"], "alerts": sorted(hit), "n_alerts": inc["n_alerts"],
                           "severity": inc["severity"], "stages": list(inc["stages"])})
    return {"accepted": len(ids), "alert_ids": ids, "incidents": linked}


# --------------------------------------------------------------------------- streaming
class StreamReq(BaseModel):
    dataset: str = ""
    split: str = "test"
    speed: float = 600.0


@app.post("/api/stream/start")
def stream_start(req: StreamReq):
    try:
        return streamer.start(req.dataset or _active_dataset(), req.split, req.speed)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/stream/stop")
def stream_stop():
    streamer.stop()
    return streamer.status()


@app.post("/api/stream/speed")
def stream_speed(req: StreamReq):
    streamer.set_speed(req.speed)
    return streamer.status()


@app.get("/api/stream")
def stream_status():
    return streamer.status()


# --------------------------------------------------------------------------- static HUD
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


app.mount("/static", StaticFiles(directory=STATIC), name="static")
