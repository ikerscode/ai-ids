"""Dataset storage plus loaders for public IDS datasets.

A dataset is a folder data/datasets/<name>/ with flows.pkl (schema columns),
events.pkl (host/auth log events, may be empty) and meta.json.

Public datasets are mapped onto the same flow schema:
  * CIC-IDS2017 (CICFlowMeter CSVs, both "MachineLearningCVE" and "TrafficLabelling")
  * UNSW-NB15 (the 49-column raw CSVs and the training/testing-set CSVs)
Features a dataset does not provide are set to 0 and listed in meta["missing_features"].
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .schema import BENIGN, FLOW_FEATURES, normalise
from .simulator import SPLITS

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
EVENT_COLUMNS = ["ts", "host", "sensor", "event", "user", "src_ip", "detail", "label", "campaign", "split"]


def safe_name(name: str) -> str:
    name = re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-")[:48] or "dataset"
    if not NAME_RE.match(name):
        raise ValueError(f"bad dataset name {name!r}")
    return name


def dataset_dir(name: str) -> Path:
    return config.DATASET_DIR / safe_name(name)


def save_dataset(name: str, flows: pd.DataFrame, events: pd.DataFrame | None, meta: dict) -> str:
    d = dataset_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    flows.to_pickle(d / "flows.pkl")
    (events if events is not None else pd.DataFrame(columns=EVENT_COLUMNS)).to_pickle(d / "events.pkl")
    meta = {**meta, "name": safe_name(name), "flows": int(len(flows))}
    (d / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return safe_name(name)


def load_dataset(name: str):
    d = dataset_dir(name)
    if not (d / "flows.pkl").exists():
        raise FileNotFoundError(f"dataset {name!r} not found")
    flows = pd.read_pickle(d / "flows.pkl")
    events = pd.read_pickle(d / "events.pkl") if (d / "events.pkl").exists() else pd.DataFrame(columns=EVENT_COLUMNS)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    return flows, events, meta


def list_datasets() -> list[dict]:
    if not config.DATASET_DIR.exists():
        return []
    out = []
    for d in sorted(config.DATASET_DIR.iterdir()):
        if (d / "meta.json").exists():
            m = json.loads((d / "meta.json").read_text())
            out.append({"name": d.name, "kind": m.get("kind", ""), "flows": m.get("flows", 0),
                        "source": m.get("source", "")})
    return out


def stats(flows: pd.DataFrame, events: pd.DataFrame) -> dict:
    by = flows.groupby(["split", "label"]).size().reset_index(name="flows") if len(flows) else pd.DataFrame()
    return {
        "flows": int(len(flows)), "events": int(len(events)),
        "time_range": [float(flows["ts"].min()), float(flows["ts"].max())] if len(flows) else None,
        "by_label": flows["label"].value_counts().to_dict(),
        "by_split_label": by.to_dict("records"),
        "by_sensor": flows["sensor"].value_counts().to_dict(),
        "events_by_type": events["event"].value_counts().to_dict() if len(events) else {},
        "hosts": int(pd.concat([flows["src_host"], flows["dst_host"]]).replace("", np.nan).nunique()),
    }


def assign_splits(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Random stratified train/val/feedback/test split for datasets without one."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    split = np.empty(len(df), dtype=object)
    for _, idx in df.groupby("label").indices.items():
        idx = rng.permutation(idx)
        cuts = (np.array([0.45, 0.65, 0.75]) * len(idx)).astype(int)
        for part, name in zip(np.split(idx, cuts), SPLITS):
            split[part] = name
    df["split"] = split
    return df


def map_label(raw: str) -> str:
    s = str(raw).strip().lower()
    if s in ("", "nan", "benign", "normal", "0"):
        return BENIGN
    table = [("sql", "web_sqli"), ("web attack", "web_attack"), ("portscan", "portscan"),
             ("reconnaissance", "reconnaissance"), ("ssh", "ssh_bruteforce"), ("patator", "bruteforce"),
             ("brute", "bruteforce"), ("ddos", "dos"), ("dos", "dos"), ("bot", "botnet"),
             ("infiltration", "infiltration"), ("backdoor", "backdoor"), ("exploit", "exploits")]
    for needle, fam in table:
        if needle in s:
            return fam
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_") or "attack"


def _downsample(df: pd.DataFrame, max_rows: int, seed: int = 0) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    parts = []
    benign_cap = int(max_rows * 0.7)
    attack_cap = max(500, int(max_rows * 0.3 / max(df["label"].nunique() - 1, 1)))
    for lab, g in df.groupby("label"):
        cap = benign_cap if lab == BENIGN else attack_cap
        parts.append(g.sample(min(len(g), cap), random_state=seed))
    return pd.concat(parts).sort_index()


# --------------------------------------------------------------------------- CIC-IDS2017
CIC_MAP = {
    "Flow Duration": ("duration", 1e-6), "Total Fwd Packets": ("fwd_pkts", 1), "Total Backward Packets": ("bwd_pkts", 1),
    "Total Length of Fwd Packets": ("fwd_bytes", 1), "Total Length of Bwd Packets": ("bwd_bytes", 1),
    "Fwd Packet Length Mean": ("fwd_len_mean", 1), "Fwd Packet Length Std": ("fwd_len_std", 1),
    "Fwd Packet Length Max": ("fwd_len_max", 1), "Bwd Packet Length Mean": ("bwd_len_mean", 1),
    "Bwd Packet Length Std": ("bwd_len_std", 1), "Bwd Packet Length Max": ("bwd_len_max", 1),
    "Packet Length Mean": ("pkt_len_mean", 1), "Packet Length Std": ("pkt_len_std", 1),
    "Flow IAT Mean": ("iat_mean", 1e-6), "Flow IAT Std": ("iat_std", 1e-6), "Flow IAT Min": ("iat_min", 1e-6),
    "Flow IAT Max": ("iat_max", 1e-6), "Fwd IAT Mean": ("fwd_iat_mean", 1e-6), "Bwd IAT Mean": ("bwd_iat_mean", 1e-6),
    "SYN Flag Count": ("syn_cnt", 1), "FIN Flag Count": ("fin_cnt", 1), "RST Flag Count": ("rst_cnt", 1),
    "PSH Flag Count": ("psh_cnt", 1), "ACK Flag Count": ("ack_cnt", 1), "URG Flag Count": ("urg_cnt", 1),
    "Flow Bytes/s": ("bytes_per_s", 1), "Flow Packets/s": ("pkts_per_s", 1), "Down/Up Ratio": ("down_up_ratio", 1),
}


def load_cic(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.rename(columns=lambda c: str(c).strip())
    out = pd.DataFrame(index=df.index)
    for col, (name, factor) in CIC_MAP.items():
        if col in df:
            out[name] = pd.to_numeric(df[col], errors="coerce") * factor
    col = lambda c, d: df[c] if c in df else pd.Series(d, index=df.index)
    out["dst_port"] = pd.to_numeric(col("Destination Port", 0), errors="coerce").fillna(0)
    out["src_port"] = pd.to_numeric(col("Source Port", 0), errors="coerce").fillna(0)
    out["src_ip"] = col("Source IP", "0.0.0.0")
    out["dst_ip"] = col("Destination IP", "0.0.0.0")
    out["proto"] = pd.to_numeric(col("Protocol", 6), errors="coerce").fillna(6)
    has_time = "Timestamp" in df and "Source IP" in df
    if has_time:
        ts = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True, format="mixed")
        out["ts"] = (ts.astype("int64") // 10**9).astype(float).where(ts.notna(), np.nan)
        out["ts"] = out["ts"].ffill().fillna(0) + np.arange(len(out)) * 1e-6
    else:
        out["ts"] = np.arange(len(out), dtype=float) * 0.01
    out["label_raw"] = col("Label", "BENIGN").astype(str)
    out["label"] = out["label_raw"].map(map_label)
    from .schema import port_class
    out["dst_port_class"] = out["dst_port"].astype(int).map(port_class)
    missing = [f for f in FLOW_FEATURES if f not in out and f not in ("dst_port_class", "proto")]
    return out, {"format": "cic-ids2017", "has_time_and_ips": bool(has_time), "missing_features": missing}


# --------------------------------------------------------------------------- UNSW-NB15
UNSW_RAW_COLS = ["srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl",
                 "sloss", "dloss", "service", "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
                 "smean", "dmean", "trans_depth", "res_bdy_len", "sjit", "djit", "stime", "ltime", "sinpkt",
                 "dinpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",
                 "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
                 "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label"]
SERVICE_PORT = {"http": 80, "dns": 53, "ftp": 21, "ftp-data": 20, "smtp": 25, "ssh": 22, "pop3": 110, "snmp": 161,
                "ssl": 443, "irc": 6667, "dhcp": 67, "radius": 1812}
PROTO_NUM = {"tcp": 6, "udp": 17, "icmp": 1}


def load_unsw(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.rename(columns=lambda c: str(c).strip().lower().replace("smeansz", "smean").replace("dmeansz", "dmean"))
    num = lambda c, d=0.0: pd.to_numeric(df[c], errors="coerce").fillna(d) if c in df else pd.Series(d, index=df.index)
    out = pd.DataFrame(index=df.index)
    out["duration"] = num("dur")
    out["fwd_pkts"], out["bwd_pkts"] = num("spkts"), num("dpkts")
    out["fwd_bytes"], out["bwd_bytes"] = num("sbytes"), num("dbytes")
    out["fwd_len_mean"], out["bwd_len_mean"] = num("smean"), num("dmean")
    pk = (out["fwd_pkts"] + out["bwd_pkts"]).clip(lower=1)
    out["pkt_len_mean"] = (out["fwd_bytes"] + out["bwd_bytes"]) / pk
    out["fwd_iat_mean"], out["bwd_iat_mean"] = num("sinpkt") / 1000, num("dinpkt") / 1000
    out["iat_mean"] = out["duration"] / (pk - 1).clip(lower=1)
    out["iat_std"] = num("sjit") / 1000
    dur = out["duration"].clip(lower=1e-3)
    out["bytes_per_s"] = (out["fwd_bytes"] + out["bwd_bytes"]) / dur
    out["pkts_per_s"] = pk / dur
    out["down_up_ratio"] = out["bwd_bytes"] / out["fwd_bytes"].clip(lower=1)
    state = df["state"].astype(str).str.upper() if "state" in df else pd.Series("", index=df.index)
    out["fin_cnt"] = (state == "FIN").astype(int) * 2
    out["rst_cnt"] = (state == "RST").astype(int)
    out["syn_cnt"] = state.isin(["REQ", "INT", "FIN", "CON"]).astype(int)
    proto = df["proto"].astype(str).str.lower() if "proto" in df else pd.Series("tcp", index=df.index)
    out["proto"] = proto.map(PROTO_NUM).fillna(0)
    if "dsport" in df:
        out["dst_port"] = pd.to_numeric(df["dsport"], errors="coerce").fillna(0)
    else:
        out["dst_port"] = df.get("service", pd.Series("-", index=df.index)).astype(str).map(SERVICE_PORT).fillna(0)
    out["src_port"] = pd.to_numeric(df["sport"], errors="coerce").fillna(0) if "sport" in df else 0
    out["src_ip"] = df["srcip"] if "srcip" in df else "0.0.0.0"
    out["dst_ip"] = df["dstip"] if "dstip" in df else "0.0.0.0"
    has_time = "stime" in df and "srcip" in df
    out["ts"] = num("stime") if has_time else np.arange(len(df)) * 0.01
    cat = df["attack_cat"].astype(str) if "attack_cat" in df else pd.Series("", index=df.index)
    lab = num("label")
    out["label_raw"] = np.where(lab > 0, cat, "Normal")
    out["label"] = out["label_raw"].map(map_label)
    from .schema import port_class
    out["dst_port_class"] = out["dst_port"].astype(int).map(port_class)
    missing = [f for f in FLOW_FEATURES if f not in out]
    return out, {"format": "unsw-nb15", "has_time_and_ips": bool(has_time), "missing_features": missing}


def load_csv(path: str | Path, max_rows: int = 200_000) -> tuple[pd.DataFrame, dict]:
    """Detect the CSV format, map it to the schema and return (flows, meta)."""
    path = Path(path)
    head = pd.read_csv(path, nrows=5, low_memory=False, encoding_errors="replace")
    cols = {str(c).strip().lower() for c in head.columns}
    if "flow duration" in cols:
        raw = pd.read_csv(path, low_memory=False, encoding_errors="replace", skipinitialspace=True)
        mapped, meta = load_cic(raw)
    elif {"sbytes", "dbytes"} <= cols:
        mapped, meta = load_unsw(pd.read_csv(path, low_memory=False))
    elif len(head.columns) == 49:  # raw UNSW-NB15 files have no header row
        mapped, meta = load_unsw(pd.read_csv(path, header=None, names=UNSW_RAW_COLS, low_memory=False))
    else:
        raise ValueError("unrecognised CSV: expected CIC-IDS2017 or UNSW-NB15 columns")
    mapped = _downsample(mapped.replace([np.inf, -np.inf], np.nan), max_rows)
    mapped["sensor"] = f"csv:{path.stem[:32]}"
    flows = normalise(mapped, compute_context=meta["has_time_and_ips"])
    flows = assign_splits(flows)
    meta.update({"kind": "csv", "source": path.name, "rows_loaded": int(len(flows)),
                 "raw_labels": mapped["label_raw"].value_counts().head(30).to_dict(),
                 "labels": flows["label"].value_counts().to_dict()})
    if not meta["has_time_and_ips"]:
        meta["note"] = "no timestamps/IPs: context features are zero and per-host baselines fall back to global"
    return flows, meta
