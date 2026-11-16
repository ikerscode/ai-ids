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


def _epoch_seconds(ts: pd.Series) -> pd.Series:
    """Datetime series -> float seconds since 1970 (NaN for NaT).

    Independent of the datetime unit, which changed from nanoseconds (pandas 2) to
    microseconds (pandas 3), so integer division by 10**9 is not safe.
    """
    return (ts - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)


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
        out["ts"] = _epoch_seconds(ts)
        out["ts"] = out["ts"].ffill().fillna(0) + np.arange(len(out)) * 1e-6
    else:
        out["ts"] = np.arange(len(out), dtype=float) * 0.01
    out["label_raw"] = col("Label", "BENIGN").astype(str)
    out["label"] = out["label_raw"].map(map_label)
    from .schema import port_class
    out["dst_port_class"] = out["dst_port"].astype(int).map(port_class)
    missing = [f for f in FLOW_FEATURES if f not in out and f not in ("dst_port_class", "proto")]
    return out, {"format": "cic-ids2017", "has_time_and_ips": bool(has_time), "missing_features": missing}


# --------------------------------------------------------------------------- CIC-IDS2017 TrafficLabelling
# The "GeneratedLabelledFlows" / TrafficLabelling CSVs keep Flow ID, IPs, ports and the
# flow start Timestamp. Their quirks, and how they are handled here:
#   * labels contain a raw 0x96 byte (cp1252 en dash, "Web Attack \x96 Brute Force"): files
#     are read as Latin-1 (never fails) and dash look-alikes are folded to "-"
#   * "Fwd Header Length" appears twice (pandas reads the copy as "Fwd Header Length.1");
#     neither is used, only the columns in CIC_MAP are read
#   * Thursday-WorkingHours-Morning-WebAttacks ends with ~288k empty rows: dropped
#   * "Flow Bytes/s" / "Flow Packets/s" hold Infinity and NaN, a few durations and IATs are
#     negative: non-finite values become 0 and negatives are clipped to 0 in normalise()
#   * timestamps are day-first (03/07/2017 = 3 July) on a 12-hour clock without AM/PM,
#     and only Monday has seconds. See parse_cic_timestamp().
TL_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TL_COLUMNS = set(CIC_MAP) | {"Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port",
                             "Protocol", "Timestamp", "Label"}
FIRST_CAPTURE_HOUR = 8  # captures ran 08:42-17:10 local time; earlier clock hours are afternoon
DASHES = str.maketrans({"\x96": "-", "–": "-", "—": "-", "�": "-"})


def clean_label(raw) -> str:
    return re.sub(r"\s+", " ", str(raw).translate(DASHES)).strip()


def parse_cic_timestamp(ts: pd.Series, row_order: np.ndarray | None = None) -> np.ndarray:
    """CIC Timestamp strings -> seconds (naive local time treated as UTC, so hour-of-day stays local).

    * Day first: "4/7/2017 8:54" is 4 July 2017 (the capture week was 3-7 July).
    * 12-hour clock with no AM/PM marker: the lab captured from about 08:40 to 17:10, so clock
      hours 1-7 are 13:00-19:00 and 8-12 stay as they are. This fixes the afternoon files
      (Thursday infiltration, Friday port scan and DDoS) that otherwise sort before the morning.
    * Seconds: only Monday has them. Every day is cut to the minute so train (Monday) and test
      days have the same resolution, then flows inside one minute are spread evenly over that
      minute in recording order (row_order). Window features below one minute therefore use
      an approximate position inside the minute; ordering never looks ahead.
    """
    p = ts.astype(str).str.extract(r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2})")
    ok = p.notna().all(axis=1).to_numpy()
    p = p.fillna(0).astype(np.int64)
    hour = p[3].to_numpy()
    hour = np.where(hour < FIRST_CAPTURE_HOUR, hour + 12, hour)
    days = _epoch_seconds(pd.to_datetime(pd.DataFrame({"year": p[2].where(ok, 1970),
                                                      "month": p[1].where(ok, 1).clip(1, 12),
                                                      "day": p[0].where(ok, 1).clip(1, 31)}),
                                         errors="coerce")).to_numpy()
    minute = (days + hour * 3600 + p[4].to_numpy() * 60).astype(float)
    minute = np.where(ok, minute, np.nan)
    if row_order is None:
        row_order = np.arange(len(minute))
    out = minute.copy()
    good = np.flatnonzero(ok)
    if len(good):
        m = pd.Series(minute[good])
        rank = pd.Series(row_order[good]).groupby(m.to_numpy()).rank(method="first").to_numpy()
        count = m.groupby(m.to_numpy()).transform("size").to_numpy()
        out[good] = minute[good] + 60.0 * (rank - 1) / count
    return out


def read_cic_tl_file(path: str | Path) -> pd.DataFrame:
    """One TrafficLabelling CSV -> raw frame with the columns this project uses (names stripped)."""
    raw = pd.read_csv(path, encoding="latin-1", skipinitialspace=True, low_memory=False,
                      usecols=lambda c: str(c).strip() in TL_COLUMNS)
    raw = raw.rename(columns=lambda c: str(c).strip())
    raw = raw.dropna(subset=["Flow ID", "Label"], how="all")
    raw = raw[raw["Label"].notna() & raw["Timestamp"].notna()]
    raw["Label"] = raw["Label"].map(clean_label)
    return raw.reset_index(drop=True)


def tl_day_files(directory: str | Path) -> dict[str, list[Path]]:
    files = sorted(Path(directory).glob("*.csv"))
    out = {}
    for day in TL_DAYS:
        fs = [f for f in files if f.name.lower().startswith(day.lower())]
        # morning file first, then afternoon (port scan before DDoS on Friday)
        key = lambda f: (0 if "morning" in f.name.lower() else 1, 0 if "portscan" in f.name.lower() else 1, f.name)
        if fs:
            out[day] = sorted(fs, key=key)
    return out


def load_cic_tl_day(files: list[Path], log=print) -> pd.DataFrame:
    """All files of one capture day -> schema flows with context and window features.

    Window features must see every flow of the day, so they are computed here, before
    any downsampling. Files of one day are joined in capture order first so a window
    can run across the morning/afternoon file boundary.
    """
    from .windows import add_window_features
    parts = [read_cic_tl_file(f) for f in files]
    raw = pd.concat(parts, ignore_index=True)
    mapped, _ = load_cic(raw.drop(columns=["Timestamp"]))
    mapped["ts"] = parse_cic_timestamp(raw["Timestamp"])
    mapped["label_raw"] = raw["Label"].to_numpy()
    mapped["label"] = mapped["label_raw"].map(map_label)
    mapped = mapped[np.isfinite(mapped["ts"].to_numpy(float))]
    mapped = mapped.replace([np.inf, -np.inf], np.nan)
    label_raw = mapped["label_raw"].to_numpy()
    mapped["campaign"] = [f"{files[0].name.split('-')[0].lower()}:{lab}" if lab.upper() != "BENIGN" else ""
                          for lab in label_raw]
    mapped["sensor"] = "cic2017:" + files[0].name.split("-")[0].lower()
    mapped["_row"] = np.arange(len(mapped))
    mapped = mapped.sort_values(["ts", "_row"], kind="stable").reset_index(drop=True)
    label_raw = mapped["label_raw"].to_numpy()
    flows = normalise(mapped, compute_context=True)  # keeps the time order (stable sort on ts)
    flows["label_raw"] = label_raw
    flows["label_note"] = tag_label_suspects(flows)
    flows = add_window_features(flows)
    log(f"  {files[0].name.split('-')[0]}: {len(flows)} flows, {int((flows['label'] != BENIGN).sum())} attack")
    return flows


LABEL_NOTE_MARGIN = 60.0


def tag_label_suspects(flows: pd.DataFrame) -> np.ndarray:
    """Mark BENIGN-labelled flows that are very probably part of an attack (one capture day).

    CIC-IDS2017 labels some attack traffic as benign (Engelen et al., "Troubleshooting an
    intrusion detection dataset: the CICIDS2017 case study", 2021). Two patterns are tagged:
      * attack_pair: same two IP addresses as a labelled attack episode (one raw label), inside
        that episode's time span +-60 s. These are mostly reply-direction and retried flows.
      * post_infiltration: flows from the host that was infiltrated on Thursday, to internal
        addresses, while the labelled infiltration was going on: its internal port scan.
    Labels are not changed. Metrics are reported on the original labels, and again with these
    flows left out ("clean benign"); benign-only training never uses them.
    """
    note = np.full(len(flows), "", dtype=object)
    benign = (flows["label"] == BENIGN).to_numpy()
    if benign.all():
        return note
    src, dst = flows["src_ip"].astype(str).to_numpy(), flows["dst_ip"].astype(str).to_numpy()
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    ts = flows["ts"].to_numpy(float)
    pair = pd.Series(lo) + "|" + pd.Series(hi)
    att = pd.DataFrame({"pair": pair[~benign].to_numpy(), "ts": ts[~benign],
                        "lab": flows["label_raw"].to_numpy()[~benign]})
    spans = att.groupby(["lab", "pair"])["ts"].agg(["min", "max"]).reset_index()
    pair_b = pair.to_numpy()
    for p, g in spans.groupby("pair"):
        m = benign & (pair_b == p)
        if not m.any():
            continue
        inside = np.zeros(len(flows), dtype=bool)
        for lo_t, hi_t in zip(g["min"], g["max"]):
            inside |= (ts >= lo_t - LABEL_NOTE_MARGIN) & (ts <= hi_t + LABEL_NOTE_MARGIN)
        note[m & inside] = "attack_pair"
    inf = (flows["label"] == "infiltration").to_numpy()
    if inf.any():
        from .schema import is_internal
        hosts = {h for h in src[inf] if is_internal(h)}
        lo_t, hi_t = ts[inf].min() - LABEL_NOTE_MARGIN, ts[inf].max() + LABEL_NOTE_MARGIN
        internal_dst = np.array([is_internal(d) for d in dst])
        m = benign & np.isin(src, list(hosts)) & internal_dst & (ts >= lo_t) & (ts <= hi_t) & (note == "")
        note[m] = "post_infiltration"
    return note


def time_split(flows: pd.DataFrame, day: str, early_cut_hour: float = 9 + 10 / 60,
               monday_val: tuple = (14.5, 15.5), attack_train_frac: float = 0.3) -> np.ndarray:
    """Time-respecting split for one day of CIC-IDS2017.

    Monday (benign only): before 14:30 train, 14:30-15:30 val (threshold calibration),
    after 15:30 feedback (the simulated analyst reviews that hour).
    Tuesday-Friday: benign flows before 09:10, ahead of every attack, are train; everything
    later is test, except the chronologically first 30% of each attack label, which the
    supervised layer may learn from (as if those families had been seen earlier).
    Anomaly detectors only ever train on benign flows from before any attack.
    """
    hour = (flows["ts"].to_numpy(float) % 86400) / 3600
    benign = (flows["label"] == BENIGN).to_numpy()
    split = np.full(len(flows), "test", dtype=object)
    if day == "Monday":
        split[:] = "train"
        split[(hour >= monday_val[0]) & (hour < monday_val[1])] = "val"
        split[hour >= monday_val[1]] = "feedback"
        split[~benign] = "test"  # Monday has no attacks; keep any stray label out of training
        return split
    split[benign & (hour < early_cut_hour)] = "train"
    lab = flows["label_raw"].to_numpy()
    for raw_label in pd.unique(lab[~benign]):
        idx = np.flatnonzero(lab == raw_label)  # flows are already in time order
        split[idx[: int(len(idx) * attack_train_frac)]] = "train"
    return split


TL_CAPS = {("train", True): 70_000, ("train", False): 2_000, ("val", True): 20_000, ("feedback", True): 20_000,
           ("test", True): 40_000, ("test", False): 5_000}


def sample_with_weights(flows: pd.DataFrame, caps: dict = TL_CAPS, seed: int = 0) -> pd.DataFrame:
    """Cap each (split, raw label) stratum; 'weight' = flows it stands for in the full capture."""
    rng = np.random.default_rng(seed)
    keep, weight = [], []
    for (split, raw_label), idx in flows.groupby(["split", "label_raw"]).indices.items():
        cap = caps.get((split, str(raw_label).upper() == "BENIGN"), 5_000)
        take = idx if len(idx) <= cap else np.sort(rng.choice(idx, cap, replace=False))
        keep.append(take)
        weight.append(np.full(len(take), len(idx) / len(take)))
    order = np.concatenate(keep)
    out = flows.iloc[order].assign(weight=np.concatenate(weight))
    return out.sort_values("ts", kind="stable").reset_index(drop=True)


def _compact(df: pd.DataFrame) -> pd.DataFrame:
    """float32 features and categorical strings: the full-benign test set is ~1.8M rows."""
    from .schema import FEATURES
    from .windows import WINDOW_FEATURES
    cols = [c for c in FEATURES + WINDOW_FEATURES if c in df]
    df[cols] = df[cols].astype(np.float32)
    for c in ("src_host", "dst_host", "sensor", "label", "campaign", "split", "label_raw", "label_note"):
        if c in df:
            df[c] = df[c].astype("category")
    return df


def load_cic_tl(directory: str | Path, log=print, seed: int = 0, keep_test_days: bool = False):
    """The five TrafficLabelling days -> one sampled dataset with a time-respecting split.

    Returns (flows, meta), plus a third frame when keep_test_days is set: every benign test
    flow (no cap) with the same attack cap, for realistic alert-rate and incident numbers.
    """
    days = tl_day_files(directory)
    if not days:
        raise FileNotFoundError(f"no CIC-IDS2017 TrafficLabelling CSVs in {directory}")
    parts, full_counts, test_days, notes = [], {}, [], {}
    for day, files in days.items():
        flows = load_cic_tl_day(files, log=log)
        flows["split"] = time_split(flows, day)
        full_counts[day] = {str(k): int(v) for k, v in flows["label_raw"].value_counts().items()}
        notes[day] = {str(k): int(v) for k, v in flows["label_note"].value_counts().items() if k}
        parts.append(sample_with_weights(flows, seed=seed))
        if keep_test_days and day != "Monday":
            caps = {("test", True): 10**9, ("test", False): TL_CAPS[("test", False)]}
            test_days.append(_compact(sample_with_weights(flows[flows["split"] == "test"], caps, seed=seed)))
        del flows
    out = pd.concat(parts, ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)
    meta = {"format": "cic-ids2017-trafficlabelling", "has_time_and_ips": True, "window_features": True,
            "split_kind": "time", "full_counts": full_counts, "label_notes": notes, "missing_features": [],
            "split_rule": time_split.__doc__.strip(), "caps": {f"{k[0]}/{'benign' if k[1] else 'attack'}": v
                                                             for k, v in TL_CAPS.items()}}
    if keep_test_days:
        full = pd.concat(test_days, ignore_index=True) if test_days else out.iloc[:0]
        for c in full.columns:
            if isinstance(full[c].dtype, pd.CategoricalDtype):
                full[c] = full[c].astype(object)  # shares the category strings, no copy per row
        return out, meta, full.sort_values("ts", kind="stable").reset_index(drop=True)
    return out, meta


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
