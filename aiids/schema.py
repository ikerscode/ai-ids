"""The single flow schema every data source is normalised to.

A flow is a bidirectional 5-tuple conversation. The "forward" direction is the
side that sent the first packet (normally the client). Every source (pcap,
simulator, CIC-IDS2017, UNSW-NB15) ends up as a DataFrame with ENTITY_COLUMNS,
LABEL_COLUMNS and FEATURES.
"""
from __future__ import annotations

import functools
import ipaddress
import math
from collections import Counter, deque

import numpy as np
import pandas as pd

from . import config

# Per-flow statistics computed from the packets of one flow.
FLOW_FEATURES = [
    "duration", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes",
    "fwd_len_mean", "fwd_len_std", "fwd_len_max",
    "bwd_len_mean", "bwd_len_std", "bwd_len_max",
    "pkt_len_mean", "pkt_len_std",
    "iat_mean", "iat_std", "iat_min", "iat_max", "fwd_iat_mean", "bwd_iat_mean",
    "syn_cnt", "fin_cnt", "rst_cnt", "psh_cnt", "ack_cnt", "urg_cnt",
    "bytes_per_s", "pkts_per_s", "down_up_ratio", "dst_port_class", "proto",
]
# Causal context features: computed only from flows seen *before* this one.
CONTEXT_FEATURES = [
    "ctx_src_flows_60s", "ctx_src_dports_60s", "ctx_src_dsts_60s",
    "ctx_dst_flows_60s", "ctx_pair_flows_1h", "ctx_pair_iat_cv",
]
FEATURES = FLOW_FEATURES + CONTEXT_FEATURES

ENTITY_COLUMNS = ["ts", "end_ts", "src_ip", "dst_ip", "src_port", "dst_port",
                  "src_host", "dst_host", "sensor"]
LABEL_COLUMNS = ["label", "campaign", "split"]
ALL_COLUMNS = ENTITY_COLUMNS + LABEL_COLUMNS + FEATURES

# Features that are categorical codes or ratios and should not be log-transformed.
NON_LOG_FEATURES = {"dst_port_class", "proto", "ctx_pair_iat_cv", "down_up_ratio"}

BENIGN = "benign"

# TCP flag bits as they appear in the TCP header.
FIN, SYN, RST, PSH, ACK, URG = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20

PORT_CLASSES = [
    ("web", {80, 443, 8080, 8443, 8000}),
    ("dns", {53}),
    ("smb_rpc", {135, 137, 138, 139, 445}),
    ("remote_admin", {22, 23, 3389, 5985, 5986}),
    ("database", {1433, 1521, 3306, 5432, 6379, 27017}),
    ("directory", {88, 389, 636, 3268, 464}),
    ("mail", {25, 110, 143, 465, 587, 993, 995}),
    ("ot", {502, 102, 20000, 44818, 5020}),
    ("infra", {123, 161, 162, 514, 873, 67, 68}),
]
PORT_CLASS_NAMES = [n for n, _ in PORT_CLASSES] + ["other_wellknown", "registered", "ephemeral"]


def port_class(port: int) -> int:
    """Map a destination port to a small integer class (see PORT_CLASS_NAMES)."""
    port = int(port)
    for i, (_, ports) in enumerate(PORT_CLASSES):
        if port in ports:
            return i
    if port < 1024:
        return len(PORT_CLASSES)
    if port < 49152:
        return len(PORT_CLASSES) + 1
    return len(PORT_CLASSES) + 2


# Python's is_private also covers the documentation ranges (192.0.2/24, 198.51.100/24,
# 203.0.113/24) that the simulator uses for "internet" attackers, so list the real
# internal ranges explicitly.
INTERNAL_NETS = [ipaddress.ip_network(n) for n in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8", "169.254.0.0/16",
    "fc00::/7", "fe80::/10", "::1/128")]


@functools.lru_cache(maxsize=65536)
def is_internal(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(str(ip))
    except ValueError:
        return False
    return any(addr in net for net in INTERNAL_NETS if net.version == addr.version)


def is_known_ip(ip) -> bool:
    """False for missing or placeholder addresses (e.g. CSV datasets without IPs).

    A suppression rule keyed on an unknown source would match every flow, so
    feedback must never turn one into a rule.
    """
    try:
        addr = ipaddress.ip_address(str(ip).strip())
    except ValueError:
        return False
    return not addr.is_unspecified


def flow_features(ts, sizes, fwd, flags, proto: int, dst_port: int) -> dict:
    """Compute FLOW_FEATURES from the packets of one flow.

    ts: packet timestamps (seconds), sizes: frame lengths, fwd: bool array
    (True = initiator to responder), flags: TCP flag byte per packet (0 for UDP).
    """
    ts = np.asarray(ts, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    fwd = np.asarray(fwd, dtype=bool)
    flags = np.asarray(flags, dtype=np.int64)
    order = np.argsort(ts, kind="stable")
    ts, sizes, fwd, flags = ts[order], sizes[order], fwd[order], flags[order]
    duration = float(ts[-1] - ts[0]) if len(ts) else 0.0
    fs, bs = sizes[fwd], sizes[~fwd]
    iat = np.diff(ts)
    f_mean, f_std, f_max, f_sum = _stats(fs)
    b_mean, b_std, b_max, b_sum = _stats(bs)
    p_mean, p_std, _, total_bytes = _stats(sizes)
    i_mean, i_std, i_max, _ = _stats(iat)
    flag_counts = [int(np.count_nonzero(flags & bit)) for bit in (SYN, FIN, RST, PSH, ACK, URG)]
    dur_floor = max(duration, 1e-3)
    return {
        "duration": duration,
        "fwd_pkts": int(len(fs)), "bwd_pkts": int(len(bs)),
        "fwd_bytes": f_sum, "bwd_bytes": b_sum,
        "fwd_len_mean": f_mean, "fwd_len_std": f_std, "fwd_len_max": f_max,
        "bwd_len_mean": b_mean, "bwd_len_std": b_std, "bwd_len_max": b_max,
        "pkt_len_mean": p_mean, "pkt_len_std": p_std,
        "iat_mean": i_mean, "iat_std": i_std,
        "iat_min": float(iat.min()) if len(iat) else 0.0, "iat_max": i_max,
        "fwd_iat_mean": _stats(np.diff(ts[fwd]))[0], "bwd_iat_mean": _stats(np.diff(ts[~fwd]))[0],
        "syn_cnt": flag_counts[0], "fin_cnt": flag_counts[1], "rst_cnt": flag_counts[2],
        "psh_cnt": flag_counts[3], "ack_cnt": flag_counts[4], "urg_cnt": flag_counts[5],
        "bytes_per_s": total_bytes / dur_floor, "pkts_per_s": len(ts) / dur_floor,
        "down_up_ratio": b_sum / max(f_sum, 1.0),
        "dst_port_class": port_class(dst_port), "proto": int(proto),
    }


def _stats(a: np.ndarray) -> tuple[float, float, float, float]:
    """(mean, std, max, sum) with as few numpy calls as possible (flows are tiny)."""
    n = len(a)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    s = float(a.sum())
    mean = s / n
    var = float(np.dot(a, a)) / n - mean * mean
    return mean, math.sqrt(var) if var > 0 else 0.0, float(a.max()), s


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add CONTEXT_FEATURES using only flows that started before each flow.

    These capture behaviour no single flow shows: fan-out (scans), bursts
    (brute force, floods) and regular repetition between a pair (beaconing).
    """
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    n = len(df)
    out = {c: np.zeros(n) for c in CONTEXT_FEATURES}
    out["ctx_pair_iat_cv"][:] = 1.0
    ts = df["ts"].to_numpy(float)
    src = df["src_ip"].astype(str).to_numpy()
    dst = df["dst_ip"].astype(str).to_numpy()
    dport = df["dst_port"].to_numpy(np.int64)
    w_short, w_long = config.CONTEXT_WINDOW_SHORT, config.CONTEXT_WINDOW_LONG

    src_win: dict[str, deque] = {}
    src_ports: dict[str, Counter] = {}
    src_dsts: dict[str, Counter] = {}
    dst_win: dict[str, deque] = {}
    pair_win: dict[tuple, deque] = {}
    for i in range(n):
        t, s, d, p = ts[i], src[i], dst[i], dport[i]
        q = src_win.setdefault(s, deque())
        ports = src_ports.setdefault(s, Counter())
        dsts = src_dsts.setdefault(s, Counter())
        while q and q[0][0] < t - w_short:
            _, op, od = q.popleft()
            ports[op] -= 1
            dsts[od] -= 1
            if ports[op] <= 0:
                del ports[op]
            if dsts[od] <= 0:
                del dsts[od]
        out["ctx_src_flows_60s"][i] = len(q)
        out["ctx_src_dports_60s"][i] = len(ports)
        out["ctx_src_dsts_60s"][i] = len(dsts)
        q.append((t, p, d))
        ports[p] += 1
        dsts[d] += 1

        dq = dst_win.setdefault(d, deque())
        while dq and dq[0] < t - w_short:
            dq.popleft()
        out["ctx_dst_flows_60s"][i] = len(dq)
        dq.append(t)

        pq = pair_win.setdefault((s, d, p), deque())
        while pq and pq[0] < t - w_long:
            pq.popleft()
        out["ctx_pair_flows_1h"][i] = len(pq)
        if len(pq) >= 3:
            gaps = np.diff(np.fromiter(pq, float)[-11:])
            mean = gaps.mean()
            out["ctx_pair_iat_cv"][i] = float(gaps.std() / mean) if mean > 0 else 1.0
        pq.append(t)
    for c in CONTEXT_FEATURES:
        df[c] = out[c]
    return df


def normalise(df: pd.DataFrame, compute_context: bool = True) -> pd.DataFrame:
    """Coerce any flow DataFrame onto ALL_COLUMNS with sane types and defaults."""
    df = df.copy()
    defaults = {"src_ip": "0.0.0.0", "dst_ip": "0.0.0.0", "src_port": 0, "dst_port": 0,
                "src_host": "", "dst_host": "", "sensor": "unknown", "label": BENIGN,
                "campaign": "", "split": ""}
    for col, val in defaults.items():
        if col not in df:
            df[col] = val
    if "ts" not in df:
        df["ts"] = np.arange(len(df), dtype=float)
    if "end_ts" not in df:
        df["end_ts"] = df["ts"] + df.get("duration", 0.0)
    for col in FLOW_FEATURES:
        if col not in df:
            df[col] = 0.0
    for col in ("src_ip", "dst_ip", "src_host", "dst_host", "sensor", "label", "campaign", "split"):
        df[col] = df[col].fillna(defaults.get(col, "")).astype(str)
    for col in ("src_port", "dst_port"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int64)
    for col in ["ts", "end_ts"] + FLOW_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    feat = df[FLOW_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df[FLOW_FEATURES] = feat.clip(lower=0.0)
    df["ts"] = df["ts"].fillna(0.0)
    df["end_ts"] = df["end_ts"].fillna(df["ts"])
    if compute_context:
        df = add_context_features(df)
    else:
        for c in CONTEXT_FEATURES:
            if c not in df:
                df[c] = 1.0 if c == "ctx_pair_iat_cv" else 0.0
    return df[ALL_COLUMNS].reset_index(drop=True)


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of schema problems (empty list = valid)."""
    problems = [f"missing column {c}" for c in ALL_COLUMNS if c not in df.columns]
    if not problems:
        bad = [c for c in FEATURES if not np.isfinite(df[c].to_numpy(float)).all()]
        problems += [f"non-finite values in {c}" for c in bad]
    return problems
