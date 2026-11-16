"""Behavioural time-window features per source host, per (src, dst) pair and per destination.

A single flow record rarely shows an attack: one port-scan probe or one failed
login looks like any short benign flow. What gives them away is the behaviour
of the host around that flow: how many ports and hosts it touched in the last
few seconds, how many of its recent flows failed, how many were the exact same
size, how regular the timing to one peer is. These features describe that.

Causality
---------
Flows are put in arrival order: by start time, ties kept in the order they were
recorded. Every feature of flow i is computed from flow i itself and the flows
that arrived before it with a start time in (t_i - W, t_i]. Later flows never
change it (tests check this by appending future flows), so scoring a test day
cannot peek at what happens next.

Everything is vectorised with numpy: per group the flows are sorted by
(group, time) and window bounds come from one searchsorted call. Distinct counts
use the identity "a key is counted once per window, by its first occurrence in
the window", turned into one +1 range per flow and a cumulative sum, so memory
stays a handful of int64 arrays the length of the input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = (10.0, 60.0, 300.0)


def _wname(w: float) -> str:
    return f"{int(w)}s"


WINDOW_FEATURES = (
    [f"win_src_{k}_{_wname(w)}" for w in WINDOWS for k in ("flows", "dports", "dsts", "fail")]
    + ["win_pair_flows_60s", "win_pair_flows_300s", "win_pair_dports_300s",
       "win_src_samesize_60s", "win_src_bytes_ratio_300s",
       "win_dst_flows_60s", "win_dst_srcs_60s", "win_pair_gap_cv_300s"]
)
# Ratios and coefficients of variation: already on a small scale, not log-transformed.
WINDOW_NON_LOG = {f for f in WINDOW_FEATURES if f.endswith(("_fail_10s", "_fail_60s", "_fail_300s"))} | {
    "win_src_samesize_60s", "win_src_bytes_ratio_300s", "win_pair_gap_cv_300s"}

WINDOW_LABELS = {
    **{f"win_src_flows_{_wname(w)}": f"flows from this source in {_wname(w)}" for w in WINDOWS},
    **{f"win_src_dports_{_wname(w)}": f"distinct ports this source hit in {_wname(w)}" for w in WINDOWS},
    **{f"win_src_dsts_{_wname(w)}": f"distinct hosts this source hit in {_wname(w)}" for w in WINDOWS},
    **{f"win_src_fail_{_wname(w)}": f"share of failed/empty flows from this source in {_wname(w)}" for w in WINDOWS},
    "win_pair_flows_60s": "flows between this pair in 60s", "win_pair_flows_300s": "flows between this pair in 300s",
    "win_pair_dports_300s": "distinct ports between this pair in 300s",
    "win_src_samesize_60s": "share of this source's recent flows with exactly this size",
    "win_src_bytes_ratio_300s": "log10 bytes received/sent by this source in 300s",
    "win_dst_flows_60s": "flows to this destination in 60s", "win_dst_srcs_60s": "distinct sources to this destination in 60s",
    "win_pair_gap_cv_300s": "timing regularity to this peer (gap CV, low = beacon-like)",
}
DEFAULT_GAP_CV = 1.0  # "no evidence of regularity" when there are too few gaps


def failed_or_empty(df: pd.DataFrame) -> np.ndarray:
    """Flows that were reset, got no reply, or carried no payload at all (probes, refused logins)."""
    rst = df["rst_cnt"].to_numpy(float) > 0
    no_reply = df["bwd_pkts"].to_numpy(float) == 0
    empty = (df["fwd_bytes"].to_numpy(float) + df["bwd_bytes"].to_numpy(float)) == 0
    return rst | no_reply | empty


class _Grouped:
    """Flows sorted by (group, arrival order) with combined search keys."""

    def __init__(self, codes: np.ndarray, t: np.ndarray, span: float):
        n = len(t)
        self.order = np.lexsort((np.arange(n), t, codes))  # group, then time, then recording order
        self.codes = codes[self.order]
        self.t = t[self.order]
        self.span = span
        self.key = self.codes.astype(np.float64) * span + self.t
        self.pos = np.arange(n)

    def window_start(self, w: float) -> np.ndarray:
        """First sorted position inside (t - w, t] of the same group (ties at exactly t - w excluded)."""
        return np.searchsorted(self.key, self.key - w, side="right")

    def unsort(self, v: np.ndarray) -> np.ndarray:
        out = np.empty_like(v)
        out[self.order] = v
        return out

    def count(self, w: float) -> np.ndarray:
        """Flows in the window including the current one, in sorted order."""
        return self.pos - self.window_start(w) + 1

    def wsum(self, v_sorted: np.ndarray, w: float) -> np.ndarray:
        cs = np.concatenate(([0.0], np.cumsum(v_sorted, dtype=np.float64)))
        return cs[self.pos + 1] - cs[self.window_start(w)]

    def distinct(self, sub: np.ndarray, w: float) -> np.ndarray:
        """Distinct values of `sub` (int codes, sorted order) in the window, current flow included.

        Flow j is the first occurrence of its value in the window of flow i exactly when
        t_i - w < t_j (j is in the window) and its previous occurrence p satisfies
        t_p <= t_i - w (p is not). For fixed j that holds for a contiguous run of later
        flows i, so each j adds +1 over one index range: a difference array does the rest.
        """
        n = len(sub)
        if n == 0:
            return np.zeros(0)
        # previous occurrence of the same value within the same group (sorted positions)
        o = np.lexsort((self.pos, sub, self.codes))
        same = np.zeros(n, dtype=bool)
        same[1:] = (self.codes[o][1:] == self.codes[o][:-1]) & (sub[o][1:] == sub[o][:-1])
        prev = np.full(n, -1, dtype=np.int64)
        prev[o[1:][same[1:]]] = o[:-1][same[1:]]
        # j counts for i in [a_j, b_j]: i >= j, t_i < t_j + w, t_i >= t_prev + w
        b = np.searchsorted(self.key, self.key + w, side="left") - 1
        has_prev = prev >= 0
        prev_key = np.where(has_prev, self.codes * self.span + self.t[np.maximum(prev, 0)], 0.0)
        a = np.where(has_prev, np.searchsorted(self.key, prev_key + w, side="left"), self.pos)
        a = np.maximum(a, self.pos)
        ok = a <= b
        diff = np.bincount(a[ok], minlength=n + 1) - np.bincount(b[ok] + 1, minlength=n + 1)
        return np.cumsum(diff)[:n].astype(float)

    def prev_gap(self) -> np.ndarray:
        """Seconds since the previous flow of the same group (nan for the first one)."""
        g = np.full(len(self.t), np.nan)
        same = np.zeros(len(self.t), dtype=bool)
        same[1:] = self.codes[1:] == self.codes[:-1]
        g[1:] = np.where(same[1:], self.t[1:] - self.t[:-1], np.nan)
        return g


def _codes(*cols) -> np.ndarray:
    """Integer code per distinct combination of the given columns."""
    if len(cols) == 1:
        return pd.factorize(pd.Series(cols[0]), sort=False)[0].astype(np.int64)
    parts = [pd.factorize(pd.Series(c), sort=False)[0].astype(np.int64) for c in cols]
    code = parts[0]
    for p in parts[1:]:
        code = pd.factorize(code * (int(p.max()) + 1) + p, sort=False)[0].astype(np.int64)
    return code


def compute_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame (same index as df) with WINDOW_FEATURES.

    Needs ts, src_ip, dst_ip, dst_port, fwd/bwd packets and bytes, rst_cnt.
    Input order is the recording order used to break timestamp ties.
    """
    n = len(df)
    out = pd.DataFrame(index=df.index, columns=WINDOW_FEATURES, dtype=float)
    if n == 0:
        return out
    t = df["ts"].to_numpy(float)
    t = t - np.nanmin(t)
    span = float(np.nanmax(t)) + 2 * max(WINDOWS) + 10.0
    src = _codes(df["src_ip"].astype(str).to_numpy())
    dst = _codes(df["dst_ip"].astype(str).to_numpy())
    dport = df["dst_port"].to_numpy(np.int64)
    fail = failed_or_empty(df).astype(float)
    fwd_b = df["fwd_bytes"].to_numpy(float)
    bwd_b = df["bwd_bytes"].to_numpy(float)
    res = {}

    g = _Grouped(src, t, span)
    dport_s, dst_s = dport[g.order], dst[g.order]
    for w in WINDOWS:
        c = g.count(w)
        res[f"win_src_flows_{_wname(w)}"] = g.unsort(c.astype(float))
        res[f"win_src_dports_{_wname(w)}"] = g.unsort(g.distinct(dport_s, w))
        res[f"win_src_dsts_{_wname(w)}"] = g.unsort(g.distinct(dst_s, w))
        res[f"win_src_fail_{_wname(w)}"] = g.unsort(g.wsum(fail[g.order], w) / c)
    fwd = g.wsum(fwd_b[g.order], 300.0)
    bwd = g.wsum(bwd_b[g.order], 300.0)
    res["win_src_bytes_ratio_300s"] = g.unsort(np.log10((bwd + 1.0) / (fwd + 1.0)))
    c60 = g.count(60.0)

    # Same size signature as this flow among the source's recent flows (scripted logins, floods).
    sig = _codes(src, df["fwd_pkts"].to_numpy(float), df["bwd_pkts"].to_numpy(float), fwd_b, bwd_b)
    gs = _Grouped(sig, t, span)
    same = gs.unsort(gs.count(60.0).astype(float))
    res["win_src_samesize_60s"] = same / g.unsort(c60.astype(float))

    pair = _codes(src, dst)
    gp = _Grouped(pair, t, span)
    res["win_pair_flows_60s"] = gp.unsort(gp.count(60.0).astype(float))
    res["win_pair_flows_300s"] = gp.unsort(gp.count(300.0).astype(float))
    res["win_pair_dports_300s"] = gp.unsort(gp.distinct(dport[gp.order], 300.0))
    # Beacon score: coefficient of variation of the gaps between this pair's flows in 300 s.
    gap = gp.prev_gap()
    has = ~np.isnan(gap)
    gz = np.where(has, gap, 0.0)
    k = gp.wsum(has.astype(float), 300.0)
    s1 = gp.wsum(gz, 300.0)
    s2 = gp.wsum(gz * gz, 300.0)
    mean = np.divide(s1, k, out=np.zeros_like(s1), where=k > 0)
    var = np.divide(s2, k, out=np.zeros_like(s2), where=k > 0) - mean * mean
    cv = np.sqrt(np.clip(var, 0, None)) / np.where(mean > 0, mean, 1.0)
    cv = np.where((k >= 3) & (mean > 0), np.minimum(cv, 10.0), DEFAULT_GAP_CV)
    res["win_pair_gap_cv_300s"] = gp.unsort(cv)

    gd = _Grouped(dst, t, span)
    res["win_dst_flows_60s"] = gd.unsort(gd.count(60.0).astype(float))
    res["win_dst_srcs_60s"] = gd.unsort(gd.distinct(src[gd.order], 60.0))
    for f in WINDOW_FEATURES:
        out[f] = res[f]
    return out


def add_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """df with WINDOW_FEATURES columns added (row order unchanged)."""
    df = df.copy()
    feats = compute_window_features(df)
    for f in WINDOW_FEATURES:
        df[f] = feats[f].to_numpy(float)
    return df


def has_window_features(df: pd.DataFrame) -> bool:
    return all(f in df.columns for f in WINDOW_FEATURES)
