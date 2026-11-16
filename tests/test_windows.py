"""Time-window behaviour features: correctness against a brute-force reference and causality."""
import numpy as np
import pandas as pd

from aiids.windows import WINDOW_FEATURES, compute_window_features, failed_or_empty


def _flows(n=300, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.choice(np.arange(0, 900, 5.0), n))  # many exact timestamp ties
    return pd.DataFrame({
        "ts": ts,
        "src_ip": rng.choice(["10.0.0.1", "10.0.0.2", "10.0.0.3"], n),
        "dst_ip": rng.choice(["10.0.1.1", "10.0.1.2", "8.8.8.8", "1.1.1.1"], n),
        "dst_port": rng.choice([22, 53, 80, 443, 8080], n),
        "fwd_pkts": rng.choice([1, 2, 5], n), "bwd_pkts": rng.choice([0, 1, 4], n),
        "fwd_bytes": rng.choice([0, 60, 600], n), "bwd_bytes": rng.choice([0, 1500], n),
        "rst_cnt": rng.choice([0, 0, 1], n),
    })


def _reference(df: pd.DataFrame) -> pd.DataFrame:
    """Slow, obviously-correct version: loop over flows, look only at earlier rows."""
    fail = failed_or_empty(df)
    rows = []
    for i in range(len(df)):
        t = df.ts.iloc[i]
        past = df.iloc[: i + 1]  # input is already in arrival order
        r = {}
        for w in (10, 60, 300):
            win = past[(past.ts > t - w) & (past.src_ip == df.src_ip.iloc[i])]
            r[f"win_src_flows_{w}s"] = len(win)
            r[f"win_src_dports_{w}s"] = win.dst_port.nunique()
            r[f"win_src_dsts_{w}s"] = win.dst_ip.nunique()
            r[f"win_src_fail_{w}s"] = fail[win.index].mean()
        s60 = past[(past.ts > t - 60) & (past.src_ip == df.src_ip.iloc[i])]
        sig = ["fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes"]
        r["win_src_samesize_60s"] = (s60[sig] == df[sig].iloc[i]).all(axis=1).mean()
        s300 = past[(past.ts > t - 300) & (past.src_ip == df.src_ip.iloc[i])]
        r["win_src_bytes_ratio_300s"] = np.log10((s300.bwd_bytes.sum() + 1) / (s300.fwd_bytes.sum() + 1))
        pair = past[(past.src_ip == df.src_ip.iloc[i]) & (past.dst_ip == df.dst_ip.iloc[i])]
        r["win_pair_flows_60s"] = int((pair.ts > t - 60).sum())
        r["win_pair_flows_300s"] = int((pair.ts > t - 300).sum())
        r["win_pair_dports_300s"] = pair[pair.ts > t - 300].dst_port.nunique()
        d = past[(past.ts > t - 60) & (past.dst_ip == df.dst_ip.iloc[i])]
        r["win_dst_flows_60s"] = len(d)
        r["win_dst_srcs_60s"] = d.src_ip.nunique()
        rows.append(r)
    return pd.DataFrame(rows)


def test_matches_bruteforce():
    df = _flows()
    got = compute_window_features(df)
    ref = _reference(df)
    for col in ref.columns:
        assert np.allclose(got[col].to_numpy(float), ref[col].to_numpy(float)), col


def test_future_flows_do_not_change_past_features():
    df = _flows(400, seed=1)
    cut = 250
    early = compute_window_features(df.iloc[:cut].copy())
    full = compute_window_features(df)
    assert np.allclose(early.to_numpy(float), full.iloc[:cut].to_numpy(float))
    # A burst of future flows from the same sources at the last early timestamp (a tie)
    # must not leak back either: ties are resolved by arrival order.
    burst = df.iloc[[cut - 1] * 50].copy()
    burst["dst_port"] = np.arange(50) + 1000
    more = compute_window_features(pd.concat([df.iloc[:cut], burst], ignore_index=True))
    assert np.allclose(early.to_numpy(float), more.iloc[:cut].to_numpy(float))


def test_tiny_scan_sequence():
    # One source probes 20 ports of one host, one probe per second, then a benign host makes 3 web flows.
    scan = pd.DataFrame({"ts": np.arange(20.0), "src_ip": "10.0.0.9", "dst_ip": "10.0.0.50",
                         "dst_port": np.arange(20) + 20, "fwd_pkts": 1, "bwd_pkts": 1, "fwd_bytes": 0,
                         "bwd_bytes": 0, "rst_cnt": 1})
    web = pd.DataFrame({"ts": [5.0, 6.0, 30.0], "src_ip": "10.0.0.7", "dst_ip": "93.184.216.34",
                        "dst_port": 443, "fwd_pkts": 8, "bwd_pkts": 12, "fwd_bytes": 900, "bwd_bytes": 9000,
                        "rst_cnt": 0})
    df = pd.concat([scan, web]).sort_values("ts", kind="stable").reset_index(drop=True)
    f = compute_window_features(df)
    last_probe = df.index[(df.src_ip == "10.0.0.9")][-1]
    assert f.loc[last_probe, "win_src_dports_10s"] == 10  # probes at t=10..19 (window is (t-10, t])
    assert f.loc[last_probe, "win_src_dports_60s"] == 20
    assert f.loc[last_probe, "win_src_fail_60s"] == 1.0
    assert f.loc[last_probe, "win_src_samesize_60s"] == 1.0
    assert f.loc[last_probe, "win_dst_srcs_60s"] == 1
    first_probe = df.index[(df.src_ip == "10.0.0.9")][0]
    assert f.loc[first_probe, "win_src_flows_300s"] == 1  # nothing before it
    web_rows = df.index[df.src_ip == "10.0.0.7"]
    assert list(f.loc[web_rows, "win_pair_flows_60s"]) == [1, 2, 3]
    assert (f.loc[web_rows, "win_src_fail_300s"] == 0).all()
    assert (f.loc[web_rows, "win_src_bytes_ratio_300s"] > 0.9).all()  # downloads ~10x more than it sends
    assert list(f.columns) == WINDOW_FEATURES


def test_beacon_regularity():
    # Every 30 s exactly to one peer -> CV 0; jittered gaps -> clearly higher.
    reg = pd.DataFrame({"ts": np.arange(0, 300, 30.0), "src_ip": "10.0.0.5", "dst_ip": "198.51.100.7",
                        "dst_port": 443, "fwd_pkts": 3, "bwd_pkts": 3, "fwd_bytes": 300, "bwd_bytes": 300,
                        "rst_cnt": 0})
    jit = reg.copy()
    jit["src_ip"] = "10.0.0.6"
    jit["ts"] = np.cumsum([0, 3, 50, 7, 61, 2, 40, 5, 70, 9]).astype(float)
    f = compute_window_features(pd.concat([reg, jit]).sort_values("ts", kind="stable").reset_index(drop=True))
    df = pd.concat([reg, jit]).sort_values("ts", kind="stable").reset_index(drop=True)
    last_reg = df.index[df.src_ip == "10.0.0.5"][-1]
    last_jit = df.index[df.src_ip == "10.0.0.6"][-1]
    assert f.loc[last_reg, "win_pair_gap_cv_300s"] < 0.01
    assert f.loc[last_jit, "win_pair_gap_cv_300s"] > 0.5
    assert f.loc[df.index[df.src_ip == "10.0.0.5"][0], "win_pair_gap_cv_300s"] == 1.0  # too few gaps yet
