"""CIC-IDS2017 TrafficLabelling loader on a tiny file written in the original byte format."""
import numpy as np
import pandas as pd

from aiids.datasets import (clean_label, load_cic_tl, load_cic_tl_day, parse_cic_timestamp, read_cic_tl_file,
                            time_split)
from aiids.schema import validate
from aiids.windows import WINDOW_FEATURES

HEADER = ("Flow ID, Source IP, Source Port, Destination IP, Destination Port, Protocol, Timestamp, Flow Duration,"
          " Total Fwd Packets, Total Backward Packets,Total Length of Fwd Packets, Total Length of Bwd Packets,"
          " Fwd Header Length, Flow Bytes/s, Flow Packets/s, RST Flag Count, Fwd Header Length, Label")


def _row(src, sport, dst, dport, stamp, label, fwd=3, bwd=2, fb=120, bb=400, rate="1200.5"):
    fid = f"{dst}-{src}-{dport}-{sport}-6"
    return (f"{fid},{src},{sport},{dst},{dport},6,{stamp},1500000,{fwd},{bwd},{fb},{bb},72,{rate},4.0,0,72,{label}")


def _write(path, rows):
    text = HEADER + "\n" + "\n".join(rows) + "\n" + ",,,,,,,,,,,,,,,,,\n" * 3  # trailing empty rows
    path.write_bytes(text.encode("latin-1").replace(b"WEBDASH", b"\x96"))


def test_latin1_label_and_quirks(tmp_path):
    p = tmp_path / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    rows = [_row("192.168.10.5", 50000 + i, "192.168.10.3", 53, "6/7/2017 8:59", "BENIGN") for i in range(5)]
    rows += [_row("172.16.0.1", 40000 + i, "192.168.10.50", 80, "6/7/2017 9:20", "Web Attack WEBDASH Brute Force",
                  rate="Infinity" if i == 0 else "NaN" if i == 1 else "10.0") for i in range(4)]
    rows += [_row("172.16.0.1", 41000, "192.168.10.50", 80, "6/7/2017 10:41", "Web Attack WEBDASH Sql Injection")]
    _write(p, rows)
    raw = read_cic_tl_file(p)
    assert len(raw) == 10  # empty rows dropped
    assert "Web Attack - Brute Force" in set(raw["Label"])
    assert "Fwd Header Length" not in raw.columns and "Fwd Header Length.1" not in raw.columns
    flows = load_cic_tl_day([p], log=lambda *a: None)
    assert validate(flows) == []
    assert set(flows["label"]) == {"benign", "web_attack", "web_sqli"}
    assert np.isfinite(flows["bytes_per_s"]).all()
    assert all(f in flows.columns for f in WINDOW_FEATURES)
    assert (np.diff(flows["ts"].to_numpy()) >= 0).all()
    attack = flows[flows.label == "web_attack"]
    assert attack["win_src_flows_60s"].max() == 4  # all four brute-force flows in the same minute
    assert flows.loc[flows.label == "web_sqli", "campaign"].iloc[0] == "thursday:Web Attack - Sql Injection"


def test_timestamp_parsing():
    s = pd.Series(["03/07/2017 08:55:58", "03/07/2017 01:05:10", "7/7/2017 3:30", "7/7/2017 3:30", "7/7/2017 12:59",
                   "garbage"])
    t = parse_cic_timestamp(s)
    hours = (t[:5] % 86400) / 3600
    assert np.isclose(hours[0], 8 + 55 / 60)           # seconds dropped: minute resolution everywhere
    assert np.isclose(hours[1], 13 + 5 / 60)           # 1:05 on a 12-hour clock is 13:05
    assert np.isclose(hours[2], 15.5) and np.isclose(t[3] - t[2], 30.0)  # two flows spread over the minute
    assert np.isclose(hours[4], 12 + 59 / 60)          # 12:59 stays noon
    assert pd.Timestamp(t[0], unit="s").day == 3 and pd.Timestamp(t[0], unit="s").month == 7  # day first
    assert np.isnan(t[5])
    assert clean_label("Web Attack � XSS") == "Web Attack - XSS"


def test_time_split_and_weights(tmp_path):
    mon = [_row("192.168.10.5", 50000 + i, "192.168.10.3", 53, f"03/07/2017 {h}:{m:02d}:00", "BENIGN")
           for i, (h, m) in enumerate([(9, 0), (10, 0), (2, 40), (3, 10), (4, 0)])]
    tue = [_row("192.168.10.5", 51000 + i, "192.168.10.3", 53, f"4/7/2017 {h}:{m:02d}", "BENIGN")
           for i, (h, m) in enumerate([(8, 55), (9, 5), (11, 0), (2, 0)])]
    tue += [_row("172.16.0.1", 52000 + i, "192.168.10.50", 22, f"4/7/2017 3:{i:02d}", "SSH-Patator") for i in range(10)]
    _write(tmp_path / "Monday-WorkingHours.pcap_ISCX.csv", mon)
    _write(tmp_path / "Tuesday-WorkingHours.pcap_ISCX.csv", tue)
    flows, meta = load_cic_tl(tmp_path, log=lambda *a: None)
    assert meta["split_kind"] == "time" and meta["window_features"]
    day = (flows["ts"] // 86400).to_numpy()
    mon_f, tue_f = flows[day == day.min()], flows[day == day.max()]
    assert list(mon_f["split"]) == ["train", "train", "val", "val", "feedback"]  # 09, 10, 14:40, 15:10, 16:00
    tb = tue_f[tue_f.label == "benign"]
    assert list(tb["split"]) == ["train", "train", "test", "test"]
    ssh = tue_f[tue_f.label == "ssh_bruteforce"]
    assert list(ssh["split"]) == ["train"] * 3 + ["test"] * 7  # first 30% in time may train the classifier
    assert (flows["weight"] == 1.0).all()  # nothing was capped in a tiny file
    # every training/validation flow is earlier than every test flow of the same day
    for d in (mon_f, tue_f):
        tr = d[d.split.isin(["train", "val"]) & (d.label == "benign")]
        te = d[d.split == "test"]
        if len(tr) and len(te):
            assert tr["ts"].max() < te["ts"].min()
    assert len(time_split(tue_f.reset_index(drop=True), "Tuesday")) == len(tue_f)


def test_label_suspects_tagged_not_relabelled(tmp_path):
    rows = [_row("192.168.10.5", 50000 + i, "192.168.10.3", 53, "6/7/2017 2:10", "BENIGN") for i in range(3)]
    rows += [_row("192.168.10.8", 50100 + i, "205.174.165.73", 444, f"6/7/2017 2:{20 + i}", "Infiltration")
             for i in range(3)]
    # the infiltrated host scans internal hosts (labelled BENIGN in the original files)
    rows += [_row("192.168.10.8", 50200 + i, f"192.168.10.{20 + i}", 445, "6/7/2017 2:21", "BENIGN") for i in range(4)]
    # reply-direction flow of the same attack pair, also labelled BENIGN
    rows += [_row("205.174.165.73", 444, "192.168.10.8", 50101, "6/7/2017 2:21", "BENIGN")]
    p = tmp_path / "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
    _write(p, rows)
    flows = load_cic_tl_day([p], log=lambda *a: None)
    notes = flows.groupby("label_note").size().to_dict()
    assert notes == {"": 6, "post_infiltration": 4, "attack_pair": 1}
    assert (flows.loc[flows.label_note != "", "label"] == "benign").all()  # labels untouched
    assert (flows.loc[flows.src_ip == "192.168.10.5", "label_note"] == "").all()
