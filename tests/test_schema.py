"""Feature schema and dataset loaders."""
import numpy as np
import pandas as pd

from aiids.datasets import load_csv, map_label
from aiids.schema import ALL_COLUMNS, CONTEXT_FEATURES, FEATURES, port_class, validate


def test_feature_list():
    assert len(FEATURES) == len(set(FEATURES)) >= 30
    assert set(CONTEXT_FEATURES) <= set(FEATURES)
    assert port_class(443) == 0 and port_class(53) == 1 and port_class(60000) == 11


def test_simulated_schema(small_sim):
    flows, events, meta = small_sim
    assert list(flows.columns) == ALL_COLUMNS
    assert validate(flows) == []
    assert set(flows.split) == {"train", "val", "feedback", "test"}
    fams = set(flows.label) - {"benign"}
    assert fams == set(meta["families"])
    assert {"auth_failure", "auth_success", "service_install"} <= set(events.event)
    assert flows.sensor.nunique() >= 4


def test_cic_csv(tmp_path):
    rows = []
    for i in range(40):
        lab = "BENIGN" if i % 4 else ("PortScan" if i % 8 else "Web Attack – Sql Injection")
        rows.append({" Destination Port": 80, " Flow Duration": 1_000_000, " Total Fwd Packets": 3,
                     " Total Backward Packets": 2, "Total Length of Fwd Packets": 300,
                     " Total Length of Bwd Packets": 900, " Fwd Packet Length Mean": 100.0, " Flow IAT Mean": 2000.0,
                     " SYN Flag Count": 1, "Flow Bytes/s": float("inf") if i == 0 else 1200.0, " Label": lab,
                     " Source IP": "192.168.10.5", " Destination IP": "192.168.10.50",
                     " Timestamp": f"3/7/2017 8:{i:02d}"})
    p = tmp_path / "cic.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    flows, meta = load_csv(p)
    assert meta["format"] == "cic-ids2017" and meta["has_time_and_ips"]
    assert validate(flows) == []
    assert set(flows.label) == {"benign", "portscan", "web_sqli"}
    assert np.isclose(flows.duration.iloc[0], 1.0) and np.isclose(flows.iat_mean.iloc[0], 0.002)


def test_unsw_csv(tmp_path):
    df = pd.DataFrame({"id": range(20), "dur": 0.5, "proto": "tcp", "service": "http", "state": "FIN",
                       "spkts": 4, "dpkts": 6, "sbytes": 400, "dbytes": 3000, "smean": 100, "dmean": 500,
                       "sinpkt": 10.0, "dinpkt": 12.0, "sjit": 1.0,
                       "attack_cat": ["Normal"] * 15 + ["Reconnaissance"] * 5, "label": [0] * 15 + [1] * 5})
    p = tmp_path / "UNSW_NB15_training-set.csv"
    df.to_csv(p, index=False)
    flows, meta = load_csv(p)
    assert meta["format"] == "unsw-nb15"
    assert validate(flows) == []
    assert (flows.label == "reconnaissance").sum() == 5
    assert (flows.dst_port == 80).all()


def test_label_map():
    assert map_label("SSH-Patator") == "ssh_bruteforce"
    assert map_label("DoS Hulk") == "dos"
    assert map_label("Web Attack � Sql Injection") == "web_sqli"
