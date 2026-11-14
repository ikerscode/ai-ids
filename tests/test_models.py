"""Models train on a small set and the decision logic behaves."""
import numpy as np

from aiids.engine import STAGES, UNKNOWN, Engine
from aiids.evaluate import analyst_oracle, flow_metrics, with_supervised
from aiids.explain import Explainer


def test_bundle_trains(small_bundle):
    b = small_bundle
    assert b.supervised is not None and "benign" in b.supervised.classes_
    th = b.thresholds
    assert th["tau_ens"] > th["tau_ens_naive"] > 0
    assert 0.5 <= th["tau_sup"] <= 1


def test_detection_and_stages(small_sim, small_bundle):
    flows = small_sim[0]
    test = flows[flows.split == "test"]
    res = Engine(small_bundle, [], "host_adaptive").detect_stream(test)
    m = flow_metrics(test, res)
    assert m["recall"] > 0.8 and m["fpr"] < 0.05
    naive = flow_metrics(test, Engine(small_bundle, [], "naive").detect_stream(test))
    assert naive["fp"] >= m["fp"]
    assert set(res.alert_class[res.alert]) - {UNKNOWN} <= set(small_bundle.supervised.classes_)


def test_open_set_unknown(small_sim, small_bundle):
    flows = small_sim[0]
    b = with_supervised(small_bundle, flows[flows.split == "train"], flows[flows.split == "val"], ("syn_flood",))
    test = flows[flows.split == "test"]
    res = Engine(b, [], "agreement").detect(test)
    flood = (test.label == "syn_flood").to_numpy()
    assert "syn_flood" not in b.supervised.classes_
    assert res.alert[flood].mean() > 0.8  # never-seen family still flagged by the anomaly layer


def test_suppression_rule(small_sim, small_bundle):
    flows = small_sim[0]
    fb = flows[flows.split == "feedback"]
    res = Engine(small_bundle, [], "host_adaptive").detect(fb)
    rules, extra = analyst_oracle(fb, res, min_count=1)
    if not rules:
        return
    res2 = Engine(small_bundle, rules, "feedback").detect(fb)
    assert res2.suppressed.sum() > 0 and res2.alert.sum() < res.alert.sum()


def test_no_rules_for_unknown_source(small_sim, small_bundle):
    # Public CSVs without IPs map every flow to 0.0.0.0; a rule on that source would mute a whole class.
    flows = small_sim[0]
    fb = flows[flows.split == "feedback"].assign(src_ip="0.0.0.0")
    res = Engine(small_bundle, [], "host_adaptive").detect(fb)
    rules, _ = analyst_oracle(fb, res, min_count=1)
    assert rules == []


def test_explanations(small_sim, small_bundle):
    flows = small_sim[0]
    test = flows[flows.split == "test"]
    res = Engine(small_bundle, [], "calibrated").detect(test)
    idx = np.flatnonzero(res.alert.to_numpy())[:12]
    out = Explainer(small_bundle).explain(test.iloc[idx], res.iloc[idx])
    assert len(out) == len(idx)
    for e in out:
        assert 1 <= len(e["contributions"]) <= 6
        w = [c["weight"] for c in e["contributions"]]
        assert w == sorted(w, reverse=True) and 0 < sum(w) <= 1.001
        assert all(isinstance(c["text"], str) and c["text"] for c in e["contributions"])
        assert e["attack"]["stage"] and "rationale" in e["attack"]
        assert e["summary"]
    # a port scan should be explained by fan-out or flags, not by an irrelevant feature
    scan = [e for e, i in zip(out, idx) if test.iloc[i].label == "portscan"]
    for e in scan:
        assert e["attack"]["technique_id"] == "T1046"
