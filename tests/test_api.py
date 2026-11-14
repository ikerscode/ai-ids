"""Ingest API schema validation and correlation through the HTTP layer."""
import importlib

from fastapi.testclient import TestClient


def client(tmp_data):
    import aiids.api as api
    importlib.reload(api)
    return TestClient(api.app)


def test_health(tmp_data):
    c = client(tmp_data)
    assert c.get("/api/health").json() == {"status": "ok", "name": "ai-ids", "version": "0.1.0"}
    assert c.get("/").status_code == 200


def test_ingest_validation(tmp_data):
    c = client(tmp_data)
    schema = c.get("/api/ingest/schema").json()
    assert "properties" in schema["json_schema"]
    ok = c.post("/api/ingest", json=schema["example"])
    assert ok.status_code == 200 and ok.json()["accepted"] == 1
    bad = [
        {"source": "honeypot-net", "category": "x"},                                 # no entity
        {"source": "Bad Source!", "category": "x", "src_ip": "1.2.3.4"},             # bad source name
        {"source": "ics-sim", "category": "x", "src_ip": "999.1.1.1"},               # bad IP
        {"source": "ics-sim", "category": "x", "src_ip": "1.2.3.4", "evil": 1},      # unknown field
        {"source": "ics-sim", "category": "x", "src_ip": "1.2.3.4", "severity": "huge"},
        {"source": "ics-sim", "category": "x", "src_ip": "1.2.3.4", "technique_id": "1110"},
    ]
    for b in bad:
        r = c.post("/api/ingest", json=b)
        assert r.status_code == 422, b
    assert c.post("/api/ingest", content=b"not json", headers={"content-type": "application/json"}).status_code == 400
    assert c.post("/api/ingest", json={"alerts": []}).status_code == 400


def test_ingest_batch_correlates(tmp_data):
    c = client(tmp_data)
    base = {"source": "honeypot-net", "src_ip": "203.0.113.77", "ts": 1_800_000_000}
    batch = {"alerts": [
        {**base, "category": "port_scan", "dst_ip": "10.10.2.10"},
        {**base, "ts": 1_800_000_300, "category": "ssh_bruteforce", "dst_ip": "10.10.2.10", "dst_port": 22},
        {"source": "malware-lab", "ts": 1_800_000_900, "category": "malware_exec", "src_host": "web01",
         "dst_host": "web01", "technique_id": "T1059", "severity": "critical"},
    ]}
    r = c.post("/api/ingest", json=batch).json()
    assert r["accepted"] == 3
    assert len(r["incidents"]) == 1 and r["incidents"][0]["n_alerts"] == 3
    inc = c.get(f"/api/incidents/{r['incidents'][0]['incident']}").json()
    assert {"recon", "initial_access", "execution_persistence"} <= set(inc["stages"])
