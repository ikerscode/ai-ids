"""Terminal HUD spec for ai-ids: views and actions mirroring the web HUD.

Drawn by aiids/termhud.py (`python -m aiids tui`). Every endpoint here is a route in aiids/api.py.
"""

SPEC = {
    "title": "AI-IDS",
    "views": [
        {"name": "Alerts", "endpoint": "/api/alerts", "rows": "",
         "columns": [["id", 6], ["severity", 9], ["alert_class", 18], ["src_ip", 16], ["dst_ip", 16],
                     ["count", 6], ["status", 10]],
         "detail": "/api/alerts/{id}"},
        {"name": "Incidents", "endpoint": "/api/incidents", "rows": "",
         "columns": [["id", 10], ["severity", 9], ["score", 7], ["title", 44], ["n_alerts", 8]],
         "detail": "/api/incidents/{id}"},
        {"name": "Models", "endpoint": "/api/models"},
        {"name": "Benchmark", "endpoint": "/api/benchmarks", "rows": "lofo",
         "columns": [["split", 7], ["windows", 8], ["held_out", 15], ["test_flows", 10], ["recall_when_trained", 10],
                     ["recall_held_out", 10], ["via_anomaly_layer", 10], ["benign_fpr", 10]]},
        {"name": "Campaigns", "endpoint": "/api/benchmarks", "rows": "campaigns",
         "columns": [["windows", 8], ["campaign", 36], ["flows_alerted", 9], ["alerts", 7], ["incidents", 9],
                     ["best_severity", 9], ["best_stages", 40]]},
        {"name": "Status", "endpoint": "/api/status"},
    ],
    "actions": [
        {"key": "s", "label": "start stream", "method": "POST", "endpoint": "/api/stream/start",
         "body": {"dataset": "", "split": "test", "speed": 600.0}},
        {"key": "x", "label": "stop stream", "method": "POST", "endpoint": "/api/stream/stop"},
        {"key": "g", "label": "generate dataset", "method": "POST", "endpoint": "/api/simulate",
         "body": {"name": "sim", "days": 4, "seed": 7, "scale": 1.0}, "confirm": True},
        {"key": "d", "label": "batch detect", "method": "POST", "endpoint": "/api/detect",
         "body": {"dataset": "", "split": "test"}},
        {"key": "t", "label": "train", "method": "POST", "endpoint": "/api/train",
         "body": {"dataset": ""}, "confirm": True},
        {"key": "e", "label": "evaluate", "method": "POST", "endpoint": "/api/evaluate",
         "body": {"dataset": "", "lofo": True}, "confirm": True},
    ],
    "refresh": 2,
}
