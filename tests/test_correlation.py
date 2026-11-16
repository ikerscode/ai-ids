"""Correlation groups a multi-stage campaign into one incident."""
from aiids.correlate import Correlator


def A(i, t, cls, stage, src="", dst="", src_host="", dst_host="", user="", sensor="sensor-dmz", conf=0.9):
    return {"id": i, "ts": t, "end_ts": t + 60, "alert_class": cls, "stage": stage, "src_ip": src, "dst_ip": dst,
            "src_host": src_host, "dst_host": dst_host, "user": user, "sensor": sensor, "confidence": conf,
            "technique_id": "", "severity": "high"}


def test_campaign_one_incident():
    atk, web, db = "203.0.113.9", "10.10.2.10", "10.10.3.10"
    t = 1_000_000
    alerts = [
        A(1, t, "portscan", "recon", atk, web),
        A(2, t + 900, "web_sqli", "initial_access", atk, web),
        A(3, t + 1500, "service_install", "execution_persistence", web, web, dst_host="web01", user="www-data",
          sensor="hostlog-web01"),
        A(4, t + 2400, "smb_lateral", "lateral_movement", web, db, sensor="sensor-srv"),
        A(5, t + 2500, "new_logon_source", "lateral_movement", web, db, dst_host="db01", user="svc_web",
          sensor="hostlog-db01"),
        A(6, t + 3000, "c2_beacon", "command_and_control", web, "198.51.100.23"),
        A(7, t + 4000, "dns_tunnel", "exfil_impact", db, "10.10.3.53", sensor="sensor-srv"),
        # unrelated: a different workstation doing something odd, and a flood of the same web server later
        A(8, t + 1200, "unknown_anomaly", "recon", "10.10.1.15", "34.120.1.1", sensor="sensor-corp", conf=0.6),
        A(9, t + 20000, "syn_flood", "exfil_impact", "1.2.3.4", web),
    ]
    incs = Correlator().correlate(alerts)
    main = max(incs, key=lambda i: i["n_alerts"])
    assert sorted(main["alert_ids"]) == [1, 2, 3, 4, 5, 6, 7]
    assert main["stage_count"] == 6 and main["progression"]
    assert main["severity"] == "critical"
    assert len(main["systems"]) >= 3
    assert len(incs) == 3
    others = [i for i in incs if i is not main]
    assert all(i["score"] < main["score"] for i in others)


def test_external_alert_joins_by_ip_and_hostname():
    t = 2_000_000
    alerts = [A(1, t, "ssh_bruteforce", "initial_access", "203.0.113.50", "10.10.2.10"),
              # honeypot only knows the attacker IP; host log only knows the host name
              A(2, t + 600, "honeypot_login", "initial_access", "203.0.113.50", "10.99.0.2", sensor="honeypot-net"),
              A(3, t + 900, "service_install", "execution_persistence", "", "", dst_host="web01", src_host="web01",
                sensor="hostlog-web01")]
    incs = Correlator().correlate(alerts)
    assert len(incs) == 1 and incs[0]["n_alerts"] == 3


def _pipeline_alert(i, **fields):
    """An alert as the cross-lab pipeline sends it (ExternalAlert JSON), normalised like /api/ingest does."""
    from aiids.ingest import ExternalAlert, to_alert
    out = to_alert(ExternalAlert(**fields))
    out["id"] = i
    return out


def test_pipeline_malware_and_ics_alerts_share_one_incident():
    """Malware on CORP-WS-37 and the blocked Modbus writes from CORP-WS-37 are one IT-to-OT incident."""
    t = 1_794_844_000.0
    alerts = [
        _pipeline_alert(1, source="malware-lab", category="malware_exec", ts=t + 14, src_host="CORP-WS-37",
                        severity="critical", confidence=0.94, technique_id="T1204", stage="execution_persistence"),
        _pipeline_alert(2, source="malware-lab", category="c2_beacon", ts=t + 15, src_host="CORP-WS-37",
                        dst_ip="203.0.113.50", dst_port=8443, severity="critical", confidence=0.94,
                        technique_id="T1071.001", stage="command_and_control"),
        # ICS fast runs are back-dated a minute and name the sim's host, case as the lab prints it
        _pipeline_alert(3, source="ics-sim", category="modbus_blocked", ts=t - 40, src_host="corp-ws-37",
                        dst_host="PLC-01", severity="high", technique_id="T0855", stage="exfil_impact"),
        _pipeline_alert(4, source="ics-sim", category="modbus_blocked", ts=t + 21, src_host="CORP-WS-37",
                        dst_host="PLC-01", severity="high", technique_id="T0855", stage="exfil_impact"),
        # unrelated: another sample that was never placed on a host, and another workstation near the PLC
        _pipeline_alert(5, source="malware-lab", category="malware_exec", ts=t + 14, src_host="sample-56dba89dfbb8",
                        severity="critical", confidence=0.94, stage="execution_persistence"),
        _pipeline_alert(6, source="honeypot-net", category="ssh_bruteforce", ts=t + 30, src_ip="129.98.51.17",
                        dst_host="web01", severity="high"),
        _pipeline_alert(7, source="malware-lab", category="malware_exec", ts=t + 18, src_host="CORP-WS-12",
                        severity="high", confidence=0.7, stage="execution_persistence"),
    ]
    incs = Correlator().correlate(alerts)
    chain = [i for i in incs if 1 in i["alert_ids"]]
    assert len(chain) == 1
    assert sorted(chain[0]["alert_ids"]) == [1, 2, 3, 4]
    assert chain[0]["systems"] == ["ics-sim", "malware-lab"]
    assert {5, 6, 7}.isdisjoint(chain[0]["alert_ids"])
    assert len(incs) == 4
