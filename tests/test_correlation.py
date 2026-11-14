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
