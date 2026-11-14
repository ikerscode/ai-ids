"""Host/auth log detector: the second data source for cross-system correlation.

Baselines learned from benign training events:
  * which (user, source address) pairs log on normally
  * which services get installed, by whom, and when (patch window)
  * how many failed logons per (host, source) in 10 minutes is normal

Rules then flag bursts of failures, a success right after a burst, logons from a
source never seen for that account, and service installs outside the norm.
"""
from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd

from .schema import BENIGN
from .simulator import BY_NAME

WINDOW = 600.0


class HostDetector:
    def fit(self, events: pd.DataFrame) -> "HostDetector":
        ev = events[events["label"] == BENIGN] if "label" in events else events
        ok = ev[ev["event"] == "auth_success"]
        self.known_pairs = set(zip(ok["user"], ok["src_ip"]))
        inst = ev[ev["event"] == "service_install"]
        self.known_services = set(inst["detail"])
        self.patch_hours = set(((inst["ts"] // 3600) % 24).astype(int)) or {2, 3, 4}
        worst = 0
        fails = ev[ev["event"] == "auth_failure"].sort_values("ts")
        for _, grp in fails.groupby(["host", "src_ip"]):
            t = grp["ts"].to_numpy()
            for i in range(len(t)):
                worst = max(worst, int(((t >= t[i] - WINDOW) & (t <= t[i])).sum()))
        self.fail_threshold = max(5, 2 * worst)
        self.reset()
        return self

    def reset(self) -> None:
        self.fail_q: dict[tuple, deque] = defaultdict(deque)

    def process(self, events: pd.DataFrame) -> list[dict]:
        """Return hits for suspicious events (time-ordered processing, stateful)."""
        hits = []
        for e in events.sort_values("ts").to_dict("records"):
            host_ip = BY_NAME[e["host"]].ip if e["host"] in BY_NAME else ""
            key = (e["host"], e["src_ip"])
            q = self.fail_q[key]
            while q and q[0] < e["ts"] - 3 * WINDOW:
                q.popleft()
            recent = sum(1 for t in q if t >= e["ts"] - WINDOW)
            hit = None
            if e["event"] == "auth_failure":
                q.append(e["ts"])
                recent += 1
                if recent >= self.fail_threshold:
                    conf = min(0.99, 0.6 + 0.02 * recent)
                    hit = ("auth_bruteforce", conf,
                           f"{recent} failed logons on {e['host']} from {e['src_ip']} in 10 min "
                           f"(baseline allows {self.fail_threshold - 1})")
            elif e["event"] == "auth_success":
                if len(q) >= self.fail_threshold:
                    hit = ("auth_after_failures", 0.97,
                           f"successful logon as '{e['user']}' on {e['host']} after {len(q)} failures from {e['src_ip']}")
                elif (e["user"], e["src_ip"]) not in self.known_pairs and e["src_ip"] != host_ip:
                    hit = ("new_logon_source", 0.75,
                           f"'{e['user']}' logged on to {e['host']} from {e['src_ip']}, a source never seen for this account")
            elif e["event"] == "service_install":
                hour = int((e["ts"] // 3600) % 24)
                odd_time = hour not in self.patch_hours
                odd_user = e["user"] != "SYSTEM"
                new = e["detail"] not in self.known_services
                if new and (odd_time or odd_user):
                    why = [w for w, c in (("outside the patch window", odd_time), (f"by '{e['user']}'", odd_user)) if c]
                    hit = ("service_install", 0.85 if odd_time and odd_user else 0.65,
                           f"new service ({e['detail']}) installed on {e['host']} " + " and ".join(why))
            if hit:
                cls, conf, text = hit
                src_host = next((h for h, v in BY_NAME.items() if v.ip == e["src_ip"]), "")
                hits.append({
                    "ts": e["ts"], "alert_class": cls, "src_ip": e["src_ip"], "dst_ip": host_ip, "src_host": src_host,
                    "dst_host": e["host"], "user": e["user"], "dst_port": 0, "sensor": e["sensor"],
                    "label": e.get("label", ""), "campaign": e.get("campaign", ""), "score": conf, "confidence": conf,
                    "row": None, "flow": e,
                    "explanation": {"summary": text, "contributions": [
                        {"feature": e["event"], "label": e["event"], "weight": 1.0, "value": e.get("detail", ""),
                         "reference": "", "text": text, "sources": {"rules": 1.0}}],
                        "attack": None, "detectors": {"rule": cls}},
                })
        return hits
