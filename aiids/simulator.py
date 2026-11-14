"""Enterprise network simulator: labelled flows + host/auth log events.

Nothing here touches a real network. It synthesises packet timings, sizes and
TCP flags per flow and pushes them through schema.flow_features, so simulated
flows have exactly the same features a pcap would produce.

Topology (four monitored segments, one sensor each):

    corp  10.10.1.0/24  ws01-ws12 (users), adm01 (IT admin)      sensor-corp
    dmz   10.10.2.0/24  web01 (public web app)                   sensor-dmz
    srv   10.10.3.0/24  dc01, dns01, db01, files01, backup01     sensor-srv
    ot    10.10.4.0/24  otgw01 (OT gateway), plc01-03, hmi01     sensor-ot

Days map to splits: day 1 = train, 2 = val (threshold calibration),
3 = feedback (the analyst reviews alerts here), 4 = test. Extra days are test.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import ACK, BENIGN, FIN, PSH, RST, SYN, flow_features, normalise

BASE_TIME = dt.datetime(2026, 10, 5, tzinfo=dt.timezone.utc).timestamp()  # a Monday
DAY = 86400.0
SPLITS = ["train", "val", "feedback", "test"]

FAMILIES = ["portscan", "ssh_bruteforce", "web_sqli", "dns_tunnel", "c2_beacon",
            "smb_lateral", "syn_flood", "slow_exfil"]

USERS = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi",
         "ivan", "judy", "mallory", "oscar"]


@dataclass(frozen=True)
class Host:
    name: str
    ip: str
    role: str
    segment: str


def build_hosts() -> list[Host]:
    hosts = [Host(f"ws{i:02d}", f"10.10.1.{10 + i}", "workstation", "corp") for i in range(1, 13)]
    hosts += [
        Host("adm01", "10.10.1.5", "admin_ws", "corp"),
        Host("web01", "10.10.2.10", "web_server", "dmz"),
        Host("dc01", "10.10.3.5", "domain_controller", "srv"),
        Host("dns01", "10.10.3.53", "dns_server", "srv"),
        Host("db01", "10.10.3.10", "database", "srv"),
        Host("files01", "10.10.3.15", "file_server", "srv"),
        Host("backup01", "10.10.3.20", "backup", "srv"),
        Host("otgw01", "10.10.4.1", "ot_gateway", "ot"),
        Host("plc01", "10.10.4.10", "plc", "ot"),
        Host("plc02", "10.10.4.11", "plc", "ot"),
        Host("plc03", "10.10.4.12", "plc", "ot"),
        Host("hmi01", "10.10.4.20", "hmi", "ot"),
    ]
    return hosts


HOSTS = build_hosts()
BY_NAME = {h.name: h for h in HOSTS}
BY_IP = {h.ip: h for h in HOSTS}
SENSORS = {"corp": "sensor-corp", "dmz": "sensor-dmz", "srv": "sensor-srv", "ot": "sensor-ot"}

# External address pools (documentation ranges for anything hostile).
CDN_PREFIXES = ["34.120", "104.18", "151.101", "142.250", "13.107", "52.84"]
RESOLVERS = ["9.9.9.9", "1.1.1.1"]
UPDATE_SERVER = "13.107.4.50"
ROLLOUT_SERVER = "13.107.9.90"
CLOUD_STORAGE = "162.125.6.18"

# Hour-of-day activity profiles.
WORKDAY = np.array([.03, .03, .03, .03, .03, .03, .05, .3, .8, 1, 1, 1, .6, 1, 1, 1, .9, .6, .25, .1, .08, .06, .05, .04])
PUBLIC = np.array([.35, .3, .25, .25, .25, .3, .4, .6, .8, .9, 1, 1, 1, 1, 1, 1, .95, .9, .85, .8, .7, .6, .5, .4])


def asset_inventory() -> dict:
    """IP -> host metadata, used by correlation to unify IPs and hostnames."""
    return {h.ip: {"host": h.name, "role": h.role, "segment": h.segment} for h in HOSTS}


class Simulator:
    def __init__(self, seed: int = 7, scale: float = 1.0):
        self.rng = np.random.default_rng(seed)
        self.scale = scale
        self.flows: list[dict] = []
        self.events: list[dict] = []
        self.campaigns: list[dict] = []
        self.stealth = False
        self.cdn_pool = [f"{p}.{self.rng.integers(0, 255)}.{self.rng.integers(1, 254)}"
                         for p in CDN_PREFIXES for _ in range(30)]
        self.web_clients = [f"{self.rng.integers(11, 223)}.{self.rng.integers(0, 255)}."
                            f"{self.rng.integers(0, 255)}.{self.rng.integers(1, 254)}" for _ in range(600)]

    # ------------------------------------------------------------------ helpers
    def n(self, mean: float) -> int:
        return int(self.rng.poisson(max(mean * self.scale, 0.0)))

    def times(self, day_start: float, count: int, profile=WORKDAY) -> np.ndarray:
        p = profile / profile.sum()
        hours = self.rng.choice(24, size=count, p=p)
        return np.sort(day_start + hours * 3600 + self.rng.uniform(0, 3600, count))

    def eport(self) -> int:
        return int(self.rng.integers(49152, 65535))

    def _sizes(self, n: int, mean: float, sd: float) -> np.ndarray:
        return np.clip(self.rng.normal(mean, sd, n), 54, 1514)

    def _record(self, ts, sizes, fwd, flags, proto, src, dst, sport, dport, label, campaign, split):
        rec = flow_features(ts, sizes, fwd, flags, proto, dport)
        sh, dh = BY_IP.get(src), BY_IP.get(dst)
        seg = dh.segment if dh else (sh.segment if sh else "dmz")
        rec.update({"ts": float(np.min(ts)), "end_ts": float(np.max(ts)), "src_ip": src, "dst_ip": dst,
                    "src_port": sport, "dst_port": dport, "src_host": sh.name if sh else "",
                    "dst_host": dh.name if dh else "", "sensor": SENSORS[seg], "label": label,
                    "campaign": campaign, "split": split})
        self.flows.append(rec)

    def tcp(self, t0, src, dst, dport, n_fwd, n_bwd, fwd_mean, bwd_mean, duration,
            label=BENIGN, campaign="", split="", close="fin", rtt=None, sport=None):
        """One TCP connection: handshake, data packets, teardown."""
        rtt = rtt if rtt is not None else float(self.rng.uniform(0.0005, 0.03))
        duration = max(duration, rtt * 4)
        ts = [t0, t0 + rtt, t0 + rtt * 1.05]
        sizes = [74, 74, 66]
        fwd = [True, False, True]
        flags = [SYN, SYN | ACK, ACK]
        if n_fwd + n_bwd > 0:
            dt_ = np.sort(self.rng.uniform(rtt * 1.2, duration - rtt * 1.5, n_fwd + n_bwd))
            direction = np.array([True] * n_fwd + [False] * n_bwd)
            self.rng.shuffle(direction)
            ts += list(t0 + dt_)
            sizes += list(np.where(direction, self._sizes(len(direction), fwd_mean, fwd_mean * .25),
                                   self._sizes(len(direction), bwd_mean, bwd_mean * .25)))
            fwd += list(direction)
            flags += list(np.where(self.rng.random(len(direction)) < .6, PSH | ACK, ACK))
        end = t0 + duration
        if close == "fin":
            ts += [end - rtt, end - rtt / 2, end]
            sizes += [66, 66, 66]
            fwd += [True, False, True]
            flags += [FIN | ACK, FIN | ACK, ACK]
        elif close == "rst":
            ts.append(end)
            sizes.append(60)
            fwd.append(False)
            flags.append(RST | ACK)
        sp = sport or self.eport()
        self._record(ts, sizes, fwd, flags, 6, src, dst, sp, dport, label, campaign, split)

    def probe(self, t0, src, dst, dport, open_port, label, campaign, split, sport=None):
        """A SYN probe: closed port answers RST, open port answers SYN-ACK then gets RST."""
        rtt = float(self.rng.uniform(0.0003, 0.01))
        if open_port:
            ts, sizes, fwd, flags = [t0, t0 + rtt, t0 + rtt * 1.1], [60, 60, 54], [True, False, True], [SYN, SYN | ACK, RST]
        else:
            ts, sizes, fwd, flags = [t0, t0 + rtt], [60, 54], [True, False], [SYN, RST | ACK]
        self._record(ts, sizes, fwd, flags, 6, src, dst, sport or self.eport(), dport, label, campaign, split)

    def udp(self, t0, src, dst, dport, req, resp, pairs=1, label=BENIGN, campaign="", split="", gap=0.5):
        ts, sizes, fwd = [], [], []
        t = t0
        for _ in range(pairs):
            rtt = float(self.rng.uniform(0.001, 0.03))
            ts += [t, t + rtt]
            sizes += [max(60, self.rng.normal(req, req * .1)), max(60, self.rng.normal(resp, resp * .15))]
            fwd += [True, False]
            t += rtt + float(self.rng.exponential(gap))
        self._record(ts, sizes, fwd, [0] * len(ts), 17, src, dst, self.eport(), dport, label, campaign, split)

    def event(self, ts, host, etype, user, src_ip, label=BENIGN, campaign="", split="", detail=""):
        self.events.append({"ts": float(ts), "host": host, "sensor": f"hostlog-{host}", "event": etype,
                            "user": user, "src_ip": src_ip, "detail": detail, "label": label,
                            "campaign": campaign, "split": split})

    # ------------------------------------------------------------------ benign
    def benign_day(self, d: int, day0: float, split: str) -> None:
        r = self.rng
        dns, dc, files, web, db = (BY_NAME[n] for n in ("dns01", "dc01", "files01", "web01", "db01"))
        clients = [h for h in HOSTS if h.role in ("workstation", "admin_ws")]
        dns_count = 0
        for i, h in enumerate(clients):
            user = USERS[i] if h.role == "workstation" else "it.admin"
            first = day0 + 3600 * float(r.uniform(7.5, 9.5))
            self.event(first, h.name, "auth_success", user, h.ip, split=split, detail="interactive logon")
            if r.random() < 0.25:
                self.event(first - 20, h.name, "auth_failure", user, h.ip, split=split, detail="bad password")
            for t in self.times(day0, self.n(260)):
                big = r.random() < 0.15
                self.tcp(t, h.ip, str(r.choice(self.cdn_pool)), 443 if r.random() < .9 else 80,
                         int(r.integers(4, 30)), int(r.integers(8, 400 if big else 90)), 300, 1200,
                         float(r.lognormal(1.5, 1.2)), split=split)
            for t in self.times(day0, self.n(220)):
                self.udp(t, h.ip, dns.ip, 53, float(r.uniform(68, 90)), float(r.uniform(90, 240)), split=split)
                dns_count += 1
            for t in self.times(day0, self.n(40)):
                self.tcp(t, h.ip, files.ip, 445, int(r.integers(6, 80)), int(r.integers(6, 120)), 400, 900,
                         float(r.lognormal(2, 1.1)), split=split)
            for t in self.times(day0, self.n(12)):
                self.tcp(t, h.ip, dc.ip, 88, int(r.integers(2, 5)), int(r.integers(2, 5)), 250, 900, float(r.uniform(.02, .3)), split=split)
            for t in self.times(day0, self.n(15)):
                self.tcp(t, h.ip, dc.ip, 389, int(r.integers(3, 12)), int(r.integers(3, 20)), 200, 600, float(r.uniform(.05, 2)), split=split)
            for hh in range(24):
                if r.random() < self.scale * 1.5:
                    self.udp(day0 + hh * 3600 + float(r.uniform(0, 3600)), h.ip, dc.ip, 123, 90, 90, split=split)
            # Background noise: a stale mapped drive or misconfigured app hitting a closed port.
            for t in self.times(day0, self.n(2)):
                tgt = r.choice([files, dc, db])
                self.probe(t, h.ip, tgt.ip, int(r.choice([8080, 139, 5000])), False, BENIGN, "", split)
        # Nightly Windows/patch updates.
        for h in clients:
            if r.random() < 0.8:
                t = day0 + 3 * 3600 + float(r.uniform(0, 3600))
                self.tcp(t, h.ip, UPDATE_SERVER, 443, int(r.integers(80, 300)), int(r.integers(1500, 4000)), 120, 1450,
                         float(r.uniform(30, 240)), split=split)
        # The DNS server forwards part of the queries upstream.
        for t in self.times(day0, int(dns_count * 0.35)):
            self.udp(t, dns.ip, str(r.choice(RESOLVERS)), 53, float(r.uniform(70, 95)), float(r.uniform(100, 300)), split=split)
        self.it_admin(d, day0, split)
        self.web_tier(day0, split)
        self.servers(d, day0, split)
        self.ot(day0, split)
        self.quirks(d, day0, split)

    def it_admin(self, d, day0, split):
        r, adm = self.rng, BY_NAME["adm01"]
        for t in self.times(day0, max(2, self.n(8))):
            tgt = BY_NAME[str(r.choice(["web01", "db01", "dns01", "backup01", "files01"]))]
            self.tcp(t, adm.ip, tgt.ip, 22, int(r.integers(80, 900)), int(r.integers(80, 1200)), 110, 180,
                     float(r.uniform(60, 1500)), split=split)
            self.event(t + 1, tgt.name, "auth_success", "it.admin", adm.ip, split=split, detail="sshd publickey")
        for t in self.times(day0, int(r.integers(1, 3))):
            self.tcp(t, adm.ip, BY_NAME["dc01"].ip, 3389, int(r.integers(200, 900)), int(r.integers(300, 1500)), 150, 700,
                     float(r.uniform(120, 1200)), split=split)
            self.event(t + 1, "dc01", "auth_success", "it.admin", adm.ip, split=split, detail="rdp logon")

    def web_tier(self, day0, split):
        r, web, db = self.rng, BY_NAME["web01"], BY_NAME["db01"]
        for t in self.times(day0, self.n(1500), PUBLIC):
            self.tcp(t, str(r.choice(self.web_clients)), web.ip, 443, int(r.integers(4, 14)), int(r.integers(6, 60)),
                     float(r.uniform(300, 700)), 1300, float(r.lognormal(0, 1)), split=split)
            if r.random() < .4:
                self.tcp(t + .01, web.ip, db.ip, 5432, int(r.integers(2, 8)), int(r.integers(2, 12)), 220, 600,
                         float(r.uniform(.005, .2)), split=split, rtt=.0004)

    def servers(self, d, day0, split):
        r = self.rng
        db, bk, dns, dc = (BY_NAME[n] for n in ("db01", "backup01", "dns01", "dc01"))
        for k in range(int(r.integers(1, 3))):
            t = day0 + 2 * 3600 + k * 900 + float(r.uniform(0, 300))
            self.tcp(t, db.ip, bk.ip, 873, int(r.integers(3000, 6000)), int(r.integers(400, 900)), 1400, 80,
                     float(r.uniform(300, 900)), split=split)
        for name in ("db01", "web01", "files01", "backup01", "dc01"):
            h = BY_NAME[name]
            for t in self.times(day0, self.n(60), PUBLIC):
                self.udp(t, h.ip, dns.ip, 53, float(r.uniform(68, 90)), float(r.uniform(90, 200)), split=split)
            for t in self.times(day0, self.n(6), PUBLIC) if name != "dc01" else []:
                self.tcp(t, h.ip, dc.ip, 88, 3, 3, 250, 900, float(r.uniform(.02, .2)), split=split)

    def ot(self, day0, split):
        r = self.rng
        gw = BY_NAME["otgw01"]
        for plc in ("plc01", "plc02", "plc03"):
            p = BY_NAME[plc]
            steps = int(288 * min(1.0, self.scale * 2))
            for k in range(steps):
                t = day0 + k * (DAY / steps) + float(r.normal(0, .05))
                self.tcp(t, gw.ip, p.ip, 502, 150, 150, 78, 85, 299.0, split=split, rtt=.002, close="none")
        hmi = BY_NAME["hmi01"]
        for t in self.times(day0, self.n(120), PUBLIC):
            self.tcp(t, hmi.ip, gw.ip, 502, int(r.integers(10, 40)), int(r.integers(10, 40)), 80, 140,
                     float(r.uniform(5, 60)), split=split)

    def quirks(self, d, day0, split):
        """Benign but unusual behaviour: the realistic source of false positives."""
        r = self.rng
        adm = BY_NAME["adm01"]
        if d >= 2:  # IT deploys an asset-inventory sweep from day 3 onwards (never seen in training).
            t = day0 + 10 * 3600 + float(r.uniform(0, 1800))
            for name in ("dc01", "db01", "files01", "backup01", "dns01", "web01"):
                for port in (22, 80, 135, 139, 443, 445, 3389, 5432, 5985, 8080, 161, 873):
                    t += float(r.uniform(.05, .4))
                    self.probe(t, adm.ip, BY_NAME[name].ip, port, port in (22, 445, 443), BENIGN, "benign:inventory-sweep", split)
        if d == 3:  # Day 4: a new software package rolls out from a server nobody used before.
            for h in [h for h in HOSTS if h.role == "workstation"]:
                t = day0 + 14 * 3600 + float(r.uniform(0, 1800))
                self.tcp(t, h.ip, ROLLOUT_SERVER, 443, int(r.integers(100, 300)), int(r.integers(2000, 4000)), 120, 1450,
                         float(r.uniform(60, 300)), split=split, campaign="benign:software-rollout")
        if d >= 1 and r.random() < .8:  # Someone syncs a big folder to cloud storage in the evening.
            ws = BY_NAME[f"ws{int(r.integers(1, 13)):02d}"]
            t = day0 + float(r.uniform(19, 22)) * 3600
            for _ in range(int(r.integers(1, 3))):
                self.tcp(t, ws.ip, CLOUD_STORAGE, 443, int(r.integers(1500, 3500)), int(r.integers(100, 300)), 1400, 90,
                         float(r.uniform(60, 400)), split=split, campaign="benign:cloud-sync")
                t += float(r.uniform(400, 1200))
        if d == 3:  # An admin installs a new backup agent on db01 during office hours.
            self.event(day0 + 11.2 * 3600, "db01", "service_install", "it.admin", adm.ip, split=split,
                       campaign="benign:new-agent", detail="service=veeam-agent")
        if d in (0, 2):  # Patch night installs services as SYSTEM.
            for h in [h for h in HOSTS if h.role == "workstation"][:5]:
                self.event(day0 + 3.3 * 3600 + float(r.uniform(0, 1200)), h.name, "service_install", "SYSTEM", h.ip,
                           split=split, detail="service=UpdateOrchestrator")

    # ------------------------------------------------------------------ attacks
    def attacks_day(self, d: int, day0: float, split: str) -> None:
        r = self.rng
        tag = f"d{d + 1}"
        # From day 4 the attackers are more careful (slower scan, more jitter,
        # smaller payloads), so the test day is not a copy of the training day.
        self.stealth = d >= 3
        self.campaign_intrusion(f"{tag}-A", day0 + 3600 * float(r.uniform(8.5, 13)), d, split)
        self.campaign_credentials(f"{tag}-B", day0 + 3600 * float(r.uniform(17, 18.5)), d, split)
        self.syn_flood(f"{tag}-C", day0 + 3600 * float(r.uniform(1, 23)), d, split)
        if d % 2 == 1:
            self.campaign_insider(f"{tag}-D", day0 + 3600 * float(r.uniform(9, 15)), d, split)

    def _stage(self, cid, family, stage, start, end, **extra):
        for c in self.campaigns:
            if c["id"] == cid:
                c["steps"].append({"family": family, "stage": stage, "start": start, "end": end, **extra})

    def campaign_intrusion(self, cid, t0, d, split):
        """Scan -> SQLi -> persistence -> lateral SMB -> C2 beacon -> DNS tunnel exfil."""
        r = self.rng
        web, db, dns = BY_NAME["web01"], BY_NAME["db01"], BY_NAME["dns01"]
        attacker = f"203.0.113.{10 + d}"
        c2 = f"198.51.100.{23 + d}"
        self.campaigns.append({"id": cid, "name": "external intrusion chain", "attacker": attacker, "steps": []})
        # 1. Port scan of the DMZ web server.
        t = t0
        ports = r.choice(np.arange(1, 10000), size=int(r.integers(350, 600)), replace=False)
        for p in ports:
            t += float(r.exponential(1 / float(r.uniform(0.4, 1.5) if self.stealth else r.uniform(15, 60))))
            self.probe(t, attacker, web.ip, int(p), int(p) in (22, 80, 443), "portscan", cid, split)
        self._stage(cid, "portscan", "recon", t0, t)
        # 2. SQL injection bursts against the web app, with the database answering big dumps.
        t = t0 + float(r.uniform(600, 1200))
        s0 = t
        for _ in range(int(r.integers(3, 5))):
            for _ in range(int(r.integers(35, 70))):
                t += float(r.exponential(1.5))
                self.tcp(t, attacker, web.ip, 443, int(r.integers(4, 10)), int(r.integers(4, 30)),
                         float(r.uniform(600, 1000) if self.stealth else r.uniform(900, 1400)), float(r.uniform(400, 1300)), float(r.uniform(.05, .8)),
                         "web_sqli", cid, split)
                if r.random() < .5:
                    self.tcp(t + .02, web.ip, db.ip, 5432, int(r.integers(3, 8)), int(r.integers(40, 300)), 400, 1300,
                             float(r.uniform(.2, 2)), "web_sqli", cid, split, rtt=.0004)
            t += float(r.uniform(60, 240))
        self._stage(cid, "web_sqli", "initial_access", s0, t)
        # 3. Persistence: a new service on the web server (host log only).
        t_p = t + float(r.uniform(300, 900))
        svc = f"svc{int(r.integers(100, 999))}"
        self.event(t_p, "web01", "service_install", "www-data", web.ip, "service_install", cid, split, f"service={svc}")
        self._stage(cid, "service_install", "persistence", t_p, t_p)
        # 4. C2 beaconing with jitter.
        period = float(r.uniform(45, 90))
        jitter = float(r.uniform(.3, .5) if self.stealth else r.uniform(.05, .25))
        t = t_p + float(r.uniform(120, 600))
        b0, b_end = t, t + 3600 * float(r.uniform(3, 4.5))
        while t < b_end:
            self.tcp(t, web.ip, c2, 443, int(r.integers(3, 6)), int(r.integers(3, 6)), float(r.uniform(250, 450)),
                     float(r.uniform(150, 350)), float(r.uniform(.2, .9)), "c2_beacon", cid, split)
            t += period * (1 + float(r.uniform(-jitter, jitter)))
        self._stage(cid, "c2_beacon", "command_and_control", b0, b_end, c2=c2)
        # 5. Lateral movement over SMB from the web server.
        t = t_p + float(r.uniform(1200, 2400))
        l0 = t
        targets = ["db01", "files01", "dc01"] + [f"ws{int(x):02d}" for x in r.choice(np.arange(1, 13), 2, replace=False)]
        for name in targets:
            tgt = BY_NAME[name]
            for _ in range(int(r.integers(4, 10))):
                t += float(r.exponential(8))
                self.tcp(t, web.ip, tgt.ip, int(r.choice([445, 445, 445, 135])), int(r.integers(20, 200)),
                         int(r.integers(10, 60)), 1300, 300, float(r.uniform(.5, 20)), "smb_lateral", cid, split)
            self.event(t, name, "auth_success", "svc_web", web.ip, "smb_lateral", cid, split, "network logon (NTLM)")
        self._stage(cid, "smb_lateral", "lateral_movement", l0, t)
        # 6. DNS tunnelling exfiltration from the database server.
        t = t + float(r.uniform(900, 1800))
        x0 = t
        ns = f"198.51.100.{53 + d}"
        for _ in range(int(r.integers(300, 500))):
            t += float(r.exponential(6 if self.stealth else 2.5))
            q = float(r.uniform(130, 190) if self.stealth else r.uniform(180, 260))
            self.udp(t, db.ip, dns.ip, 53, q, float(r.uniform(220, 420)), 1, "dns_tunnel", cid, split)
            if r.random() < .6:
                self.udp(t + .003, dns.ip, ns, 53, q, float(r.uniform(220, 420)), 1, "dns_tunnel", cid, split)
        self._stage(cid, "dns_tunnel", "exfiltration", x0, t)

    def campaign_credentials(self, cid, t0, d, split):
        """SSH brute force -> valid login -> pivot to db01 -> slow-and-low exfil."""
        r = self.rng
        web, db = BY_NAME["web01"], BY_NAME["db01"]
        attacker = f"203.0.113.{100 + d}"
        self.campaigns.append({"id": cid, "name": "credential attack and slow exfil", "attacker": attacker, "steps": []})
        t = t0
        for k in range(int(r.integers(150, 280))):
            t += float(r.uniform(1.5, 5))
            self.tcp(t, attacker, web.ip, 22, int(r.integers(10, 18)), int(r.integers(10, 18)), 120, 150,
                     float(r.uniform(1.5, 5)), "ssh_bruteforce", cid, split)
            self.event(t + 1, "web01", "auth_failure", str(r.choice(["root", "admin", "deploy", "ubuntu", "test"])),
                       attacker, "ssh_bruteforce", cid, split, "sshd password")
        self.event(t + 6, "web01", "auth_success", "deploy", attacker, "ssh_bruteforce", cid, split, "sshd password")
        self._stage(cid, "ssh_bruteforce", "initial_access", t0, t)
        t += float(r.uniform(300, 900))
        self.tcp(t, web.ip, db.ip, 22, 60, 80, 110, 200, 400.0, "slow_exfil", cid, split)
        self.event(t + 1, "db01", "auth_success", "deploy", web.ip, "ssh_bruteforce", cid, split, "sshd password")
        dest = f"192.0.2.{40 + d}"
        t += float(r.uniform(900, 1800))
        x0 = t
        end = min(t0 + 6.5 * 3600, t0 - (t0 % DAY) + DAY - 60)
        while t < end:
            self.tcp(t, db.ip, dest, 443, int(r.integers(40, 150) if self.stealth else r.integers(150, 500)), int(r.integers(20, 60)), 1400, 90,
                     float(r.uniform(20, 90)), "slow_exfil", cid, split)
            t += float(r.uniform(600, 1200))
        self._stage(cid, "slow_exfil", "exfiltration", x0, t, destination=dest)

    def syn_flood(self, cid, t0, d, split):
        r = self.rng
        web = BY_NAME["web01"]
        self.campaigns.append({"id": cid, "name": "SYN flood", "attacker": "spoofed", "steps": []})
        t = t0
        for _ in range(int(r.integers(800, 1300))):
            t += float(r.exponential(.08))
            src = f"{r.integers(11, 223)}.{r.integers(0, 255)}.{r.integers(0, 255)}.{r.integers(1, 254)}"
            rtt = float(r.uniform(.001, .02))
            n_syn = int(r.integers(1, 3))
            ts = [t + i * 1.0 for i in range(n_syn)] + [t + rtt]
            self._record(ts, [60] * n_syn + [58], [True] * n_syn + [False], [SYN] * n_syn + [SYN | ACK], 6,
                         src, web.ip, self.eport(), 443, "syn_flood", cid, split)
        self._stage(cid, "syn_flood", "impact", t0, t)

    def campaign_insider(self, cid, t0, d, split):
        """A compromised workstation sweeps the server segment then moves over SMB."""
        r = self.rng
        ws = BY_NAME[f"ws{int(r.integers(1, 13)):02d}"]
        self.campaigns.append({"id": cid, "name": "internal recon from workstation", "attacker": ws.ip, "steps": []})
        t = t0
        for name in ("dc01", "db01", "files01", "backup01", "dns01", "web01", "otgw01"):
            for p in (21, 22, 23, 80, 135, 139, 443, 445, 1433, 3306, 3389, 5432, 5985, 502, 8080):
                t += float(r.exponential(.3))
                self.probe(t, ws.ip, BY_NAME[name].ip, p, p in (22, 445, 443, 502), "portscan", cid, split)
        self._stage(cid, "portscan", "recon", t0, t)
        t += float(r.uniform(600, 1500))
        l0 = t
        for name in ("files01", "backup01"):
            for _ in range(int(r.integers(5, 12))):
                t += float(r.exponential(6))
                self.tcp(t, ws.ip, BY_NAME[name].ip, 445, int(r.integers(30, 200)), int(r.integers(10, 50)), 1300, 300,
                         float(r.uniform(.5, 15)), "smb_lateral", cid, split)
            self.event(t, name, "auth_success", "it.admin", ws.ip, "smb_lateral", cid, split, "network logon (NTLM)")
        self._stage(cid, "smb_lateral", "lateral_movement", l0, t)

    # ------------------------------------------------------------------ driver
    def run(self, days: int = 4) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        for d in range(days):
            day0 = BASE_TIME + d * DAY
            split = SPLITS[min(d, len(SPLITS) - 1)]
            self.benign_day(d, day0, split)
            self.attacks_day(d, day0, split)
        flows = normalise(pd.DataFrame(self.flows))
        events = pd.DataFrame(self.events).sort_values("ts").reset_index(drop=True)
        meta = {"kind": "simulated", "days": days, "scale": self.scale, "campaigns": self.campaigns,
                "assets": asset_inventory(), "families": FAMILIES}
        return flows, events, meta


def simulate(days: int = 4, seed: int = 7, scale: float = 1.0):
    """Return (flows, host_events, meta) for a simulated enterprise."""
    return Simulator(seed=seed, scale=scale).run(days)
