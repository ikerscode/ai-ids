"""Cross-system correlation: alerts from every sensor -> incidents.

Each alert is reduced to entities (hosts, IPs, users) in two roles:
  actor  = source address/host and the account used
  target = destination address/host

An alert joins an incident when they share an entity and:
  * the new alert's actor was an actor earlier (same attacker IP, account or
    compromised host), within CORR_GAP_SECONDS
  * pivot: the new alert's actor was the target of an earlier compromising step
    (initial access, persistence, lateral movement), e.g. web01 is exploited and
    then web01 attacks db01, within CORR_GAP_SECONDS
  * the new alert only shares a target (same victim, possibly different
    attackers), within the much shorter VICTIM_GAP_SECONDS, so a busy server
    does not glue unrelated activity together for hours

If an alert matches several incidents they merge. Incidents are then scored
by how far they walk along the kill chain, how many systems saw them and how
confident the evidence is.
"""
from __future__ import annotations

from collections import defaultdict

from . import config
from .attack import STAGE_LABELS, STAGES
from .simulator import asset_inventory

VICTIM_GAP_SECONDS = 120
COMPROMISING = {"initial_access", "execution_persistence", "lateral_movement"}
IGNORED_USERS = {"", "system", "root", "-", "unknown", "anonymous"}


class Correlator:
    def __init__(self, assets: dict | None = None, gap: float = config.CORR_GAP_SECONDS,
                 victim_gap: float = VICTIM_GAP_SECONDS):
        self.assets = assets if assets is not None else asset_inventory()
        self.host_names = {v["host"] for v in self.assets.values()}
        self.gap = gap
        self.victim_gap = victim_gap

    # ------------------------------------------------------------------ entities
    def _ent(self, ip: str, host: str) -> str | None:
        host = (host or "").strip().lower()
        ip = (ip or "").strip()
        if ip in self.assets:
            return f"host:{self.assets[ip]['host']}"
        if host:
            return f"host:{host}"
        if ip and ip not in ("0.0.0.0", "::"):
            return f"ip:{ip}"
        return None

    def roles(self, a: dict) -> tuple[set, set]:
        actors, targets = set(), set()
        src = self._ent(a.get("src_ip", ""), a.get("src_host", ""))
        dst = self._ent(a.get("dst_ip", ""), a.get("dst_host", ""))
        if src:
            actors.add(src)
        if dst:
            targets.add(dst)
        user = str(a.get("user") or "").strip().lower()
        if user not in IGNORED_USERS:
            actors.add(f"user:{user}")
        return actors, targets

    # ------------------------------------------------------------------ grouping
    def correlate(self, alerts: list[dict]) -> list[dict]:
        alerts = sorted(alerts, key=lambda a: (a["ts"], a.get("id", 0)))
        parent: dict[int, int] = {}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        # entity -> role -> [(alert index, start, end, compromising)]
        seen: dict[str, dict[str, list]] = defaultdict(lambda: {"actor": [], "target": []})
        for i, a in enumerate(alerts):
            parent[i] = i
            actors, targets = self.roles(a)
            t = float(a["ts"])
            links = set()
            for e in actors:
                # Same actor again (attacker IP, account, compromised host) within the gap.
                for j, _, last, _ in seen[e]["actor"]:
                    if t - last <= self.gap:
                        links.add(find(j))
                # Pivot: this actor was the target of an earlier step that can compromise a
                # host (initial access, execution/persistence, lateral movement). Being
                # scanned or flooded does not make a host part of the attacker's chain.
                for j, _, last, compromising in seen[e]["target"]:
                    if compromising and t - last <= self.gap:
                        links.add(find(j))
            for e in targets:
                # Shared victim only: link just when (nearly) simultaneous, so busy
                # servers such as DNS or the web server do not glue unrelated activity.
                for j, start, _, _ in seen[e]["actor"] + seen[e]["target"]:
                    if abs(t - start) <= self.victim_gap:
                        links.add(find(j))
            for j in links:
                parent[find(j)] = i
            end = float(a.get("end_ts") or t)
            compromising = a.get("stage") in COMPROMISING
            for role, ents in (("actor", actors), ("target", targets)):
                for e in ents:
                    keep = [x for x in seen[e][role] if t - x[2] <= self.gap][-20:]
                    seen[e][role] = keep + [(i, t, end, compromising)]

        groups: dict[int, list[dict]] = defaultdict(list)
        for i, a in enumerate(alerts):
            groups[find(i)].append(a)
        incidents = [self.summarise(g) for g in groups.values()]
        return sorted(incidents, key=lambda x: (-x["score"], -x["last_ts"]))

    # ------------------------------------------------------------------ scoring
    def summarise(self, alerts: list[dict]) -> dict:
        alerts = sorted(alerts, key=lambda a: a["ts"])
        stages: dict[str, float] = {}
        for a in alerts:
            st = a.get("stage") or "recon"
            stages[st] = min(stages.get(st, a["ts"]), a["ts"])
        ordered = [s for s in STAGES if s in stages]
        firsts = [stages[s] for s in ordered]
        progression = len(ordered) >= 2 and all(x <= y + 60 for x, y in zip(firsts, firsts[1:]))
        systems = sorted({a.get("sensor") or a.get("source", "") for a in alerts})
        max_conf = max(float(a.get("confidence", 0)) for a in alerts)
        mean_conf = sum(float(a.get("confidence", 0)) for a in alerts) / len(alerts)
        points = (1.2 * (len(ordered) - 1) + 0.8 * min(len(systems) - 1, 4) + 2.0 * max_conf
                  + (1.0 if progression and len(ordered) >= 3 else 0.0)
                  + (0.5 if any(a.get("severity") == "critical" for a in alerts) else 0.0))
        score = round(min(100.0, points * 10), 1)
        sev = "low" if points < 2.5 else "medium" if points < 4.5 else "high" if points < 7 else "critical"
        nodes, links = {}, []
        for a in alerts:
            actors, targets = self.roles(a)
            for e in actors | targets:
                kind, name = e.split(":", 1)
                if kind == "host":
                    kind = "internal host" if name in self.host_names else "host"
                nodes[e] = {"id": e, "label": name, "type": kind}
            for s in actors:
                for d in targets:
                    if s != d:
                        links.append({"source": s, "target": d, "label": a.get("technique_id") or a["alert_class"]})
        uniq_links = {(l["source"], l["target"], l["label"]): l for l in links}
        campaigns = defaultdict(int)
        for a in alerts:
            if a.get("campaign"):
                campaigns[a["campaign"]] += 1
        first = alerts[0]
        title_bits = [STAGE_LABELS[s] for s in ordered]
        return {
            "id": f"INC-{first.get('id', 0)}",
            "first_ts": first["ts"], "last_ts": max(a.get("end_ts", a["ts"]) for a in alerts),
            "n_alerts": len(alerts), "alert_ids": [a.get("id") for a in alerts],
            "stages": {s: stages[s] for s in ordered}, "stage_count": len(ordered), "progression": progression,
            "systems": systems, "max_confidence": round(max_conf, 3), "mean_confidence": round(mean_conf, 3),
            "score": score, "severity": sev,
            "title": " -> ".join(title_bits) + f" ({', '.join(sorted({n['label'] for n in nodes.values() if n['type'] == 'internal host'})[:4])})",
            "techniques": sorted({a.get("technique_id", "") for a in alerts if a.get("technique_id")}),
            "entities": sorted(nodes), "graph": {"nodes": list(nodes.values()), "links": list(uniq_links.values())},
            "campaigns": dict(campaigns),
            "timeline": [{"id": a.get("id"), "ts": a["ts"], "stage": a.get("stage"), "alert_class": a["alert_class"],
                          "technique_id": a.get("technique_id", ""), "sensor": a.get("sensor", ""),
                          "summary": a.get("summary", ""), "severity": a.get("severity", ""),
                          "count": a.get("count", 1)} for a in alerts],
        }
