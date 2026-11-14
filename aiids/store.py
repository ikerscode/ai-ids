"""SQLite storage for alerts, analyst feedback, suppression rules and ingested events."""
from __future__ import annotations

import json
import sqlite3
import threading
import time

import numpy as np

from . import config

ALERT_COLUMNS = ["ts", "end_ts", "source", "sensor", "src_ip", "dst_ip", "src_host", "dst_host", "user", "dst_port",
                 "alert_class", "technique_id", "technique", "stage", "rationale", "confidence", "severity", "score",
                 "count", "summary", "explanation", "raw", "dataset", "rows", "truth", "truth_attack_frac", "campaign",
                 "rule_id", "status", "run"]
JSON_COLUMNS = {"explanation", "raw", "rows"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, end_ts REAL, source TEXT, sensor TEXT, src_ip TEXT, dst_ip TEXT,
  src_host TEXT, dst_host TEXT, user TEXT, dst_port INTEGER, alert_class TEXT, technique_id TEXT, technique TEXT,
  stage TEXT, rationale TEXT, confidence REAL, severity TEXT, score REAL, count INTEGER, summary TEXT,
  explanation TEXT, raw TEXT, dataset TEXT, rows TEXT, truth TEXT, truth_attack_frac REAL, campaign TEXT,
  rule_id TEXT, status TEXT, run TEXT, created REAL);
CREATE INDEX IF NOT EXISTS alerts_ts ON alerts(ts);
CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, alert_id INTEGER, verdict TEXT,
  note TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS rules (id INTEGER PRIMARY KEY AUTOINCREMENT, src_ip TEXT, dst_ip TEXT, dst_port TEXT,
  alert_class TEXT, reason TEXT, created REAL, hits INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS labelled_flows (id INTEGER PRIMARY KEY AUTOINCREMENT, alert_id INTEGER, dataset TEXT,
  row INTEGER, verdict TEXT, alert_class TEXT);
CREATE TABLE IF NOT EXISTS external (id INTEGER PRIMARY KEY AUTOINCREMENT, received REAL, source TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
"""


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.ndarray, set, tuple)):
        return list(o)
    return str(o)


def dumps(o) -> str:
    return json.dumps(o, default=_default)


class Store:
    def __init__(self, path=None):
        config.ensure_dirs()
        self.path = str(path or config.DB_PATH)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()
        self.version = 0  # bumps on every alert change so incident caches know when to refresh

    # ---------------------------------------------------------------- alerts
    def _row(self, a: dict) -> list:
        vals = []
        for c in ALERT_COLUMNS:
            v = a.get(c, "")
            vals.append(dumps(v) if c in JSON_COLUMNS else _default(v) if isinstance(v, np.generic) else v)
        return vals

    def insert_alerts(self, alerts: list[dict]) -> list[int]:
        ids = []
        with self.lock:
            cur = self.conn.cursor()
            for a in alerts:
                cur.execute(f"INSERT INTO alerts ({','.join(ALERT_COLUMNS)}, created) VALUES "
                            f"({','.join('?' * len(ALERT_COLUMNS))}, ?)", self._row(a) + [time.time()])
                ids.append(cur.lastrowid)
            self.conn.commit()
            self.version += 1
        return ids

    def update_alert(self, alert_id: int, a: dict) -> None:
        cols = [c for c in ALERT_COLUMNS if c in a and c not in ("status", "run")]
        row = dict(zip(ALERT_COLUMNS, self._row(a)))
        with self.lock:
            self.conn.execute(f"UPDATE alerts SET {','.join(c + '=?' for c in cols)} WHERE id=?",
                              [row[c] for c in cols] + [alert_id])
            self.conn.commit()
            self.version += 1

    def _decode(self, r: sqlite3.Row, full: bool = True) -> dict:
        d = dict(r)
        for c in JSON_COLUMNS:
            if c in d:
                if full:
                    try:
                        d[c] = json.loads(d[c]) if d[c] else None
                    except json.JSONDecodeError:
                        d[c] = None
                else:
                    d.pop(c)
        return d

    def list_alerts(self, limit: int = 500, since_id: int = 0, cls: str = "", severity: str = "", source: str = "",
                    status: str = "", q: str = "", full: bool = False) -> list[dict]:
        sql, args = "SELECT * FROM alerts WHERE id > ?", [since_id]
        for col, val in (("alert_class", cls), ("severity", severity), ("status", status)):
            if val:
                sql += f" AND {col} = ?"
                args.append(val)
        if source:
            sql += " AND (source LIKE ? OR sensor LIKE ?)"
            args += [f"%{source}%"] * 2
        if q:
            sql += " AND (src_ip LIKE ? OR dst_ip LIKE ? OR src_host LIKE ? OR dst_host LIKE ? OR summary LIKE ?)"
            args += [f"%{q}%"] * 5
        sql += " ORDER BY ts DESC, id DESC LIMIT ?"
        args.append(limit)
        with self.lock:
            return [self._decode(r, full) for r in self.conn.execute(sql, args).fetchall()]

    def all_alerts_light(self) -> list[dict]:
        with self.lock:
            rows = self.conn.execute("SELECT id, ts, end_ts, source, sensor, src_ip, dst_ip, src_host, dst_host, user, "
                                     "alert_class, technique_id, stage, confidence, severity, count, summary, campaign, "
                                     "truth, status FROM alerts WHERE status != 'fp' ORDER BY ts").fetchall()
        return [dict(r) for r in rows]

    def get_alert(self, alert_id: int) -> dict | None:
        with self.lock:
            r = self.conn.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
        return self._decode(r) if r else None

    def set_status(self, alert_id: int, status: str) -> None:
        with self.lock:
            self.conn.execute("UPDATE alerts SET status=? WHERE id=?", (status, alert_id))
            self.conn.commit()
            self.version += 1

    def clear_alerts(self) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM alerts")
            self.conn.commit()
            self.version += 1

    def alert_counts(self) -> dict:
        with self.lock:
            rows = self.conn.execute("SELECT severity, COUNT(*) n FROM alerts GROUP BY severity").fetchall()
            total = self.conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        return {"total": total, **{r["severity"]: r["n"] for r in rows}}

    # ---------------------------------------------------------------- feedback / rules
    def add_feedback(self, alert_id: int, verdict: str, note: str = "") -> None:
        with self.lock:
            self.conn.execute("INSERT INTO feedback (alert_id, verdict, note, ts) VALUES (?,?,?,?)",
                              (alert_id, verdict, note, time.time()))
            self.conn.commit()

    def add_labelled_flows(self, alert_id: int, dataset: str, rows: list[int], verdict: str, cls: str) -> None:
        with self.lock:
            self.conn.executemany("INSERT INTO labelled_flows (alert_id, dataset, row, verdict, alert_class) "
                                  "VALUES (?,?,?,?,?)", [(alert_id, dataset, int(r), verdict, cls) for r in rows])
            self.conn.commit()

    def labelled_flows(self) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute("SELECT * FROM labelled_flows").fetchall()]

    def feedback(self) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(
                "SELECT f.*, a.alert_class, a.src_ip, a.dst_ip FROM feedback f LEFT JOIN alerts a ON a.id=f.alert_id "
                "ORDER BY f.id DESC").fetchall()]

    def add_rule(self, rule: dict) -> int:
        with self.lock:
            cur = self.conn.execute("INSERT INTO rules (src_ip, dst_ip, dst_port, alert_class, reason, created) "
                                    "VALUES (?,?,?,?,?,?)",
                                    (rule.get("src_ip", "*"), rule.get("dst_ip", "*"), str(rule.get("dst_port", "*")),
                                     rule.get("alert_class", "*"), rule.get("reason", ""), time.time()))
            self.conn.commit()
            return cur.lastrowid

    def rules(self) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute("SELECT * FROM rules ORDER BY id").fetchall()]

    def delete_rule(self, rule_id: int) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM rules WHERE id=?", (rule_id,))
            self.conn.commit()

    def bump_rule_hits(self, counts: dict) -> None:
        with self.lock:
            for rid, n in counts.items():
                self.conn.execute("UPDATE rules SET hits = hits + ? WHERE id=?", (int(n), int(rid)))
            self.conn.commit()

    # ---------------------------------------------------------------- external + settings
    def add_external(self, source: str, payload: dict) -> int:
        with self.lock:
            cur = self.conn.execute("INSERT INTO external (received, source, payload) VALUES (?,?,?)",
                                    (time.time(), source, dumps(payload)))
            self.conn.commit()
            return cur.lastrowid

    def get_setting(self, key: str, default: str = "") -> str:
        with self.lock:
            r = self.conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return r["value"] if r else default

    def set_setting(self, key: str, value: str) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?,?)", (key, value))
            self.conn.commit()
