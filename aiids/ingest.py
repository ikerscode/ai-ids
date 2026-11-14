"""External alert ingestion (POST /api/ingest) with strict schema validation.

Other lab tools (honeypot-net, ics-sim, malware-lab, ...) push alerts here so
they correlate with network and host detections. Unknown top-level fields are
rejected; anything tool-specific goes into "details".
"""
from __future__ import annotations

import datetime as dt
import ipaddress
import re
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .attack import FAMILY_MAP, STAGES, TECHNIQUES, map_family, technique

SEVERITY_CONF = {"low": 0.5, "medium": 0.7, "high": 0.85, "critical": 0.95}
KEYWORDS = [("scan", "T1046"), ("recon", "T1046"), ("brute", "T1110"), ("login", "T1110"), ("credential", "T1110"),
            ("exploit", "T1190"), ("modbus_write", "T0855"), ("ics_write", "T0855"), ("plc", "T0855"),
            ("malware", "T1204"), ("exec", "T1059"), ("persist", "T1543.003"), ("beacon", "T1071.001"),
            ("c2", "T1071.001"), ("exfil", "T1048"), ("lateral", "T1021"), ("dos", "T1498.001")]


class ExternalAlert(BaseModel):
    """One alert from another system. At least one entity field is required."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source: str = Field(..., description="Sending system, e.g. honeypot-net, ics-sim, malware-lab")
    ts: float | str | None = Field(None, validate_default=True, description="Epoch seconds or ISO-8601; default = now")
    category: str = Field(..., max_length=64, description="What happened, e.g. ssh_bruteforce, modbus_write, malware_exec")
    src_ip: str | None = None
    dst_ip: str | None = None
    src_host: str | None = Field(None, max_length=64)
    dst_host: str | None = Field(None, max_length=64)
    user: str | None = Field(None, max_length=64)
    dst_port: int | None = Field(None, ge=0, le=65535)
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    confidence: float | None = Field(None, ge=0, le=1)
    technique_id: str | None = Field(None, description="MITRE ATT&CK id, e.g. T1110")
    stage: str | None = Field(None, description="Kill-chain stage override: " + ", ".join(STAGES))
    summary: str | None = Field(None, max_length=500)
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source")
    @classmethod
    def _source(cls, v: str) -> str:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", v):
            raise ValueError("source must be lowercase letters, digits, '.', '_' or '-' (max 64)")
        return v

    @field_validator("src_ip", "dst_ip")
    @classmethod
    def _ip(cls, v):
        if v in (None, ""):
            return None
        return str(ipaddress.ip_address(v))

    @field_validator("technique_id")
    @classmethod
    def _tid(cls, v):
        if v and not re.fullmatch(r"T\d{4}(\.\d{3})?", v):
            raise ValueError("technique_id must look like T1110 or T1110.001")
        return v

    @field_validator("stage")
    @classmethod
    def _stage(cls, v):
        if v and v not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        return v

    @field_validator("ts")
    @classmethod
    def _ts(cls, v):
        if v is None:
            return time.time()
        if isinstance(v, (int, float)):
            return float(v)
        return dt.datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()

    @model_validator(mode="after")
    def _entity(self):
        if not any([self.src_ip, self.dst_ip, self.src_host, self.dst_host, self.user]):
            raise ValueError("at least one of src_ip, dst_ip, src_host, dst_host, user is required")
        if len(str(self.details)) > 4000:
            raise ValueError("details too large (max ~4 KB)")
        return self


def to_alert(e: ExternalAlert) -> dict:
    """Normalise an external alert into the common alert record."""
    cat = e.category.lower()
    if e.technique_id:
        attack = technique(e.technique_id)
        attack["rationale"] = f"technique supplied by {e.source}"
    elif cat in FAMILY_MAP:
        attack = map_family(cat)
    else:
        tid = next((t for k, t in KEYWORDS if k in cat), "")
        attack = technique(tid) if tid else {"technique_id": "", "technique": "Unmapped", "stage": "recon"}
        attack["rationale"] = f"mapped from category keyword '{cat}'" if tid else "no mapping for this category"
    if e.stage:
        attack["stage"] = e.stage
    conf = e.confidence if e.confidence is not None else SEVERITY_CONF[e.severity]
    summary = e.summary or f"{e.source}: {e.category} {e.src_ip or e.src_host or ''} -> {e.dst_ip or e.dst_host or ''}"
    return {
        "ts": e.ts, "end_ts": e.ts, "source": f"external:{e.source}", "sensor": e.source,
        "src_ip": e.src_ip or "", "dst_ip": e.dst_ip or "", "src_host": e.src_host or "", "dst_host": e.dst_host or "",
        "user": e.user or "", "dst_port": e.dst_port or 0, "alert_class": cat,
        "technique_id": attack.get("technique_id", ""), "technique": attack.get("technique", ""),
        "stage": attack["stage"], "rationale": attack.get("rationale", ""), "confidence": round(float(conf), 3),
        "severity": e.severity, "score": float(conf), "count": 1, "summary": summary,
        "explanation": {"summary": summary, "attack": attack, "contributions": [
            {"feature": "external", "label": f"reported by {e.source}", "weight": 1.0, "value": cat, "reference": "",
             "text": summary, "sources": {e.source: 1.0}}], "detectors": {"external": e.source}},
        "raw": e.model_dump(), "dataset": "", "rows": [], "truth": "", "truth_attack_frac": 0, "campaign": "",
        "rule_id": "", "status": "open",
    }


EXAMPLE = {
    "source": "honeypot-net", "ts": "2026-10-08T11:05:00Z", "category": "ssh_bruteforce",
    "src_ip": "203.0.113.13", "dst_ip": "10.10.2.10", "dst_port": 22, "user": "root", "severity": "high",
    "confidence": 0.9, "summary": "cowrie: 212 login attempts in 4 minutes",
    "details": {"sensor": "ssh-2222", "attempts": 212},
}


def schema_doc() -> dict:
    return {"json_schema": ExternalAlert.model_json_schema(), "example": EXAMPLE,
            "batch": "POST either one object or {\"alerts\": [ ... up to 500 objects ... ]}",
            "stages": STAGES, "known_techniques": sorted(TECHNIQUES)}
