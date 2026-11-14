"""Bidirectional flow aggregator with idle/active timeouts and TCP teardown.

Packets go in (in time order), finished flows come out. This is the same
logic a NetFlow/IPFIX exporter or CICFlowMeter uses, kept small enough to read:

* key = the unordered pair of (ip, port) endpoints plus the protocol
* forward direction = whoever sent the first packet we saw
* a flow ends on: idle timeout, active timeout, RST, or FIN from both sides
  (a later SYN on the same key then starts a new flow)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .schema import FIN, RST, SYN, flow_features


@dataclass
class Packet:
    ts: float
    src: str
    dst: str
    sport: int
    dport: int
    proto: int
    size: int
    flags: int = 0


@dataclass
class _Flow:
    src: str
    dst: str
    sport: int
    dport: int
    proto: int
    start: float
    last: float
    ts: list = field(default_factory=list)
    sizes: list = field(default_factory=list)
    fwd: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    fin_fwd: bool = False
    fin_bwd: bool = False
    closed: bool = False

    def add(self, p: Packet, is_fwd: bool) -> None:
        self.ts.append(p.ts)
        self.sizes.append(p.size)
        self.fwd.append(is_fwd)
        self.flags.append(p.flags)
        self.last = p.ts
        if p.flags & RST:
            self.closed = True
        if p.flags & FIN:
            if is_fwd:
                self.fin_fwd = True
            else:
                self.fin_bwd = True
            if self.fin_fwd and self.fin_bwd:
                self.closed = True


class FlowAggregator:
    """Turn a packet stream into flow records (dicts with schema features)."""

    def __init__(self, idle_timeout: float = 120.0, active_timeout: float = 1800.0,
                 sensor: str = "pcap"):
        self.idle_timeout = idle_timeout
        self.active_timeout = active_timeout
        self.sensor = sensor
        self.active: dict[tuple, _Flow] = {}
        self.finished: list[dict] = []
        self._since_sweep = 0

    @staticmethod
    def key(p: Packet) -> tuple:
        a, b = (p.src, p.sport), (p.dst, p.dport)
        return (min(a, b), max(a, b), p.proto)

    def add(self, p: Packet) -> None:
        k = self.key(p)
        flow = self.active.get(k)
        if flow is not None:
            expired = (p.ts - flow.last > self.idle_timeout
                       or p.ts - flow.start > self.active_timeout
                       or (flow.closed and p.flags & SYN and not p.flags & 0x10))
            if expired:
                self._finish(k)
                flow = None
        if flow is None:
            flow = _Flow(p.src, p.dst, p.sport, p.dport, p.proto, p.ts, p.ts)
            self.active[k] = flow
        flow.add(p, is_fwd=(p.src == flow.src and p.sport == flow.sport))
        self._since_sweep += 1
        if self._since_sweep >= 5000:
            self.sweep(p.ts)

    def sweep(self, now: float) -> None:
        """Expire idle flows so memory stays bounded on long captures."""
        self._since_sweep = 0
        for k in [k for k, f in self.active.items() if now - f.last > self.idle_timeout]:
            self._finish(k)

    def flush(self) -> list[dict]:
        for k in list(self.active):
            self._finish(k)
        return self.finished

    def _finish(self, k: tuple) -> None:
        f = self.active.pop(k)
        rec = flow_features(f.ts, f.sizes, f.fwd, f.flags, f.proto, f.dport)
        rec.update({"ts": f.start, "end_ts": f.last, "src_ip": f.src, "dst_ip": f.dst,
                    "src_port": f.sport, "dst_port": f.dport, "sensor": self.sensor})
        self.finished.append(rec)


def packets_to_flows(packets, sensor: str = "pcap", idle_timeout: float = 120.0,
                     active_timeout: float = 1800.0) -> pd.DataFrame:
    """Aggregate an iterable of Packet (any order) into a raw flow DataFrame."""
    agg = FlowAggregator(idle_timeout, active_timeout, sensor)
    for p in sorted(packets, key=lambda x: x.ts):
        agg.add(p)
    return pd.DataFrame(agg.flush())
