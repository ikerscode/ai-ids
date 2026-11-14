"""pcap -> flows. Two packet readers feed the same FlowAggregator.

* tshark field export: fast, used automatically when tshark is installed
* scapy PcapReader: pure Python fallback, always available

Only IPv4/IPv6 TCP, UDP and ICMP packets are used; everything else is skipped.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import pandas as pd

from .flows import FlowAggregator, Packet
from .schema import normalise

TSHARK_FIELDS = ["frame.time_epoch", "ip.src", "ip.dst", "ipv6.src", "ipv6.dst", "ip.proto",
                 "ipv6.nxt", "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport",
                 "frame.len", "tcp.flags"]


def iter_packets_scapy(path: Path) -> Iterator[Packet]:
    from scapy.layers.inet import ICMP, IP, TCP, UDP
    from scapy.layers.inet6 import IPv6
    from scapy.utils import PcapReader

    with PcapReader(str(path)) as reader:
        for pkt in reader:
            if IP in pkt:
                ip, proto = pkt[IP], int(pkt[IP].proto)
            elif IPv6 in pkt:
                ip, proto = pkt[IPv6], int(pkt[IPv6].nh)
            else:
                continue
            sport = dport = flags = 0
            if TCP in pkt:
                sport, dport, flags = int(pkt[TCP].sport), int(pkt[TCP].dport), int(pkt[TCP].flags)
            elif UDP in pkt:
                sport, dport = int(pkt[UDP].sport), int(pkt[UDP].dport)
            elif ICMP not in pkt and proto not in (1, 58):
                continue
            yield Packet(float(pkt.time), str(ip.src), str(ip.dst), sport, dport, proto, len(pkt), flags)


def iter_packets_tshark(path: Path) -> Iterator[Packet]:
    cmd = ["tshark", "-n", "-r", str(path), "-T", "fields", "-E", "separator=|",
           "-E", "occurrence=f"]
    for f in TSHARK_FIELDS:
        cmd += ["-e", f]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        parts = line.rstrip("\n").split("|")
        if len(parts) < len(TSHARK_FIELDS):
            continue
        t, s4, d4, s6, d6, p4, p6, tsp, tdp, usp, udp_, flen, tflags = parts[:13]
        src, dst = (s4, d4) if s4 else (s6, d6)
        if not src:
            continue
        proto = int(p4 or p6 or 0)
        if proto not in (1, 6, 17, 58):
            continue
        sport = int(tsp or usp or 0)
        dport = int(tdp or udp_ or 0)
        flags = int(tflags, 16) if tflags else 0
        yield Packet(float(t), src, dst, sport, dport, proto, int(flen or 0), flags)
    proc.wait()


def pcap_to_flows(path: str | Path, engine: str = "auto", sensor: str = "pcap",
                  idle_timeout: float = 120.0, active_timeout: float = 1800.0) -> pd.DataFrame:
    """Read a pcap/pcapng file and return normalised flows (schema.ALL_COLUMNS)."""
    path = Path(path)
    if engine == "auto":
        engine = "tshark" if shutil.which("tshark") else "scapy"
    reader = iter_packets_tshark if engine == "tshark" else iter_packets_scapy
    agg = FlowAggregator(idle_timeout, active_timeout, sensor)
    for p in reader(path):
        agg.add(p)
    raw = pd.DataFrame(agg.flush())
    if raw.empty:
        return normalise(pd.DataFrame(columns=["ts"]))
    raw["label"] = "unlabelled"
    return normalise(raw)
