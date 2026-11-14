"""pcap -> flow aggregation on a small capture written with scapy."""
from scapy.layers.inet import IP, TCP, UDP
from scapy.packet import Raw
from scapy.utils import wrpcap

from aiids.pcap import pcap_to_flows
from aiids.schema import ALL_COLUMNS, FEATURES, validate


def _pkt(t, p):
    p.time = t
    return p


def make_pcap(path):
    c, s = "10.0.0.5", "10.0.0.9"
    pk = [
        # TCP session 1: handshake, request, response, FIN teardown
        _pkt(100.0, IP(src=c, dst=s) / TCP(sport=40000, dport=80, flags="S")),
        _pkt(100.01, IP(src=s, dst=c) / TCP(sport=80, dport=40000, flags="SA")),
        _pkt(100.02, IP(src=c, dst=s) / TCP(sport=40000, dport=80, flags="A")),
        _pkt(100.03, IP(src=c, dst=s) / TCP(sport=40000, dport=80, flags="PA") / Raw(b"G" * 200)),
        _pkt(100.05, IP(src=s, dst=c) / TCP(sport=80, dport=40000, flags="PA") / Raw(b"R" * 1000)),
        _pkt(100.06, IP(src=c, dst=s) / TCP(sport=40000, dport=80, flags="FA")),
        _pkt(100.07, IP(src=s, dst=c) / TCP(sport=80, dport=40000, flags="FA")),
        _pkt(100.08, IP(src=c, dst=s) / TCP(sport=40000, dport=80, flags="A")),
        # Same 5-tuple reused after close: must become a second flow
        _pkt(101.0, IP(src=c, dst=s) / TCP(sport=40000, dport=80, flags="S")),
        _pkt(101.01, IP(src=s, dst=c) / TCP(sport=80, dport=40000, flags="RA")),
        # UDP DNS query/response
        _pkt(102.0, IP(src=c, dst="10.0.0.53") / UDP(sport=5353, dport=53) / Raw(b"q" * 40)),
        _pkt(102.01, IP(src="10.0.0.53", dst=c) / UDP(sport=53, dport=5353) / Raw(b"a" * 120)),
        # UDP flow split by the idle timeout
        _pkt(103.0, IP(src=c, dst="10.0.0.7") / UDP(sport=6000, dport=123)),
        _pkt(400.0, IP(src=c, dst="10.0.0.7") / UDP(sport=6000, dport=123)),
    ]
    wrpcap(str(path), pk)


def test_aggregation(tmp_path):
    f = tmp_path / "t.pcap"
    make_pcap(f)
    flows = pcap_to_flows(f, engine="scapy", idle_timeout=60)
    assert list(flows.columns) == ALL_COLUMNS
    assert validate(flows) == []
    assert len(flows) == 5
    web = flows[(flows.dst_port == 80)].sort_values("ts")
    first, second = web.iloc[0], web.iloc[1]
    assert first.src_ip == "10.0.0.5" and first.fwd_pkts == 5 and first.bwd_pkts == 3
    assert first.syn_cnt == 2 and first.fin_cnt == 2
    assert first.bwd_bytes > first.fwd_bytes  # the response is bigger
    assert abs(first.duration - 0.08) < 1e-6
    assert second.rst_cnt == 1 and second.fwd_pkts == 1
    dns = flows[flows.dst_port == 53].iloc[0]
    assert dns.proto == 17 and dns.fwd_pkts == 1 and dns.bwd_pkts == 1
    assert (flows.dst_port == 123).sum() == 2
    # context: the second web flow sees the first one from the same source in the window
    assert second.ctx_src_flows_60s >= 1


def test_tshark_matches_scapy(tmp_path):
    import shutil
    import pytest
    if not shutil.which("tshark"):
        pytest.skip("tshark not installed")
    f = tmp_path / "t.pcap"
    make_pcap(f)
    a = pcap_to_flows(f, engine="scapy", idle_timeout=60).sort_values(["ts", "dst_port"]).reset_index(drop=True)
    b = pcap_to_flows(f, engine="tshark", idle_timeout=60).sort_values(["ts", "dst_port"]).reset_index(drop=True)
    assert len(a) == len(b)
    assert (a[FEATURES].round(4) == b[FEATURES].round(4)).all().all()
