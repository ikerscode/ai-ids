"""MITRE ATT&CK mapping and kill-chain stages.

Known families map directly. Unknown anomalies are mapped from what the
explanation says is unusual (fan-out -> discovery, big uploads -> exfiltration,
regular repetition -> C2 ...), and the rationale says which evidence was used.
"""
from __future__ import annotations

from .schema import is_internal

# Kill-chain stages in order; incident severity rises as an incident walks further along.
STAGES = ["recon", "initial_access", "execution_persistence", "lateral_movement",
          "command_and_control", "exfil_impact"]
STAGE_LABELS = {
    "recon": "Recon", "initial_access": "Initial access", "execution_persistence": "Execution / persistence",
    "lateral_movement": "Lateral movement", "command_and_control": "C2", "exfil_impact": "Exfil / impact",
}

TECHNIQUES = {
    "T1046": ("Network Service Discovery", "recon"),
    "T1595": ("Active Scanning", "recon"),
    "T1110": ("Brute Force", "initial_access"),
    "T1110.001": ("Brute Force: Password Guessing", "initial_access"),
    "T1190": ("Exploit Public-Facing Application", "initial_access"),
    "T1078": ("Valid Accounts", "initial_access"),
    "T1059": ("Command and Scripting Interpreter", "execution_persistence"),
    "T1543.003": ("Create or Modify System Process: Windows Service", "execution_persistence"),
    "T1021.002": ("Remote Services: SMB/Windows Admin Shares", "lateral_movement"),
    "T1021": ("Remote Services", "lateral_movement"),
    "T1071.001": ("Application Layer Protocol: Web Protocols", "command_and_control"),
    "T1071.004": ("Application Layer Protocol: DNS", "command_and_control"),
    "T1048": ("Exfiltration Over Alternative Protocol", "exfil_impact"),
    "T1048.003": ("Exfiltration Over Unencrypted Non-C2 Protocol", "exfil_impact"),
    "T1030": ("Data Transfer Size Limits", "exfil_impact"),
    "T1041": ("Exfiltration Over C2 Channel", "exfil_impact"),
    "T1498.001": ("Network Denial of Service: Direct Network Flood", "exfil_impact"),
    "T0846": ("Remote System Discovery (ICS)", "recon"),
    "T0855": ("Unauthorized Command Message (ICS)", "exfil_impact"),
    "T1204": ("User Execution", "execution_persistence"),
}

FAMILY_MAP = {
    "portscan": ("T1046", "Many destination ports probed in a short window with SYN/RST-only exchanges."),
    "ssh_bruteforce": ("T1110.001", "Repeated short SSH sessions from one source, consistent with password guessing."),
    "web_sqli": ("T1190", "Bursts of oversized HTTP requests to a public web server followed by large database replies."),
    "dns_tunnel": ("T1071.004", "High-rate DNS queries with unusually large payloads to a single resolver."),
    "c2_beacon": ("T1071.001", "Small HTTPS sessions to one external host repeating on a regular, jittered period."),
    "smb_lateral": ("T1021.002", "SMB/RPC sessions pushing data from a host that does not normally use SMB to those peers."),
    "syn_flood": ("T1498.001", "Very high rate of half-open TCP connections from many sources to one service."),
    "slow_exfil": ("T1030", "Repeated large outbound uploads in small chunks to an external host, spaced out over hours."),
    "bruteforce": ("T1110", "Repeated authentication attempts."),
    "dos": ("T1498.001", "Traffic volume consistent with denial of service."),
    "web_attack": ("T1190", "Malicious requests to a public web application."),
    "infiltration": ("T1204", "Infiltration activity labelled in the dataset."),
    "botnet": ("T1071.001", "Bot command-and-control traffic labelled in the dataset."),
    # Host-log detections
    "auth_bruteforce": ("T1110", "Burst of failed logons on one host from one source."),
    "auth_after_failures": ("T1078", "Successful logon from a source that just failed many times."),
    "new_logon_source": ("T1021", "Account logged on from a source never seen for it during the baseline."),
    "service_install": ("T1543.003", "New service installed outside the patch window or by an unusual account."),
}


def technique(tid: str) -> dict:
    name, stage = TECHNIQUES.get(tid, ("Unmapped", "recon"))
    return {"technique_id": tid, "technique": name, "stage": stage}


def map_family(family: str, external_dst: bool = False) -> dict:
    tid, why = FAMILY_MAP.get(family, ("", ""))
    if not tid:
        return {"technique_id": "", "technique": "Unmapped", "stage": "recon",
                "rationale": f"No mapping for class '{family}'."}
    out = technique(tid)
    if family == "new_logon_source" and external_dst:
        out = technique("T1078")
    out["rationale"] = why
    return out


def map_unknown(flow: dict, top_features: list[str]) -> dict:
    """Heuristic mapping for an unknown anomaly from its flow and strongest features."""
    feats = set(top_features)
    port = int(flow.get("dst_port", 0))
    internal_dst = is_internal(flow.get("dst_ip", ""))
    fanout = {"ctx_src_dports_60s", "ctx_src_dsts_60s", "win_pair_dports_300s"} | {
        f"win_src_{k}_{w}" for k in ("dports", "dsts", "fail") for w in ("10s", "60s", "300s")}
    repeats = {"ctx_src_flows_60s", "ctx_pair_flows_1h", "win_pair_flows_60s", "win_pair_flows_300s",
               "win_src_samesize_60s"} | {f"win_src_flows_{w}" for w in ("10s", "60s", "300s")}
    rules = [
        (fanout, "T1046", "fan-out to many ports/hosts or many failed probes in a short window"),
        ({"ctx_pair_iat_cv", "ctx_pair_flows_1h", "win_pair_gap_cv_300s"}, "T1071.001",
         "connections to the same peer repeat on a regular period"),
        ({"ctx_dst_flows_60s", "syn_cnt", "pkts_per_s", "win_dst_flows_60s", "win_dst_srcs_60s"}, "T1498.001",
         "unusual connection rate towards one destination"),
    ]
    if port == 53 and feats & {"fwd_len_mean", "fwd_bytes", "fwd_len_max", "ctx_src_flows_60s"}:
        out = technique("T1071.004")
        out["rationale"] = "DNS traffic with abnormal query size or rate (possible tunnelling)."
        return out
    if port in (445, 135, 139) and internal_dst:
        out = technique("T1021.002")
        out["rationale"] = "SMB/RPC to an internal peer this host does not normally use."
        return out
    if port in (21, 22, 23, 3389, 5985) and feats & ({"ctx_src_flows_60s", "ctx_pair_flows_1h"} | repeats):
        out = technique("T1110")
        out["rationale"] = "Repeated login sessions (FTP, SSH, Telnet, RDP, WinRM) from one source."
        return out
    for keys, tid, why in rules:
        if feats & keys:
            out = technique(tid)
            out["rationale"] = f"Mapped from evidence: {why}."
            return out
    if feats & {"fwd_bytes", "fwd_pkts", "down_up_ratio", "peer"} and not internal_dst:
        out = technique("T1048")
        out["rationale"] = "Outbound volume or destination is unusual for this host (possible exfiltration)."
        return out
    if feats & {"port", "service", "peer"}:
        out = technique("T1046" if internal_dst else "T1071.001")
        out["rationale"] = "First-time port/peer for this host; weak evidence, mapped by destination type."
        return out
    return {"technique_id": "", "technique": "Unmapped", "stage": "recon",
            "rationale": "Statistically unusual, but no feature pattern points at a specific technique."}
