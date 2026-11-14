# ai-ids

A machine-learning intrusion detection system for the seclab. It turns network traffic into flows, scores
them with three unsupervised novelty detectors plus a supervised classifier, explains every alert in
plain sentences, cuts false positives in measured steps, and correlates alerts from network sensors, host
logs and other lab tools into incidents that follow the kill chain.

The default data is **simulated**. The built-in generator models a small enterprise (workstations, web
server, DB, DNS, domain controller, file and backup servers, an OT segment) with four network sensors and
host/auth logs, and plays eight attack families through it with ground-truth labels. Every number in this
README comes from that simulated data. How to run the same evaluation on CIC-IDS2017 is described below.

HUD: http://127.0.0.1:8105

## Quick start

```bash
./run.sh                                   # creates .venv on first run, serves the HUD on 127.0.0.1:8105
```

Then in the HUD: Models & data -> Generate simulated dataset -> Train -> Evaluate, then Live alerts ->
Start stream (or Batch-detect split).

The same from the command line:

```bash
.venv/bin/python -m aiids simulate                 # 4 days, ~60k flows, ~1k host events (about 15 s)
.venv/bin/python -m aiids train                    # all models + calibration (about 10 s)
.venv/bin/python -m aiids evaluate                 # ablation + leave-one-family-out (about 50 s)
.venv/bin/python -m aiids detect                   # batch-detect the test split into the alert store
.venv/bin/python -m aiids stream --speed 3000 --seconds 30
.venv/bin/python -m aiids ingest-pcap capture.pcapng
.venv/bin/python -m aiids ingest-csv Wednesday-workingHours.pcap_ISCX.csv
.venv/bin/python -m pytest -q                      # 17 tests, about 10 s
```

## Architecture

```
  pcap/pcapng ──(tshark fields or scapy)──> FlowAggregator ──┐
  simulator (packets per flow, 4 sensors) ───────────────────┤   one flow schema:
  CIC-IDS2017 / UNSW-NB15 CSV loaders ───────────────────────┘   entity cols + 30 flow features
                                                                 + 6 causal context features
                                                                          │
                ┌─────────────────────────────────────────────────────────┤
                │ benign only                                             │ labelled
   ┌────────────┴───────────┬────────────────────────┐          ┌─────────┴──────────┐
   │ IsolationForest        │ autoencoder (MLP       │          │ HistGradientBoosting│
   │                        │ 24-8-24 bottleneck)    │          │ known families      │
   │ per-host baselines: ports, peers, hours,        │          └─────────┬──────────┘
   │ volumes (Good-Turing surprisal, MAD z-scores)   │                    │
   └────────────┬───────────┴────────────┬───────────┘                    │
          quantile normalisation -> ensemble + votes                      │
                │                                                         │
                └──────────── engine: thresholds, agreement, per-host ────┘
                              thresholds, open-set rule, suppression
                                          │
                     explanations (AE error, baseline z, perturbation) + ATT&CK
                                          │
   host/auth logs -> rule/baseline host detector ─┐
   POST /api/ingest (honeypot-net, ics-sim, ...) ─┼─> alert dedup/aggregation -> SQLite
                                                  │
                                       correlation -> incidents (entity graph, kill chain)
                                                  │
                                    FastAPI + static HUD, CLI, streaming replay
```

| module | job |
|---|---|
| `schema.py` | flow schema, feature maths, causal context features |
| `flows.py`, `pcap.py` | flow aggregator (idle 120 s, active 1800 s, FIN/RST teardown), tshark and scapy readers |
| `simulator.py` | enterprise network, benign patterns, attack campaigns, host log events |
| `datasets.py` | dataset storage, CIC-IDS2017 and UNSW-NB15 loaders |
| `features.py`, `detectors.py`, `supervised.py`, `models.py` | transforms, the four detectors, training and calibration |
| `engine.py` | per-flow decisions and every FP-reduction stage |
| `explain.py`, `attack.py` | explanations and MITRE ATT&CK mapping |
| `alerts.py`, `hostdetect.py`, `correlate.py`, `ingest.py` | aggregation, host-log detector, incidents, external alerts |
| `evaluate.py`, `pipeline.py`, `stream.py`, `api.py`, `__main__.py` | evaluation, orchestration, replay, HTTP API, CLI |

## Data sources

**Flow schema.** Entity columns `ts, end_ts, src_ip, dst_ip, src_port, dst_port, src_host, dst_host, sensor`,
labels `label, campaign, split`, and 36 features: duration, fwd/bwd packets and bytes, packet size
mean/std/max per direction, inter-arrival mean/std/min/max (overall and per direction), SYN/FIN/RST/PSH/ACK/URG
counts, bytes/s, packets/s, down/up ratio, destination port class (web, dns, smb_rpc, remote_admin,
database, directory, mail, ot, infra, other, registered, ephemeral), protocol, and six context features
computed only from earlier flows: flows, distinct ports and distinct hosts from this source in 60 s, flows
to this destination in 60 s, repeat connections to the same peer in 1 h and the regularity (coefficient of
variation) of those repeats. The context features are what make scans, floods and beacons visible to a
per-flow model.

**pcap.** `pcap.py` reads packets with a tshark field export when tshark exists (fast) or scapy otherwise,
and feeds the same `FlowAggregator`. A test checks both readers give identical features.

**Simulator.** 25 hosts on four segments, each with its own sensor (`sensor-corp`, `sensor-dmz`,
`sensor-srv`, `sensor-ot`), and host logs per host (`hostlog-<host>`). Benign traffic: web browsing, DNS
with upstream forwarding, SMB to the file server, Kerberos/LDAP, NTP, admin SSH and RDP, nightly backups and
patch downloads, public web traffic into the DMZ with DB queries behind it, Modbus polling in the OT
segment, all with a diurnal profile. Benign oddities that cause realistic false positives: an IT inventory
sweep that starts on day 3, a software rollout from a never-seen server on day 4, evening cloud syncs,
stray connections to closed ports. Attack campaigns each day, with labels and timestamps:

* A, intrusion chain: port scan of web01 -> SQL injection bursts (with DB dumps) -> service install on
  web01 -> C2 beaconing with jitter -> SMB lateral movement with new-source logons -> DNS tunnelling exfil
* B, credentials: SSH brute force (failed logons in the auth log) -> successful logon -> pivot to db01 ->
  slow-and-low HTTPS exfil
* C, SYN flood against web01 from spoofed sources
* D (every other day), a compromised workstation sweeps the server segment and moves over SMB

Day 1 = train, day 2 = val (threshold calibration), day 3 = feedback (the analyst reviews alerts), day 4 =
test. From day 4 the attackers are stealthier (slow scan, 30 to 50% beacon jitter, smaller DNS payloads,
smaller exfil chunks) so the test day is not a copy of the training day.

**CIC-IDS2017 / UNSW-NB15.** `ingest-csv` detects the format, maps the columns onto the schema (CIC
durations and IATs are microseconds, UNSW inter-packet times are milliseconds), maps labels to families
(PortScan -> portscan, SSH-Patator -> ssh_bruteforce, Web Attack Sql Injection -> web_sqli, DoS/DDoS ->
dos, Bot -> botnet, and so on), downsamples to `--max-rows` and assigns a stratified random
train/val/feedback/test split. Features a dataset lacks are zero and listed in the dataset metadata.

## Detection layers

* **IsolationForest** (150 trees, 512 samples) on standardised log features.
* **Autoencoder**: `MLPRegressor` 36-24-8-24-36 (tanh) trained to reproduce benign flows. The score is
  the mean of log(1 + per-feature error / that feature's median benign error), so features that are always
  hard to rebuild do not dominate.
* **Per-entity baselines**: one profile per internal host as a client and per internal server for inbound
  traffic. Each profile keeps port, peer and peer+port counters, an hour-of-day histogram, and median/MAD of
  log volumes and context rates. New ports or peers are scored with Good-Turing surprisal: the probability
  of something unseen is (things seen once) / (flows). A workstation that visits new websites all day gets a
  high unseen mass, so novelty is cheap; db01, which only ever talks to three peers, gets almost none, so a
  new external peer is very surprising.
* **Ensemble**: each detector's raw score becomes "benign tail surprise", -log10 of the share of benign
  training scores above it (0 typical, 2 beyond the 99th percentile). The ensemble is the mean.
* **Supervised**: `HistGradientBoostingClassifier` (150 iterations, class weights capped at 50, at most two
  threads) over the known families.
* **Open-set rule**: if the classifier is confident (p(attack) above its calibrated threshold) the alert
  gets its family. If the anomaly layer fires and the classifier is unsure (best family probability below
  0.5) the alert is `unknown_anomaly`. The unknown class is then mapped to ATT&CK from its evidence.

### Unseen attacks: leave-one-family-out

Each row retrains the classifier without one family; the anomaly layer is untouched because it never sees
attacks. Test day, simulated data, after all FP-reduction stages:

| held out | test flows | recall when trained on it | recall never seen | flagged by anomaly layer | labelled unknown_anomaly | attack steps caught | benign FPR |
|---|---|---|---|---|---|---|---|
| c2_beacon | 200 | 100.0% | 100.0% | 100.0% | 70.0% | 1/1 | 0.0024 |
| dns_tunnel | 737 | 97.6% | 16.4% | 16.4% | 16.4% | 1/1 | 0.0018 |
| portscan | 632 | 99.2% | 99.4% | 99.2% | 96.4% | 2/2 | 0.0024 |
| slow_exfil | 25 | 100.0% | 100.0% | 100.0% | 4.0% | 1/1 | 0.0021 |
| smb_lateral | 52 | 100.0% | 98.1% | 98.1% | 46.2% | 2/2 | 0.0025 |
| ssh_bruteforce | 169 | 100.0% | 100.0% | 100.0% | 97.0% | 1/1 | 0.0020 |
| syn_flood | 1006 | 100.0% | 100.0% | 100.0% | 50.3% | 1/1 | 0.0024 |
| web_sqli | 272 | 99.6% | 81.6% | 81.6% | 53.3% | 1/1 | 0.0023 |

Every held-out family is still caught as a campaign step. The weak spot is the stealthy DNS tunnel: each
query is a single small UDP flow, so only 16% of its flows stand out without a trained classifier (the
flows that do alert are enough for the incident). Where "labelled unknown_anomaly" is lower than the
recall, the classifier recognised the flows as some other attack family (slow exfil looks like other bulk
uploads, a SYN flood looks partly like a scan); they still alert, just under a sibling class.

## False-positive reduction

Stages are cumulative and each is measured on the test day (14,995 flows, 3,093 of them attack, 24 h):

| stage | precision | recall | FPR | PR-AUC | alerts/h | false alerts/h | attack steps caught |
|---|---|---|---|---|---|---|---|
| no FP reduction (training 95th percentile, argmax) | 0.803 | 0.998 | 0.0636 | 0.977 | 160.25 | 31.55 | 100% |
| + thresholds calibrated on benign val | 0.966 | 0.994 | 0.0090 | 0.998 | 132.57 | 4.46 | 100% |
| + ensemble agreement (2 of 3) | 0.968 | 0.992 | 0.0086 | 0.998 | 132.12 | 4.25 | 100% |
| + per-host adaptive baselines/thresholds | 0.971 | 0.992 | 0.0078 | 0.998 | 131.78 | 3.88 | 100% |
| + analyst feedback (rules + retrain) | 0.993 | 0.992 | 0.0018 | 0.999 | 128.78 | 0.88 | 100% |
| + alert dedup/aggregation | 0.993 | 0.992 | 0.0018 | 0.999 | **2.38** | **0.79** | 100% |

Rows 1 to 5 count alerted flows; the last row counts the deduplicated alerts an analyst would actually see.
Measured the same way (aggregated alerts) the stages give 18.3, 2.75, 2.5, 2.29, 2.38 alerts per hour with
alert-level precision 0.08, 0.50, 0.53, 0.58, 0.67. False positives went from 757 flows to 21.

How each stage works:

1. **Calibration**: thresholds are chosen on benign validation traffic (day 2) for target FPRs of 0.3%
   (anomaly ensemble) and 0.2% (classifier, never below 0.5).
2. **Ensemble agreement**: each detector votes when it is past its own 1% benign tail; an anomaly alert needs
   two votes. A single detector tripping on one weird feature is not enough.
3. **Per-host adaptive baselines**: hosts whose benign traffic is naturally odd (the admin workstation, the
   database during backups, the HMI) get a higher threshold, set from their own 99.5th percentile on val.
   Baselines also keep learning: flows that did not alert are folded into the host's port/peer/hour counters
   chunk by chunk, so a new benign pattern stops alerting after its first appearance. This is a poisoning
   risk as well (see limitations).
4. **Analyst feedback**: in the HUD, marking an alert as a false positive stores the verdict, stores its
   flows as extra benign data for the next retrain, and can create a suppression rule (exact
   src/dst/port/class, or class from this source). In the evaluation an oracle analyst reviews the feedback
   day: false positives are grouped by (source, class), the biggest groups become rules and every false
   positive flow is added to the benign set before retraining. On this run it learned "portscan from adm01"
   (the inventory sweep) and one unknown-anomaly rule.
5. **Aggregation**: hits with the same key (class + source for scans and DNS tunnels, class + target for
   floods, class + source + destination otherwise) merge while they are less than 5 minutes apart (30
   minutes for beacons, slow exfil and unknown anomalies).

## Explanations

Every alert carries the top six features behind it, a bar chart and sentences such as:

* "slow exfil from db01 to 192.0.2.43:443. flow duration (s) 1204x above this host's median (22.1 vs
  0.018); download/upload ratio 955x below the benign web median (0.012 vs 11.1); packets sent 130x above
  this host's median"
* "c2 beacon from web01 to 198.51.100.26:443. 198.51.100.26:443 is a new peer for web01; port 443 never
  seen for web01 during the baseline"
* "'deploy' logged on to db01 from 10.10.2.10, a source never seen for this account"

Four evidence sources are each normalised to sum to 1 and weighted by how strongly that detector fired:

* autoencoder per-feature reconstruction error,
* baseline components (port/peer/service/hour surprisal, robust z-scores of volumes and rates),
* supervised local perturbation: replace one feature with its benign median and measure the drop in
  p(attack), done for all features in one batched call,
* the same perturbation applied to the IsolationForest score.

Features whose value is ordinary for that kind of service (compared with the benign median of the same port
class, so DNS is compared with DNS) are pushed down so the text names the evidence that actually differs.
Sentences compare against the host's own median when it has a profile, otherwise the service median.

**ATT&CK mapping**: known families map directly (portscan T1046, SSH brute force T1110.001, SQLi T1190,
DNS tunnel T1071.004, beacon T1071.001, SMB lateral T1021.002, SYN flood T1498.001, slow exfil T1030;
host-log rules map to T1110, T1078, T1021, T1543.003) with a stated rationale. Unknown anomalies are mapped
from their top features (fan-out -> T1046, regular repeats -> T1071.001, DNS with odd sizes -> T1071.004,
internal SMB -> T1021.002, unusual outbound volume -> T1048) and the rationale says which evidence was used
and when it is weak.

## Cross-system correlation

Alerts from network sensors, host logs, pcap uploads and external tools share one record
(`source, sensor, src/dst ip and host, user, class, technique, stage, confidence, severity`). IPs are
resolved to host names through the asset inventory, so a honeypot alert that only knows an IP joins a host
log alert that only knows a host name.

Incidents are built with union-find over time-ordered alerts. Each alert has actor entities (source
address/host, account) and target entities. A new alert joins an incident when:

* its actor was an actor in the incident in the last 2 hours (same attacker, account or compromised host),
* its actor was the **target of an earlier compromising step** (initial access, persistence, lateral
  movement) in the last 2 hours. This is the pivot pattern: web01 is exploited, then web01 attacks db01.
  Being scanned or flooded does not make a host part of the chain,
* it only shares a target with the incident, and only within 2 minutes (a busy DNS or web server would
  otherwise glue unrelated activity together).

Severity points: 1.2 per extra kill-chain stage, 0.8 per extra reporting system (up to 4), 2 x the
highest confidence, +1 when the stages appear in kill-chain order (recon -> initial access ->
execution/persistence -> lateral movement -> C2 -> exfil/impact) with at least three of them, +0.5 if any
alert is critical. On the test day the intrusion chain becomes one critical incident with 30 alerts, all 6
stages and 9 systems (3 network sensors and 6 host logs); campaign B becomes a separate critical incident
(initial access -> lateral movement -> exfil, 4 systems). Posting a honeypot-net alert about the campaign A
attacker joins that incident (verified through the API).

### External ingest

`POST /api/ingest` accepts one object or `{"alerts": [...]}` (up to 500). Full JSON schema at
`GET /api/ingest/schema`. Unknown top-level fields are rejected; tool-specific data goes in `details`.

```text
{
  "source": "honeypot-net",            // required: [a-z0-9_.-]{1,64}
  "category": "ssh_bruteforce",        // required: what happened
  "ts": "2026-10-08T11:05:00Z",        // epoch seconds or ISO-8601, default now
  "src_ip": "203.0.113.13", "dst_ip": "10.10.2.10", "dst_port": 22,
  "src_host": null, "dst_host": null, "user": "root",   // at least one entity field required
  "severity": "high",                  // low | medium | high | critical
  "confidence": 0.9,                   // 0..1, default from severity
  "technique_id": "T1110",             // optional ATT&CK id; otherwise mapped from category
  "stage": null,                       // optional kill-chain stage override
  "summary": "cowrie: 212 login attempts in 4 minutes",
  "details": {"sensor": "ssh-2222"}
}
```

The response lists the incidents the new alerts joined. honeypot-net (8103), ics-sim (8104) and malware-lab
(8101) can push here; nothing in those projects needs to change beyond an HTTP POST.

## Streaming

`Streamer` replays a dataset split in accelerated time (default 600 simulated seconds per wall second):
every 0.5 s it feeds the flows and host events of that slice through the engine (with its adaptive
baseline), merges hits into open alert groups and writes new or updated alerts to SQLite. Start, stop and
speed are in the Live tab and at `/api/stream/*`; `python -m aiids stream` prints alerts as they arrive.

## HUD tour

* **Live alerts**: stream controls (dataset, split, speed, start/stop, batch-detect, clear), a flows vs
  alerted flows rate chart, a filterable alert table. Clicking an alert opens the explanation: summary,
  contribution bars, evidence table with per-source weights, ATT&CK technique and rationale, detector
  scores, raw flow or event, and TP/FP buttons with an optional suppression rule.
* **Incidents**: correlated incidents ranked by score, the timeline and entity graph of the selected
  incident, and a box to paste an external alert ("Use top incident" fills in that incident's attacker IP).
* **Models & data**: generate a simulated dataset, upload a pcap or CSV, switch the active dataset; train
  (uses analyst feedback) and evaluate; the ablation table, the leave-one-family-out table, per-class
  results, confusion matrix, thresholds and suppression rules.

## Terminal HUD

`python -m aiids tui` draws a terminal dashboard over the running API (start `./run.sh` first). It shows
alerts, incidents, model metrics and status, and it has the same stream, generate, detect, train and
evaluate controls as the web HUD. Use `--url` to point it at another address.

Keys: `1`-`9` or `Tab` switch views, arrows move the selection, `Enter` opens the detail (alert
explanation or incident), `Esc` goes back, `s` start stream, `x` stop stream, `g` generate dataset,
`d` batch detect, `t` train and `e` evaluate (`g`, `t` and `e` ask `y/N` first), `q` quits.

Over SSH or in a script, `python -m aiids tui --once` prints every view once as plain text and exits.

## API

`GET /api/health`, `/api/status`, `/api/datasets`, `/api/datasets/{name}`, `/api/models`, `/api/alerts`,
`/api/alerts/{id}`, `/api/rules`, `/api/incidents`, `/api/incidents/{id}`, `/api/ingest/schema`,
`/api/stream`, `/api/jobs/{id}`; `POST /api/simulate`, `/api/train`, `/api/evaluate`, `/api/detect`,
`/api/upload?kind=pcap|csv&filename=...` (raw body), `/api/datasets/active`, `/api/alerts/{id}/feedback`,
`/api/alerts/clear`, `/api/ingest`, `/api/stream/start|stop|speed`; `DELETE /api/rules/{id}`.

## Evaluating on CIC-IDS2017

1. Download the CSVs (the "TrafficLabelling" set keeps IPs and timestamps, which the context features and
   per-host baselines need; "MachineLearningCVE" works too but those features become zero).
2. `python -m aiids ingest-csv Wednesday-workingHours.pcap_ISCX.csv --name cic-wed` (repeat per day, or
   concatenate days first). Large files are downsampled to `--max-rows` (default 200k: 70% benign, attacks
   capped per class).
3. `python -m aiids train --dataset cic-wed` then `python -m aiids evaluate --dataset cic-wed`.

### Results on CIC-IDS2017 (MachineLearningCVE, all eight days)

Sample: up to 40k benign and 20k per attack label from each day file, then the loader's own cap (300k rows:
210k benign, attacks capped per family), giving 248,028 flows. Random stratified split, test = 62,010 flows.
These CSVs have no IPs or timestamps, so per-host baselines, context features and correlation are inactive.

| Stage | Precision | Recall | FPR | PR-AUC | False alerts/h |
|---|---|---|---|---|---|
| naive | 0.709 | 0.994 | 0.0741 | 0.955 | 3336 |
| calibrated | 0.974 | 0.955 | 0.0046 | 0.945 | 205 |
| + agreement | 0.976 | 0.955 | 0.0043 | 0.947 | 194 |
| + analyst feedback | 0.977 | 0.952 | 0.0041 | 0.949 | 186 |
| + aggregation (alert level) | | | | | 12.9 |

Known attacks are caught well once thresholds are calibrated. Unseen attacks are not: with one family held
out of the classifier, the anomaly layer flagged 2.2% of DoS, 0.1% of port scan, 0% of brute force and 0%
of botnet flows (heartbleed 3/3 and infiltration 2/9 were caught, but those counts are tiny). On this data
the benign-trained detectors cannot separate short scan and login flows from normal short flows, which
matches what published work on CIC-IDS2017 reports. The simulator's leave-one-family-out table above is
therefore far more optimistic than real traffic.

This run also exposed a bug, now fixed: with no IPs every flow has source 0.0.0.0, so a feedback rule
"suppress class X from 0.0.0.0" muted that class for everyone and recall fell to 0.36. Feedback no longer
creates rules for unknown sources (evaluation and the HUD button), and a test covers it.

UNSW-NB15 works the same way with either the 49-column raw files or the training/testing-set CSVs. For
public datasets the split is random and stratified, so it is easier than the simulator's day-based split
(flows from the same attack sit on both sides), and the host-log and correlation parts have nothing to work
with because those datasets have no host logs.

## Verification done for this version

* simulate -> train -> evaluate from the CLI, numbers above.
* A real loopback capture (217 packets of my own traffic: HTTP requests to a local test server, UDP to
  unused ports, local service polling) recorded with dumpcap and run through `ingest-pcap` (tshark reader):
  24 flows, 3 alerts. A scapy-written pcap through the scapy reader: 5 flows. Every loopback flow alerts,
  as expected for a model that has only ever seen the simulated enterprise.
* Streaming from the CLI and the API, HUD screens checked in a browser, all endpoints above called with curl.
* External honeypot-net alert about the campaign A attacker joined the intrusion incident.
* `pytest`: 17 tests (flow aggregation from a scapy pcap and tshark/scapy parity, schema and CSV loaders,
  training, stages, open-set detection of a held-out family, suppression, explanation shape and sanity,
  correlation of a multi-stage campaign into one incident, ingest validation and correlation via HTTP).

## Safety

* Binds to 127.0.0.1 only. No traffic is ever sent anywhere; attacks exist only as synthetic flow records.
  Attacker and C2 addresses come from the documentation ranges 192.0.2.0/24, 198.51.100.0/24 and
  203.0.113.0/24.
* Uploads are stored under `data/uploads/` with sanitised names, capped at 300 MB, and parsed by tshark or
  scapy, never executed.
* External ingest is validated strictly (types, IPs, ATT&CK id format, size limits, unknown fields rejected)
  and all API data is escaped in the HUD.

## Limitations

* **The headline numbers are from simulated traffic.** The simulator was written together with the
  detector, so the families are cleaner and more separable than real attacks; expect much lower numbers on
  real or public data. The leave-one-family-out table is the more meaningful result.
* Small test campaign counts (1 or 2 instances per family on the test day), so step-level figures are coarse.
* Models trained on the simulated enterprise flag almost everything in an unrelated capture (see the
  loopback run). On a real network you would collect benign traffic first and train on that.
* Adaptive baselines learn from flows that did not alert, so slow, undetected attacker activity can be
  absorbed into the baseline (poisoning). Suppression rules can hide a later real attack from the same
  source; the rule table shows hit counts so this can be reviewed.
* Correlation is entity and time based. A shared privileged account (`it.admin`) links an insider campaign
  with benign admin activity in the test day; campaigns that share no entities are never linked.
* Unknown-anomaly ATT&CK mapping is heuristic and sometimes weak (the rationale says so).
* Beacons with heavy jitter are partly classified as ssh_bruteforce on the test day; they still alert and
  correlate, but the family and technique on those alerts are wrong.
* Context features need timestamps and IPs; without them (some CSV exports) they are zero.
* Single process, SQLite, one background job at a time. Not built for high flow rates.

## Ideas for extension

* Sequence models over per-host flow windows (needs a framework this machine does not have installed).
* Payload or TLS metadata features (JA3/JA4, DNS query entropy) from tshark.
* Learn the correlation weights from labelled incidents instead of hand-set points.
* Push incidents back to the lab portal and accept live Zeek or Suricata logs.
