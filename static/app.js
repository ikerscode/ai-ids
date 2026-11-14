// ai-ids HUD logic. Everything shown from the API is escaped with hud.esc.
// Wrapped in a function so these names do not clash with the globals hud.js declares.
(() => {
const { api, toast, esc, renderTable, severityTag, barChart, lineChart, forceGraph, fmtTime } = hud;
const $ = (s) => document.querySelector(s);
let selectedAlert = null;
let selectedIncident = null;
let lastStatus = null;

const pct = (v) => (v === null || v === undefined ? "-" : (100 * v).toFixed(1) + "%");
const num = (v, d = 3) => (v === null || v === undefined ? "-" : Number(v).toFixed(d));
const utc = (t) => new Date(t * 1000).toISOString().replace("T", " ").slice(0, 19);

// ------------------------------------------------------------------ jobs
async function runJob(path, body, after) {
  try {
    const job = await api(path, { method: "POST", body: body || {} });
    toast(`${job.kind} started`);
    pollJob(job.id, after);
  } catch (e) { toast(e.message); }
}

function pollJob(id, after) {
  const t = setInterval(async () => {
    const j = await api(`/api/jobs/${id}`);
    $("#jobLog").textContent = `[${j.kind}] ${j.status}\n` + j.log.join("\n") + (j.error ? "\n" + j.error : "");
    if (j.status !== "running") {
      clearInterval(t);
      toast(`${j.kind}: ${j.status}`);
      refreshStatus();
      if (after) after(j);
    }
  }, 1000);
}

// ------------------------------------------------------------------ status
async function refreshStatus() {
  const s = await api("/api/status");
  lastStatus = s;
  const st = s.stream;
  $("#status").textContent = `dataset ${s.dataset} | model ${s.trained ? "trained" : "not trained"} | ` +
    `${s.alerts.total} alerts | ${s.rules} rules | stream ${st.running ? "running" : "idle"}` + (s.job ? ` | job: ${s.job.kind}` : "");
  for (const sel of ["#ds", "#dsActive"]) {
    const cur = $(sel).value || s.dataset;
    $(sel).innerHTML = s.datasets.map((d) => `<option ${d.name === cur ? "selected" : ""}>${esc(d.name)}</option>`).join("");
  }
}

// ------------------------------------------------------------------ live
async function refreshStream() {
  const st = await api("/api/stream");
  $("#streamInfo").textContent = st.dataset
    ? `${st.running ? "streaming" : "stopped"} ${st.dataset}/${st.split} | sim clock ${utc(st.clock)} UTC | ` +
      `${st.flows_done}/${st.flows_total} flows (${pct(st.progress)}) | ${st.flow_alerts} flow alerts -> ${st.alerts_new} alerts | ${st.speed}x` +
      (st.error ? ` | ERROR ${st.error}` : "")
    : "Stream idle. Start it to replay traffic through the pipeline, or batch-detect a whole split.";
  if (st.rate && st.rate.length > 1) {
    lineChart("#rate", [
      { name: "flows / 10 min", points: st.rate.map((r) => [r[0], r[1]]) },
      { name: "alerted flows / 10 min", points: st.rate.map((r) => [r[0], r[2]]) },
    ], { height: 150, xFormat: (x) => utc(x).slice(11, 16) });
  } else $("#rate").innerHTML = "";
}

async function refreshAlerts() {
  const q = new URLSearchParams({ q: $("#fq").value, cls: $("#fcls").value, severity: $("#fsev").value,
    status: $("#fstatus").value, limit: 400 });
  const rows = await api(`/api/alerts?${q}`);
  const classes = new Set([...$("#fcls").options].map((o) => o.value));
  rows.forEach((r) => { if (!classes.has(r.alert_class)) { classes.add(r.alert_class); $("#fcls").add(new Option(r.alert_class, r.alert_class)); } });
  renderTable("#alerts", rows, [
    { key: "ts", label: "time (UTC)", render: (v) => esc(utc(v).slice(5, 16)) },
    { key: "severity", render: (v) => severityTag(v) },
    { key: "alert_class", label: "class" },
    { key: "src_ip", label: "source", render: (v, r) => esc(r.src_host || v) },
    { key: "dst_ip", label: "dest", render: (v, r) => esc((r.dst_host || v) + (r.dst_port ? ":" + r.dst_port : "")) },
    { key: "count", label: "n" },
    { key: "sensor" },
    { key: "status" },
  ], { empty: "No alerts yet. Train a model, then start the stream or batch-detect." });
  $("#alerts").querySelectorAll("tbody tr").forEach((tr, i) => {
    tr.classList.add("click");
    if (rows[i].id === selectedAlert) tr.classList.add("sel");
    tr.onclick = () => showAlert(rows[i].id);
  });
}

async function showAlert(id) {
  selectedAlert = id;
  const a = await api(`/api/alerts/${id}`);
  const ex = a.explanation || {};
  const at = ex.attack || {};
  const contribs = ex.contributions || [];
  $("#detail").innerHTML = `
    <p>${severityTag(a.severity)} <b>${esc(a.alert_class)}</b> &middot; confidence ${num(a.confidence, 2)} &middot; ${esc(a.count)} event(s) &middot; ${esc(a.status)}</p>
    <p>${esc(ex.summary || a.summary)}</p>
    <div id="bars"></div>
    <table><thead><tr><th>evidence</th><th>weight</th><th>sources</th></tr></thead><tbody>
      ${contribs.map((c) => `<tr><td>${esc(c.text)}</td><td>${num(c.weight, 2)}</td><td class="muted">${esc(Object.entries(c.sources || {}).filter(([, v]) => v > 0.005).map(([k, v]) => k + " " + num(v, 2)).join(", "))}</td></tr>`).join("")}
    </tbody></table>
    <h2 style="margin-top:12px">MITRE ATT&amp;CK</h2>
    <table class="kv"><tbody>
      <tr><td>technique</td><td>${esc(at.technique_id || a.technique_id)} ${esc(at.technique || a.technique)}</td></tr>
      <tr><td>kill-chain stage</td><td>${esc(a.stage)}</td></tr>
      <tr><td>rationale</td><td>${esc(at.rationale || a.rationale)}</td></tr>
    </tbody></table>
    <h2 style="margin-top:12px">Detector scores</h2>
    <table class="kv"><tbody>${Object.entries(ex.detectors || {}).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("")}
      <tr><td>source / sensor</td><td>${esc(a.source)} / ${esc(a.sensor)}</td></tr>
      <tr><td>window</td><td>${esc(utc(a.ts))} to ${esc(utc(a.end_ts))}</td></tr>
      <tr><td>ground truth (simulated)</td><td>${esc(a.truth)} ${esc(a.campaign)}</td></tr></tbody></table>
    <h2 style="margin-top:12px">Analyst feedback</h2>
    <div class="row">
      <button class="btn" id="fbTp">True positive</button>
      <button class="btn danger" id="fbFp">False positive</button>
      <select id="fbSup"><option value="none">no suppression</option><option value="exact">suppress this src/dst/port/class</option><option value="source">suppress this class from this source</option></select>
    </div>
    <p class="muted">FP flows are added to the benign set on the next retrain; rules apply to new detections.</p>
    <details><summary class="muted">raw flow / event</summary><pre>${esc(JSON.stringify(a.raw, null, 1))}</pre></details>`;
  barChart("#bars", contribs.map((c) => ({ label: c.label, value: Number(c.weight.toFixed(3)) })), { color: "#b8322a" });
  $("#fbTp").onclick = () => sendFeedback(id, "tp");
  $("#fbFp").onclick = () => sendFeedback(id, "fp");
  refreshAlerts();
}

async function sendFeedback(id, verdict) {
  try {
    const r = await api(`/api/alerts/${id}/feedback`, { method: "POST", body: { verdict, suppress: $("#fbSup").value } });
    toast(`marked ${verdict}` + (r.rule_id ? `, rule #${r.rule_id} created` : "") + `, ${r.labelled_flows} flows stored`);
    showAlert(id);
    refreshRules();
  } catch (e) { toast(e.message); }
}

// ------------------------------------------------------------------ incidents
async function refreshIncidents() {
  const rows = await api("/api/incidents");
  renderTable("#incTable", rows, [
    { key: "id" },
    { key: "severity", render: (v) => severityTag(v) },
    { key: "score" },
    { key: "title", label: "kill-chain stages (hosts)" },
    { key: "n_alerts", label: "alerts" },
    { key: "systems", render: (v) => esc(v.length + ": " + v.join(", ")) },
    { key: "first_ts", label: "first (UTC)", render: (v) => esc(utc(v).slice(5, 16)) },
    { key: "techniques", render: (v) => esc(v.join(" ")) },
  ], { empty: "No incidents yet." });
  $("#incTable").querySelectorAll("tbody tr").forEach((tr, i) => {
    tr.classList.add("click");
    if (rows[i].id === selectedIncident) tr.classList.add("sel");
    tr.onclick = () => showIncident(rows[i].id);
  });
}

async function showIncident(id) {
  selectedIncident = id;
  const inc = await api(`/api/incidents/${id}`);
  $("#incInfo").innerHTML = `${severityTag(inc.severity)} score ${esc(inc.score)} | ${esc(inc.stage_count)} stages` +
    `${inc.progression ? " in kill-chain order" : ""} | ${esc(inc.systems.length)} systems | max confidence ${esc(inc.max_confidence)}`;
  renderTable("#incTimeline", inc.timeline, [
    { key: "ts", label: "time (UTC)", render: (v) => esc(utc(v).slice(5, 19)) },
    { key: "stage" }, { key: "alert_class", label: "class" }, { key: "technique_id", label: "ATT&CK" },
    { key: "sensor" }, { key: "summary", render: (v) => esc(String(v).slice(0, 140)) },
  ]);
  forceGraph("#incGraph", inc.graph.nodes, inc.graph.links, { height: 420 });
  refreshIncidents();
}

async function loadSample(useIncident) {
  const doc = await api("/api/ingest/schema");
  const ex = { ...doc.example };
  if (useIncident) {
    const incs = await api("/api/incidents");
    if (incs.length) {
      const inc = await api(`/api/incidents/${incs[0].id}`);
      const ip = inc.graph.nodes.find((n) => n.type === "ip");
      if (ip) ex.src_ip = ip.label;
      ex.ts = new Date((inc.first_ts + 300) * 1000).toISOString();
      ex.summary = `honeypot saw ${ex.src_ip} trying credentials`;
    }
  }
  $("#ingestBody").value = JSON.stringify(ex, null, 2);
}

async function sendIngest() {
  let body;
  try { body = JSON.parse($("#ingestBody").value); } catch (e) { toast("not valid JSON"); return; }
  const res = await fetch("/api/ingest", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const out = await res.json();
  $("#ingestOut").textContent = `HTTP ${res.status}\n` + JSON.stringify(out, null, 1);
  refreshIncidents();
}

// ------------------------------------------------------------------ models & data
async function refreshDataset() {
  const name = $("#dsActive").value || (lastStatus && lastStatus.dataset);
  if (!name) { $("#dsStats").innerHTML = '<p class="muted">No dataset yet. Generate one.</p>'; return; }
  try {
    const d = await api(`/api/datasets/${encodeURIComponent(name)}`);
    const s = d.stats;
    const labels = Object.entries(s.by_label).map(([k, v]) => ({ label: k, flows: v }));
    $("#dsStats").innerHTML = `<p class="muted">${esc(name)}: ${esc(s.flows)} flows, ${esc(s.events)} host events, ${esc(s.hosts)} hosts, ` +
      `${s.time_range ? esc(utc(s.time_range[0])) + " to " + esc(utc(s.time_range[1])) : ""} | ${esc(d.meta.kind || "")} ${esc(d.meta.source || "")}</p>` +
      (d.meta.note ? `<p class="muted">${esc(d.meta.note)}</p>` : "") + '<div id="dsLabels"></div><div id="dsSensors"></div>';
    renderTable("#dsLabels", labels);
    renderTable("#dsSensors", Object.entries(s.by_sensor).map(([k, v]) => ({ sensor: k, flows: v })));
  } catch (e) { $("#dsStats").innerHTML = `<p class="muted">${esc(e.message)}</p>`; }
}

async function refreshModels() {
  const m = await api("/api/models");
  if (!m.trained) { $("#modelInfo").innerHTML = '<p class="muted">Not trained yet.</p>'; }
  else {
    const t = (m.metrics && m.metrics.test) || {};
    $("#modelInfo").innerHTML = `<table class="kv"><tbody>
      <tr><td>dataset</td><td>${esc(m.meta.dataset)}</td></tr>
      <tr><td>trained</td><td>${esc(fmtTime(m.meta.trained_at))} (${esc(m.meta.seconds)} s, ${esc(m.meta.benign_flows)} benign flows)</td></tr>
      <tr><td>classes</td><td>${esc(m.meta.classes.join(", "))}</td></tr>
      <tr><td>test precision / recall</td><td>${num(t.precision)} / ${num(t.recall)}</td></tr>
      <tr><td>test FPR / PR-AUC</td><td>${num(t.fpr, 4)} / ${num(t.pr_auc)}</td></tr>
      <tr><td>alerted flows per hour</td><td>${esc(t.alerts_per_hour)}</td></tr></tbody></table>`;
    const th = m.thresholds;
    $("#thresholds").innerHTML = `<table class="kv"><tbody>
      <tr><td>anomaly ensemble (target FPR ${th.target_fpr_anomaly})</td><td>${num(th.tau_ens)}</td></tr>
      <tr><td>naive ensemble (train 95th pct)</td><td>${num(th.tau_ens_naive)}</td></tr>
      <tr><td>supervised p(attack) (target FPR ${th.target_fpr_supervised})</td><td>${num(th.tau_sup)}</td></tr>
      <tr><td>detector votes (each at FPR ${th.vote_fpr}, need ${th.min_votes})</td><td>${esc(Object.entries(th.vote).map(([k, v]) => k + " " + num(v, 2)).join(", "))}</td></tr>
      <tr><td>per-host overrides</td><td>${esc(Object.entries(m.entity_thresholds || {}).map(([k, v]) => k + " " + num(v, 2)).join(", ") || "none")}</td></tr>
      <tr><td>calibration flows</td><td>${esc(th.calibration_flows)}</td></tr></tbody></table>`;
    if (m.metrics && m.metrics.per_family) {
      renderTable("#perFamily", m.metrics.per_family, [
        { key: "label" }, { key: "flows" }, { key: "alerted", render: pct },
        { key: "correct_class", label: "right class", render: pct }, { key: "unknown_anomaly", label: "as unknown", render: pct }]);
      const c = m.metrics.confusion;
      $("#confusion").innerHTML = `<table><thead><tr><th></th>${c.cols.map((x) => `<th>${esc(x)}</th>`).join("")}</tr></thead><tbody>` +
        c.rows.map((r, i) => `<tr><td>${esc(r)}</td>${c.matrix[i].map((v, j) => `<td style="${c.cols[j] === r ? "color:#fff" : v ? "color:var(--warn)" : "color:var(--muted)"}">${v}</td>`).join("")}</tr>`).join("") + "</tbody></table>";
    }
  }
  const ev = m.evaluation;
  if (ev) {
    renderTable("#ablation", ev.ablation, [
      { key: "description", label: "stage" }, { key: "precision", render: (v) => num(v) }, { key: "recall", render: (v) => num(v) },
      { key: "fpr", label: "FPR", render: (v) => num(v, 4) }, { key: "pr_auc", label: "PR-AUC", render: (v) => num(v) },
      { key: "alerts_per_hour", label: "alerts/h" }, { key: "false_alerts_per_hour", label: "false alerts/h" },
      { key: "step_recall", label: "attack steps caught", render: pct }]);
    $("#ablation").insertAdjacentHTML("beforeend", `<p class="muted">${esc(ev.test_flows)} test flows over ${esc(ev.test_hours)} h. Rows 1-5 count alerted flows; the last row counts deduplicated alerts. Rules learned from the feedback day: ${esc(ev.rules_learned.map((r) => `${r.alert_class} from ${r.src_ip}`).join("; ") || "none")}.</p>`);
    renderTable("#lofo", ev.lofo || [], [
      { key: "held_out", label: "family held out of training" }, { key: "test_flows", label: "test flows" },
      { key: "recall_when_trained", label: "recall (trained on it)", render: pct },
      { key: "recall_held_out", label: "recall (never seen)", render: pct },
      { key: "via_anomaly_layer", label: "flagged by anomaly layer", render: pct },
      { key: "labelled_unknown_anomaly", label: "labelled unknown_anomaly", render: pct },
      { key: "steps_detected", label: "attack steps caught" }, { key: "benign_fpr", label: "benign FPR", render: (v) => num(v, 4) }],
      { empty: "Run Evaluate." });
  } else {
    $("#ablation").innerHTML = '<p class="muted">Run Evaluate.</p>';
    $("#lofo").innerHTML = '<p class="muted">Run Evaluate.</p>';
  }
}

async function refreshRules() {
  const r = await api("/api/rules");
  renderTable("#rules", r.rules, [
    { key: "id" }, { key: "alert_class", label: "class" }, { key: "src_ip", label: "src" }, { key: "dst_ip", label: "dst" },
    { key: "dst_port", label: "port" }, { key: "hits" }, { key: "reason" },
    { key: "id", label: "", render: (v) => `<button class="btn danger" data-rule="${esc(v)}">delete</button>` }],
    { empty: "No rules. Mark an alert as a false positive with suppression to create one." });
  $("#rules").querySelectorAll("button[data-rule]").forEach((b) => {
    b.onclick = async () => { await api(`/api/rules/${b.dataset.rule}`, { method: "DELETE" }); refreshRules(); };
  });
}

async function upload() {
  const f = $("#upFile").files[0];
  if (!f) { toast("choose a file"); return; }
  const kind = f.name.toLowerCase().endsWith(".csv") ? "csv" : "pcap";
  try {
    // Raw body upload (hud.api would JSON-encode a File object).
    const res = await fetch(`/api/upload?kind=${kind}&filename=${encodeURIComponent(f.name)}`, { method: "POST", body: f });
    if (!res.ok) throw new Error(`${res.status}: ${(await res.text()).slice(0, 200)}`);
    const job = await res.json();
    toast(`uploaded ${f.name}`);
    pollJob(job.id, () => { refreshDataset(); refreshAlerts(); });
  } catch (e) { toast(e.message); }
}

// ------------------------------------------------------------------ wiring
$("#btnStart").onclick = async () => {
  try { await api("/api/stream/start", { method: "POST", body: { dataset: $("#ds").value, split: $("#split").value, speed: Number($("#speed").value) } }); toast("stream started"); }
  catch (e) { toast(e.message); }
};
$("#btnStop").onclick = async () => { await api("/api/stream/stop", { method: "POST" }); toast("stream stopped"); };
$("#btnSpeed").onclick = async () => { await api("/api/stream/speed", { method: "POST", body: { speed: Number($("#speed").value) } }); };
$("#btnDetect").onclick = () => runJob("/api/detect", { dataset: $("#ds").value, split: $("#split").value }, refreshAlerts);
$("#btnClear").onclick = async () => { if (confirm("Delete all alerts?")) { await api("/api/alerts/clear", { method: "POST" }); refreshAlerts(); } };
["#fq", "#fcls", "#fsev", "#fstatus"].forEach((s) => $(s).addEventListener("change", refreshAlerts));
$("#btnSample").onclick = () => loadSample(false);
$("#btnSampleInc").onclick = () => loadSample(true);
$("#btnIngest").onclick = sendIngest;
$("#btnSim").onclick = () => runJob("/api/simulate", { scale: Number($("#simScale").value) }, refreshDataset);
$("#btnUpload").onclick = upload;
$("#btnActive").onclick = async () => { await api("/api/datasets/active", { method: "POST", body: { dataset: $("#dsActive").value } }); refreshStatus(); refreshDataset(); };
$("#dsActive").onchange = refreshDataset;
$("#btnTrain").onclick = () => runJob("/api/train", { dataset: $("#dsActive").value }, refreshModels);
$("#btnEval").onclick = () => runJob("/api/evaluate", { dataset: $("#dsActive").value }, refreshModels);

let currentView = "live";
hud.initTabs((id) => {
  currentView = id;
  if (id === "incidents") refreshIncidents();
  if (id === "models") { refreshDataset(); refreshModels(); refreshRules(); }
});
loadSample(false);
refreshStatus().then(() => { if (currentView === "models") refreshDataset(); });
hud.every(3000, async () => {
  await refreshStatus();
  if (currentView === "live") { refreshStream(); refreshAlerts(); }
  if (currentView === "incidents" && lastStatus.stream.running) refreshIncidents();
});
})();
