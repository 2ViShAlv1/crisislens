"use strict";
/* CrisisLens console. All post text is untrusted: it only ever reaches the DOM via textContent. */

// ---------- tiny DOM helpers ----------
const $ = (q, r = document) => r.querySelector(q);
function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return e;
}
const NS = "http://www.w3.org/2000/svg";
function s(tag, attrs, text) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}
const fmt3 = v => (v == null || isNaN(v) ? "—" : Number(v).toFixed(3));
const pct = (v, d = 0) => (v == null || isNaN(v) ? "—" : (100 * v).toFixed(d) + "%");

// ---------- domain vocabulary ----------
const CAT = {
  requests_or_urgent_needs: ["Urgent needs", "life"], injured_or_dead_people: ["Injured / dead", "life"],
  missing_or_found_people: ["Missing / found", "life"], displaced_people_and_evacuations: ["Displaced / evacuations", "ops"],
  infrastructure_and_utility_damage: ["Infrastructure damage", "ops"], caution_and_advice: ["Caution & advice", "ops"],
  rescue_volunteering_or_donation_effort: ["Rescue / donations", "info"], other_relevant_information: ["Other relevant info", "info"],
  sympathy_and_support: ["Sympathy & support", "low"], not_humanitarian: ["Not humanitarian", "low"],
};
const URG = { 1: "Minimal", 2: "Low", 3: "Moderate", 4: "High", 5: "Life-threatening" };
const EVENTS = { kerala_floods_2018: "Kerala floods 2018", cyclone_idai_2019: "Cyclone Idai 2019",
  mexico_earthquake_2017: "Mexico earthquake 2017", hurricane_dorian_2019: "Hurricane Dorian 2019" };
const evName = e => EVENTS[e] || String(e).replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
const catName = c => (CAT[c] || [c || "unknown"])[0];
const tier = c => (CAT[c] || [0, "low"])[1];
const U = u => Math.max(1, Math.min(5, parseInt(u, 10) || 1));

const catChip = (c, cls = "") => h("span", { class: `cat t-${tier(c)} ${cls}`, title: c, text: catName(c) });
const urgBadge = u => h("span", { class: "urg" }, h("span", { class: `urg-badge u${U(u)}`, text: U(u) }), h("span", { class: "urg-word", text: URG[U(u)] }));
function meter(u) {
  u = U(u);
  return h("span", { class: `meter u${u}`, role: "img", "aria-label": `Urgency ${u} of 5, ${URG[u]}` },
    [1, 2, 3, 4, 5].map(i => h("span", { class: i <= u ? "on" : "" })), h("em", { text: `${u} · ${URG[u]}` }));
}
const needChips = n => (Array.isArray(n) && n.length ? h("span", { class: "chips" }, n.map(x => h("span", { class: "need", text: x }))) : h("span", { class: "muted", text: "none stated" }));

// ---------- state ----------
const S = { feed: [], dash: null, examples: [], event: "all", filter: "all", shown: [], idx: 0, timer: null, sel: null, seq: 0, evalDrawn: false };

// ---------- boot ----------
async function getJSON(u) { const r = await fetch(u, { cache: "no-cache" }); if (!r.ok) throw new Error(u + " " + r.status); return r.json(); }
async function boot() {
  initTabs(); initTheme();
  try {
    const t = await (await fetch("/cache/feed.jsonl", { cache: "no-cache" })).text();
    S.feed = t.split("\n").filter(Boolean).map(l => JSON.parse(l));
  } catch (e) { S.feed = []; }
  try { S.dash = await getJSON("/cache/dashboard.json"); } catch (e) { S.dash = null; }
  try { S.examples = await getJSON("/cache/examples.json"); } catch (e) { S.examples = []; }
  initConsole(); initAnalyze(); renderModel();
  if (S.dash) $("#genStamp").textContent = `Results generated ${S.dash.generated}`;
  health(); setInterval(health, 20000);
  window.addEventListener("resize", debounce(() => { if (!$("#view-eval").hidden) renderEval(); }, 150));
}
function debounce(f, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => f(...a), ms); }; }

async function health() {
  const el = $("#status");
  try {
    const j = await getJSON("/api/health");
    el.dataset.state = j.live ? "live" : "cached";
    el.lastElementChild.textContent = j.live ? "Live model online" : "Model offline · cached outputs";
    el.title = j.live ? "Analyze requests run on the fine-tuned model" : `Live model unavailable (${j.reason || "unknown"}); known posts use cached outputs`;
  } catch (e) { el.dataset.state = "cached"; el.lastElementChild.textContent = "Static mode · cached outputs"; }
}

// ---------- tabs / theme ----------
function initTabs() {
  const tabs = [...document.querySelectorAll(".tabs button")];
  const show = v => {
    tabs.forEach(b => b.setAttribute("aria-selected", String(b.dataset.view === v)));
    document.querySelectorAll(".view").forEach(sec => (sec.hidden = sec.id !== `view-${v}`));
    window.scrollTo(0, 0);
    if (v === "eval") renderEval();
    try { history.replaceState(null, "", "#" + v); } catch (e) {}
  };
  tabs.forEach(b => b.addEventListener("click", () => show(b.dataset.view)));
  const want = location.hash.slice(1);
  if (tabs.some(b => b.dataset.view === want)) setTimeout(() => show(want), 0);
}
function initTheme() {
  $("#themeBtn").addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("cl-theme", root.dataset.theme); } catch (e) {}
    if (!$("#view-eval").hidden) renderEval();
  });
}

// =====================================================================
// TRIAGE CONSOLE
// =====================================================================
function initConsole() {
  const evs = ["all", ...Object.keys(EVENTS).filter(e => S.feed.some(f => f.event === e))];
  const seg = $("#eventSeg");
  evs.forEach(e => seg.append(h("button", { class: e === "all" ? "on" : "", "data-e": e, text: e === "all" ? "All disasters" : evName(e),
    onclick: ev => { seg.querySelectorAll("button").forEach(b => b.classList.toggle("on", b === ev.currentTarget)); S.event = e; resetFeed(); } })));
  $("#playBtn").addEventListener("click", () => (S.timer ? pause() : play()));
  $("#resetBtn").addEventListener("click", resetFeed);
  $("#speed").addEventListener("change", () => { if (S.timer) { pause(); play(); } });
  $("#queueFilter").addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    $("#queueFilter").querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
    S.filter = b.dataset.f; renderQueue();
  });
  renderKPIs(); renderMix();
}
const pool = () => S.feed.filter(f => S.event === "all" || f.event === S.event);
function play() {
  if (S.idx >= pool().length) resetFeed();
  const ms = +$("#speed").value;
  S.timer = setInterval(step, ms); step();
  $("#playBtn .ico").textContent = "❚❚"; $("#playBtn .lbl").textContent = "Pause";
}
function pause() {
  clearInterval(S.timer); S.timer = null;
  $("#playBtn .ico").textContent = "▶"; $("#playBtn .lbl").textContent = S.idx ? "Resume" : "Start feed";
}
function resetFeed() { pause(); S.shown = []; S.idx = 0; S.sel = null; renderQueue(); renderKPIs(); renderMix(); renderDetail(); $("#playBtn .lbl").textContent = "Start feed"; }
function step() {
  const p = pool();
  if (S.idx >= p.length) { pause(); $("#playBtn .lbl").textContent = "Replay"; return; }
  const item = { ...p[S.idx++], seq: ++S.seq };
  S.shown.push(item); S.lastNew = item.seq;
  renderQueue(); renderKPIs(); renderMix();
}
function ranked() {
  const f = S.filter;
  return S.shown.filter(x => f === "all" || (f === "critical" && U(x.pred.urgency) >= 4) || (f === "actionable" && x.pred.actionable) || (f === "life" && tier(x.pred.category) === "life"))
    .sort((a, b) => U(b.pred.urgency) - U(a.pred.urgency) || (b.pred.actionable === true) - (a.pred.actionable === true) || b.seq - a.seq);
}
function renderQueue() {
  const q = $("#queue"), rows = ranked().slice(0, 120);
  $("#queueEmpty").hidden = S.shown.length > 0;
  q.hidden = S.shown.length === 0;
  q.replaceChildren(...rows.map(x => {
    const p = x.pred;
    const meta = [
      p.location ? h("span", { class: "loc", text: p.location }) : null,
      ...(Array.isArray(p.needs) ? p.needs.slice(0, 3).map(n => h("span", { class: "need", text: n })) : []),
      h("span", { text: evName(x.event) }),
    ];
    return h("button", { class: `qrow${x.seq === S.lastNew ? " new" : ""}${S.sel === x.seq ? " sel" : ""}`, role: "listitem",
      "aria-label": `${URG[U(p.urgency)]} urgency, ${catName(p.category)}`, onclick: () => { S.sel = x.seq; renderQueue(); renderDetail(); } },
      urgBadge(p.urgency),
      h("div", {}, h("div", { class: "qtop" }, catChip(p.category), p.actionable ? h("span", { class: "pill act", text: "Actionable" }) : null),
        h("div", { class: "qtext", text: x.text }), h("div", { class: "qmeta" }, meta)));
  }));
  S.lastNew = null;
}
function kpi(k, v, d, dot) {
  return h("div", { class: "kpi" }, h("div", { class: "k" }, dot ? h("i", { class: "dot", style: `background:${dot}` }) : null, k),
    h("div", { class: "v", text: v }), h("div", { class: "d" }, d));
}
function renderKPIs() {
  const n = S.shown.length, P = pool().length;
  const crit = S.shown.filter(x => U(x.pred.urgency) >= 4).length, act = S.shown.filter(x => x.pred.actionable).length;
  const life = S.shown.filter(x => tier(x.pred.category) === "life").length;
  const agree = S.shown.filter(x => x.pred.category === x.gold).length;
  $("#kpis").replaceChildren(
    kpi("Posts triaged", n.toLocaleString(), `of ${P.toLocaleString()} held-out posts`),
    kpi("Urgency ≥ 4", crit, n ? `${pct(crit / n)} of triaged` : "high or life-threatening", "var(--u5)"),
    kpi("Actionable", act, n ? `${pct(act / n)} need a response` : "a responder can act on it", "var(--good)"),
    kpi("Life-safety categories", life, "urgent needs · injured · missing", "var(--tier-life)"),
    kpi("Agrees with human label", n ? pct(agree / n, 1) : "—", n ? `${agree} / ${n} vs HumAID annotators` : "category vs HumAID gold"),
  );
}
function renderMix() {
  const counts = {}; S.shown.forEach(x => (counts[x.pred.category] = (counts[x.pred.category] || 0) + 1));
  const rows = Object.keys(CAT).map(c => [c, counts[c] || 0]).sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...rows.map(r => r[1]));
  $("#mixN").textContent = S.shown.length ? `${S.shown.length} posts` : "";
  $("#mix").replaceChildren(...rows.map(([c, n]) => h("div", { class: `mix-row t-${tier(c)}` },
    h("span", { class: "nm", title: c, text: catName(c) }), h("span", {}, h("div", { class: "bar", style: `width:${(100 * n) / max}%;opacity:${n ? 1 : .25}` })), h("span", { class: "n", text: n }))));
  const nc = {}; S.shown.forEach(x => (Array.isArray(x.pred.needs) ? x.pred.needs : []).forEach(t => { t = String(t).toLowerCase().trim(); if (t) nc[t] = (nc[t] || 0) + 1; }));
  const top = Object.entries(nc).sort((a, b) => b[1] - a[1]).slice(0, 14);
  $("#needs").replaceChildren(...(top.length ? top.map(([t, n]) => h("span", { class: "need" }, t, h("b", { text: n }))) : [h("span", { class: "muted small", text: "Needs extracted by the model appear here." })]));
}
function renderDetail() {
  const d = $("#detail"), x = S.shown.find(y => y.seq === S.sel);
  if (!x) { d.replaceChildren(h("div", { class: "detail-empty muted", text: "Select a post in the queue to see the full triage record." })); return; }
  const p = x.pred, ok = p.category === x.gold;
  d.replaceChildren(
    h("h3", { text: `Triage record · ${evName(x.event)}` }),
    h("p", { class: "post", text: x.text }),
    h("dl", { class: "kv" },
      h("dt", { text: "Category" }), h("dd", {}, catChip(p.category)),
      h("dt", { text: "Urgency" }), h("dd", {}, meter(p.urgency)),
      h("dt", { text: "Actionable" }), h("dd", {}, p.actionable ? h("span", { class: "pill act", text: "Yes, needs a response" }) : h("span", { class: "pill", text: "No, informational" })),
      h("dt", { text: "Location" }), h("dd", { text: p.location || "not stated" }),
      h("dt", { text: "Needs" }), h("dd", {}, needChips(p.needs)),
      h("dt", { text: "Rationale" }), h("dd", { class: "rationale", text: p.rationale || "—" }),
      h("dt", { text: "Model latency" }), h("dd", { text: x.latency_s ? `${x.latency_s.toFixed(2)} s on a laptop CPU` : "—" })),
    h("div", { class: `gold ${ok ? "ok" : "no"}` }, h("span", { class: "mk", text: ok ? "✓" : "✗" }),
      h("span", {}, ok ? "Matches the human (HumAID) label." : "Human (HumAID) label differs: ", ok ? null : h("b", { text: catName(x.gold) }))),
  );
}

// =====================================================================
// ANALYZE
// =====================================================================
function initAnalyze() {
  const ta = $("#postIn");
  ta.addEventListener("input", () => ($("#charN").textContent = ta.value.length));
  ta.addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); analyze(); } });
  $("#runBtn").addEventListener("click", analyze);
  const ex = S.examples.map(e => ({ text: e.text, src: "Example" }));
  const want = ["requests_or_urgent_needs", "injured_or_dead_people", "infrastructure_and_utility_damage", "displaced_people_and_evacuations"];
  want.forEach(c => { const f = S.feed.find(x => x.gold === c && x.pred.category === c && x.text.length > 60); if (f) ex.push({ text: f.text, src: `Held-out · ${evName(f.event)}` }); });
  $("#examples").replaceChildren(...ex.map(e => h("button", { class: "ex", title: e.text,
    onclick: () => { ta.value = e.text; $("#charN").textContent = ta.value.length; analyze(); } },
    h("span", { class: "t", text: e.text }), h("span", { class: "src", text: e.src }))));
}
let busy = false;
async function analyze() {
  const text = $("#postIn").value.trim(), box = $("#result");
  if (!text || busy) { if (!text) $("#postIn").focus(); return; }
  busy = true; $("#runBtn").disabled = true;
  const t0 = performance.now(), el = h("span", { text: "0.0 s" });
  const tick = setInterval(() => (el.textContent = ((performance.now() - t0) / 1000).toFixed(1) + " s"), 100);
  box.replaceChildren(h("div", { class: "thinking" }, h("div", { class: "spinner" }), h("div", {}, "Running the fine-tuned model on CPU… ", el),
    h("div", { style: "width:70%;display:grid;gap:8px" }, h("div", { class: "skel" }), h("div", { class: "skel", style: "width:70%" }), h("div", { class: "skel", style: "width:85%" }))));
  try {
    const r = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.detail || `Request failed (${r.status})`);
    renderResult(j);
  } catch (e) {
    box.replaceChildren(h("div", { class: "err" }, h("b", { text: "Could not triage this post. " }), e.message));
  } finally { clearInterval(tick); busy = false; $("#runBtn").disabled = false; }
}
function renderResult(j) {
  const o = j.result, box = $("#result");
  if (!o) { box.replaceChildren(h("div", { class: "err" }, h("b", { text: "The model returned unparseable output." }), h("pre", { class: "json", text: j.raw || "" }))); return; }
  const json = JSON.stringify(o, null, 2);
  const copy = h("button", { class: "btn ghost", text: "Copy JSON", onclick: async ev => {
    try { await navigator.clipboard.writeText(json); ev.currentTarget.textContent = "Copied"; } catch (e) { ev.currentTarget.textContent = "Copy failed"; } } });
  box.replaceChildren(
    h("div", { class: "r-head" }, catChip(o.category, "r-cat"), o.actionable ? h("span", { class: "pill act", text: "Actionable" }) : h("span", { class: "pill", text: "Informational" }),
      j.valid ? null : h("span", { class: "pill", style: "border-color:var(--bad)", text: "Schema check failed" })),
    h("dl", { class: "kv" },
      h("dt", { text: "Urgency" }), h("dd", {}, meter(o.urgency)),
      h("dt", { text: "Location" }), h("dd", { text: o.location || "not stated in the post" }),
      h("dt", { text: "Needs" }), h("dd", {}, needChips(o.needs)),
      h("dt", { text: "Rationale" }), h("dd", { class: "rationale", text: o.rationale || "—" })),
    h("div", { class: "r-foot" },
      j.source === "live"
        ? h("span", { class: "src-badge" }, h("i"), `Live model · ${j.latency_s.toFixed(2)} s on CPU`)
        : h("span", { class: "src-badge cached", title: j.note || "" }, h("i"), "Cached output from evaluation (live model unavailable)"),
      copy),
    h("details", {}, h("summary", { text: "Raw JSON record" }), h("pre", { class: "json", text: json })),
  );
}

// =====================================================================
// EVALUATION (SVG charts, hover tooltips)
// =====================================================================
const tip = () => $("#tip");
function bindTip(node, html) {
  node.addEventListener("mousemove", e => { const t = tip(); t.replaceChildren(...html()); t.hidden = false;
    const x = Math.min(e.clientX + 14, innerWidth - t.offsetWidth - 8), y = Math.min(e.clientY + 14, innerHeight - t.offsetHeight - 8);
    t.style.left = x + "px"; t.style.top = y + "px"; });
  node.addEventListener("mouseleave", () => (tip().hidden = true));
}
const tipLines = (title, lines) => () => [h("b", { text: title }), ...lines.map(l => h("div", { text: l }))];
function barPath(x0, y, w, hh, r = 4) {
  if (w <= 0) return "";
  r = Math.min(r, w, hh / 2);
  return `M${x0},${y}h${w - r}a${r},${r} 0 0 1 ${r},${r}v${hh - 2 * r}a${r},${r} 0 0 1 ${-r},${r}h${-(w - r)}z`;
}
function axis(svg, x, x0, W, top, H, ticks, fmtT) {
  ticks.forEach(t => { svg.append(s("line", { class: "grid", x1: x(t), x2: x(t), y1: top - 4, y2: H - 20 }));
    svg.append(s("text", { x: x(t), y: H - 4, "text-anchor": "middle", "font-size": 11 }, fmtT(t))); });
}
function labelCol(W) { return Math.max(110, Math.min(250, W * 0.36)); }
function clip(t, n) { return t.length > n ? t.slice(0, n - 1) + "…" : t; }

// single-series horizontal bars (+ optional CI); highlighted rows use series-1, the rest stay recessive
function hbar(el, rows, { max = 1, ticks = [0, .25, .5, .75, 1], fmt = fmt3 } = {}) {
  const W = el.clientWidth || 640, L = labelCol(W), rh = 30, top = 8, H = top + rows.length * rh + 26;
  const x = v => L + ((W - L - 54) * v) / max;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img" });
  axis(svg, x, L, W, top, H, ticks, t => t.toFixed(2));
  rows.forEach((r, i) => {
    const y = top + i * rh, bh = 14, by = y + (rh - bh) / 2;
    svg.append(s("text", { x: L - 10, y: y + rh / 2 + 4, "text-anchor": "end", class: r.hl ? "lbl-strong" : "" }, clip(r.label, Math.floor(L / 6.6))));
    svg.append(s("path", { d: barPath(x(0), by, x(r.value) - x(0), bh), class: r.hl ? "s1" : "s0" }));
    let end = x(r.value);
    if (r.ci) { const [a, b] = r.ci.map(x); end = Math.max(end, b);
      svg.append(s("line", { class: "ci", x1: a, x2: b, y1: by + bh / 2, y2: by + bh / 2 }));
      [a, b].forEach(c => svg.append(s("line", { class: "ci", x1: c, x2: c, y1: by + 3, y2: by + bh - 3 }))); }
    svg.append(s("text", { x: end + 6, y: by + bh / 2 + 4, class: "val" }, fmt(r.value)));
    const hit = s("rect", { class: "hit", x: 0, y, width: W, height: rh }); svg.append(hit); bindTip(hit, r.tip);
  });
  el.replaceChildren(svg);
}
// grouped horizontal bars, 2 series (legend + values on bars)
function hbar2(el, cats, series, { max = 1, labelFmt = c => c, tipFor } = {}) {
  const W = el.clientWidth || 640, L = labelCol(W), bh = 10, gap = 2, rh = 2 * bh + gap + 14, top = 6, H = top + cats.length * rh + 26;
  const x = v => L + ((W - L - 46) * v) / max;
  const leg = h("div", { class: "legend" }, series.map(sr => h("span", {}, h("i", { style: `background:var(--${sr.cls})` }), sr.name)));
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img" });
  axis(svg, x, L, W, top, H, [0, .25, .5, .75, 1], t => t.toFixed(2));
  cats.forEach((c, i) => {
    const y = top + i * rh + 7;
    svg.append(s("text", { x: L - 10, y: y + bh + 4, "text-anchor": "end" }, clip(labelFmt(c, i), Math.floor(L / 6.6))));
    series.forEach((sr, k) => {
      const v = sr.values[i], by = y + k * (bh + gap);
      if (v == null) return;
      svg.append(s("path", { d: barPath(x(0), by, x(v) - x(0), bh, 3), class: sr.cls }));
      svg.append(s("text", { x: x(v) + 5, y: by + bh - 1, class: "val", "font-size": 10.5 }, v.toFixed(2)));
    });
    const hit = s("rect", { class: "hit", x: 0, y: top + i * rh, width: W, height: rh }); svg.append(hit);
    bindTip(hit, tipFor ? tipFor(c, i) : tipLines(labelFmt(c, i), series.map(sr => `${sr.name}: ${sr.values[i] == null ? "—" : sr.values[i].toFixed(3)}`)));
  });
  el.replaceChildren(leg, svg);
}
// AutoScientist ladder: one row per run, dots = iterations (filled = best), x = judge win-rate
function ladder(el, runs, oursId) {
  const W = el.clientWidth || 640, L = labelCol(W), rh = 34, top = 8, H = top + runs.length * rh + 26;
  const all = runs.flatMap(r => r.pts.map(p => p.w)); const lo = Math.max(0, Math.floor((Math.min(...all) - .05) * 10) / 10);
  const x = v => L + ((W - L - 60) * (v - lo)) / (1 - lo);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img" });
  const ticks = []; for (let t = lo; t <= 1.0001; t += .1) ticks.push(+t.toFixed(1));
  axis(svg, x, L, W, top, H, ticks, t => t.toFixed(1));
  runs.forEach((r, i) => {
    const y = top + i * rh + rh / 2, ours = r.id === oursId, best = Math.max(...r.pts.map(p => p.w));
    svg.append(s("text", { x: L - 10, y: y + 4, "text-anchor": "end", class: ours ? "lbl-strong" : "" }, clip(r.label, Math.floor(L / 6.6))));
    if (r.pts.length > 1) svg.append(s("polyline", { class: `ln${ours ? " ours" : ""}`, points: r.pts.map(p => `${x(p.w)},${y}`).join(" ") }));
    r.pts.forEach(p => svg.append(s("circle", { cx: x(p.w), cy: y, r: 5, class: `pt${ours ? " ours" : ""}${p.w === best ? " best" : ""}` })));
    svg.append(s("text", { x: x(Math.max(...r.pts.map(p => p.w))) + 10, y: y + 4, class: "val" }, best.toFixed(3)));
    const hit = s("rect", { class: "hit", x: 0, y: y - rh / 2, width: W, height: rh }); svg.append(hit);
    bindTip(hit, tipLines(r.full || r.label, r.pts.map(p => `iter ${p.it}: win-rate ${p.w.toFixed(3)} · lr ${p.lr} · ${p.ep} ep · LoRA r${p.r}`)));
  });
  el.replaceChildren(svg);
}

function renderEval() {
  const D = S.dash;
  if (!D) { $("#view-eval").querySelector(".eval-grid").replaceChildren(h("div", { class: "panel", text: "Results file not found (demo/cache/dashboard.json)." })); return; }
  const sys = D.systems, ours = sys.find(x => x.kind === "ours") || sys[0];
  const others = sys.filter(x => x !== ours), apis = sys.filter(x => x.kind === "api");
  const bestApi = apis.reduce((a, b) => (b.macro_f1 > (a?.macro_f1 ?? -1) ? b : a), null);
  const bestApiUrg = apis.reduce((a, b) => (b.recall_urgent > (a?.recall_urgent ?? -1) ? b : a), null);
  const tfidf = sys.find(x => x.key === "tfidf");
  const apiCost = Math.min(...apis.map(a => a.usd_per_1k ?? Infinity));
  $("#heroStats").replaceChildren(
    Object.assign(kpi("Macro-F1 on unseen disasters", fmt3(ours.macro_f1), h("span", {}, "best frontier API ", h("b", { text: fmt3(bestApi?.macro_f1) }), tfidf ? ` · TF-IDF ${fmt3(tfidf.macro_f1)}` : "")), { className: "kpi hero" }),
    kpi("Urgent-needs recall", fmt3(ours.recall_urgent), h("span", {}, bestApiUrg ? `${bestApiUrg.label.replace(/ zero-shot.*/, "")}: ` : "", h("b", { text: fmt3(bestApiUrg?.recall_urgent) }))),
    kpi("Cost per 1,000 posts", "$0", isFinite(apiCost) ? h("span", {}, "self-hosted CPU · cheapest API ", h("b", { text: `$${apiCost.toFixed(3)}` })) : "self-hosted CPU"),
    kpi("Valid JSON records", pct(ours.json_valid, 1), `strict schema · n = ${ours.n.toLocaleString()} posts`),
  );
  const sorted = [...sys].sort((a, b) => b.macro_f1 - a.macro_f1);
  hbar($("#chMacro"), sorted.map(x => ({ label: x.label, value: x.macro_f1, ci: x.ci, hl: x === ours,
    tip: tipLines(x.label, [`macro-F1 ${fmt3(x.macro_f1)} [${fmt3(x.ci[0])}, ${fmt3(x.ci[1])}]`, `weighted-F1 ${fmt3(x.weighted_f1)}`, `n = ${x.n}`]) })));
  hbar2($("#chRecall"), sorted.map(x => x.label), [
    { name: "Urgent needs", cls: "s1", values: sorted.map(x => x.recall_urgent) },
    { name: "Injured / dead", cls: "s2", values: sorted.map(x => x.recall_injured) }]);
  const pe = D.per_event || {}, peOurs = pe[ours.label], peApi = bestApi ? pe[bestApi.label] : null;
  if (peOurs) { const evs = Object.keys(peOurs);
    hbar2($("#chEvent"), evs, [{ name: ours.label, cls: "s1", values: evs.map(e => peOurs[e]) },
      ...(peApi ? [{ name: bestApi.label, cls: "s2", values: evs.map(e => peApi[e]) }] : [])], { labelFmt: evName }); }
  const pc = D.per_class;
  hbar2($("#chClass"), pc.classes, [{ name: ours.label, cls: "s1", values: pc.ours }, { name: pc.baseline_label, cls: "s2", values: pc.baseline }],
    { labelFmt: (c, i) => `${catName(c)} (n=${pc.support[i]})` });
  const sv = D.silver || {};
  $("#silver").replaceChildren(
    kpi("Actionable accuracy", pct(sv.actionable_acc, 1), `F1 ${fmt3(sv.actionable_f1)}`),
    kpi("Urgency within ±1", pct(sv.urgency_within1, 1), `MAE ${sv.urgency_mae?.toFixed(2) ?? "—"} levels`),
    kpi("Location exact match", pct(sv.location_exact, 1), `overlap ${pct(sv.location_overlap, 1)}`),
    kpi("Needs overlap (Jaccard)", fmt3(sv.needs_jaccard), `urgent ≥4 F1 ${fmt3(sv.urgent_ge4_f1)}`),
  );
  const it = D.autoscientist || [], byRun = new Map();
  it.forEach(r => { if (!byRun.has(r.run_id)) byRun.set(r.run_id, { id: r.run_id, label: String(r.run).split(" · ").slice(0, 3).join(" · "), full: r.run, pts: [] });
    if (r.best_win_rate != null && !isNaN(r.best_win_rate)) byRun.get(r.run_id).pts.push({ it: r.iteration, w: +r.best_win_rate, lr: r.learning_rate, ep: r.n_epochs, r: r.lora_r }); });
  const runs = [...byRun.values()].filter(r => r.pts.length);
  if (runs.length) ladder($("#chIter"), runs, D.model.run_id);
  const cols = ["System", "Macro-F1 [95% CI]", "Weighted-F1", "Urgent recall", "Injured recall", "Valid JSON", "Latency / post", "$ / 1k posts", "n"];
  const lat = v => (v == null ? "—" : v < 0.01 ? `${(v * 1000).toFixed(2)} ms` : `${v.toFixed(2)} s`);
  $("#sysTable").replaceChildren(h("table", {}, h("thead", {}, h("tr", {}, cols.map(c => h("th", { text: c })))),
    h("tbody", {}, sorted.map(x => h("tr", { class: x === ours ? "ours" : "" },
      h("td", {}, x.label, h("span", { class: "kind", text: x.kind === "ours" ? "ours" : x.kind === "api" ? "API" : "baseline" })),
      h("td", { text: `${fmt3(x.macro_f1)} [${fmt3(x.ci[0])}, ${fmt3(x.ci[1])}]` }), h("td", { text: fmt3(x.weighted_f1) }),
      h("td", { text: fmt3(x.recall_urgent) }), h("td", { text: fmt3(x.recall_injured) }), h("td", { text: x.json_valid == null ? "—" : pct(x.json_valid, 1) }),
      h("td", { text: lat(x.latency_s) }), h("td", { text: x.usd_per_1k == null ? "—" : `$${x.usd_per_1k.toFixed(3)}` }), h("td", { text: x.n.toLocaleString() }))))));
  const partial = sys.filter(x => x.n !== ours.n);
  $("#tableNote").textContent = partial.length ? `${partial.map(x => x.label).join(", ")}: scored on the ${partial[0].n.toLocaleString()} posts it answered before its run was stopped.` : "";
}

// =====================================================================
// MODEL
// =====================================================================
function renderModel() {
  const D = S.dash || {}, m = D.model || {}, d = D.data || {}, hp = m.hyperparams || {}, lat = D.latency || {};
  const kv = pairs => h("dl", { class: "kv" }, pairs.flatMap(([k, v]) => [h("dt", { text: k }), h("dd", { text: v })]));
  const n = v => (v == null ? "—" : Number(v).toLocaleString());
  const steps = [
    ["Data", `HumAID, ${n(d.raw_rows)} human-labelled posts from 19 disasters`],
    ["Clean & split", "Exact + near-duplicate removal; split by disaster, not at random"],
    ["Silver fields", "Teacher LLM adds actionable / urgency / location / needs; human category kept"],
    ["AutoScientist", "Recipe search with a win-rate judge across several runs"],
    ["Select", "Best checkpoint chosen on validation posts, never on test"],
    ["Serve", "LoRA → GGUF, llama.cpp on CPU, $0 per post"],
  ];
  $("#modelGrid").replaceChildren(
    h("div", { class: "panel pipeline" }, h("h2", { text: "Pipeline" }), h("div", { class: "steps" }, steps.map(([a, b]) => h("div", { class: "step" }, h("b", { text: a }), b)))),
    h("div", { class: "panel" }, h("h2", { text: "Model" }), kv([
      ["Name", m.label || "CrisisLens"], ["Base", m.base || "Qwen/Qwen3.5-2B"], ["Adapter", hp.lora_r ? `LoRA r${hp.lora_r} / α${hp.lora_alpha}, all linear layers` : "LoRA"],
      ["Training", "Adaption Labs AutoScientist, instruction SFT"], ["Recipe", hp.learning_rate ? `lr ${hp.learning_rate}, ${hp.n_epochs} epochs, warmup ${hp.warmup_ratio}` : "—"],
      ["Train rows", m.train_rows ? n(m.train_rows) : "—"], ["Serving", "llama.cpp, Q4_K_M base + f16 adapter"],
      ["CPU latency", lat.latency_s_p50 ? `p50 ${lat.latency_s_p50.toFixed(1)} s · p95 ${lat.latency_s_p95.toFixed(1)} s (laptop)` : "—"]])),
    h("div", { class: "panel" }, h("h2", { text: "Data & evaluation" }), kv([
      ["Raw posts", n(d.raw_rows)], ["Train / val", `${n(d.train_rows)} / ${n(d.val_rows)} (15 disasters)`],
      ["Test", `${n(d.test_rows)} posts, 4 unseen disasters`], ["LLM eval set", "1,219 posts (300 per disaster + all missing/found)"],
      ["Test disasters", Object.values(EVENTS).join(", ")], ["Leakage guard", `${n((d.train_rows_removed_exact_overlap_test || 0) + (d.train_rows_removed_near_dup_of_test || 0))} train posts removed as (near-)duplicates of test`]])),
    h("div", { class: "panel" }, h("h2", { text: "Use & limits" }), h("ul", {},
      h("li", { text: "Decision support for response desks: ranks posts, a human acts." }),
      h("li", { text: "English posts; HumAID's 10 categories. Not a rumour or misinformation detector." }),
      h("li", { text: "Weakest on 'other relevant information' and on separating donation drives from urgent requests; it errs toward escalating." }),
      h("li", { text: "Urgency, location and needs were learned from teacher (silver) labels, not human annotation." }),
      h("li", { text: "HumAID is CC BY-NC-SA 4.0: non-commercial use." }))),
  );
}

boot();
