"use strict";
/* trade_X dashboard. Read-only. Everything from the ledger is put on the page as text nodes
   (never as markup), so agent-written text cannot inject markup. */

const TABS = [
  ["today", "Today"], ["positions", "Positions & exposure"], ["decisions", "Decisions"], ["performance", "Performance"],
  ["strategies", "Strategies"], ["system", "Accounts & system"], ["readiness", "Readiness"],
];
let META = { tz: "UTC" };
let current = "today";
let refreshTimer = null;

// ---- tiny DOM helpers -------------------------------------------------------------------
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
const $ = (s) => document.querySelector(s);
const dash = "-";
function num(v, d = 2) { return v === null || v === undefined ? dash : Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }); }
function usd(v, signed = false) {
  if (v === null || v === undefined) return dash;
  const s = Math.abs(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return (v < 0 ? "-" : signed && v > 0 ? "+" : "") + "$" + s;
}
function rr(v) { return v === null || v === undefined ? dash : (v > 0 ? "+" : "") + Number(v).toFixed(2) + "R"; }
function pct(v, d = 1) { return v === null || v === undefined ? dash : Number(v).toFixed(d) + "%"; }
function pct01(v) { return v === null || v === undefined ? dash : (v * 100).toFixed(0) + "%"; }
function when(iso, withDate = true) {
  if (!iso) return dash;
  const d = new Date(iso);
  if (isNaN(d)) return String(iso);
  const o = { timeZone: META.tz, hour: "2-digit", minute: "2-digit", hour12: false };
  if (withDate) Object.assign(o, { month: "short", day: "numeric" });
  return d.toLocaleString("en-GB", o);
}
function ago(iso) {
  if (!iso) return "never";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (isNaN(s)) return dash;
  if (s < 90) return "just now";
  if (s < 5400) return Math.round(s / 60) + " min ago";
  if (s < 172800) return Math.round(s / 3600) + " h ago";
  return Math.round(s / 86400) + " days ago";
}
function tone(v) { return v > 0 ? "good" : v < 0 ? "bad" : ""; }
function arrow(v) { return v > 0 ? "▲ " : v < 0 ? "▼ " : ""; }
function pill(text, kind) { return h("span", { class: "pill " + (kind || "") }, text); }
function empty(msg) { return h("div", { class: "empty" }, msg); }
function card(label, big, sub, cls) {
  return h("div", { class: "card" }, h("div", { class: "label" }, label), h("div", { class: "big num " + (cls || "") }, big), sub ? h("div", { class: "sub" }, sub) : null);
}
function table(cols, rows, onclick) {
  const head = h("tr", null, cols.map((c) => h("th", { class: c.r ? "r" : "" }, c.t)));
  const body = rows.map((row) => {
    const tr = h("tr", { class: onclick ? "click" : "", tabindex: onclick ? "0" : null },
      cols.map((c) => h("td", { class: (c.r ? "r " : "") + "num" }, c.f(row))));
    if (onclick) {
      tr.addEventListener("click", () => onclick(row));
      tr.addEventListener("keydown", (e) => { if (e.key === "Enter") onclick(row); });
    }
    return tr;
  });
  return h("div", { class: "tablewrap" }, h("table", null, h("thead", null, head), h("tbody", null, body)));
}
async function api(path) {
  const r = await fetch(path, { headers: { Accept: "application/json" } });
  if (!r.ok) throw new Error(path + " -> " + r.status);
  return r.json();
}
function outcomePill(o) {
  const m = { closed: ["Closed", "info"], open: ["Open", "good"], blocked: ["Blocked", "warn"], declined: ["Declined by risk", "warn"], accepted: ["Accepted, waiting to fill", "info"], pending: ["Pending", ""] };
  const [t, k] = m[o] || [o, ""];
  return pill(t, k);
}

// ---- shell ------------------------------------------------------------------------------
function buildTabs() {
  const nav = $("#tabs");
  nav.replaceChildren(...TABS.map(([id, label]) => h("button", { role: "tab", id: "tab-" + id, "aria-selected": String(id === current), onclick: () => go(id) }, label)));
}
function go(id) { location.hash = id; }
async function show() {
  const [id, sub] = (location.hash || "#today").slice(1).split("/");
  current = TABS.some((t) => t[0] === id) ? id : "today";
  buildTabs();
  await render();
  if (current === "decisions" && sub) openDecision(decodeURIComponent(sub));   // deep link: #decisions/2026-10-06-0001
}
async function render() {
  const view = $("#view");
  try {
    META = await api("/api/meta");
    const b = $("#banner");
    b.hidden = !META.banner;
    b.textContent = META.banner || "";
    const node = await RENDER[current]();
    view.replaceChildren(node);
  } catch (e) {
    view.replaceChildren(empty("Could not load this page: " + e.message));
  }
}
function scheduleRefresh() {
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(() => { if (!$("#drawer").hidden) return; render(); }, 400);
}
function connect() {
  const live = $("#live"), txt = $("#live-text");
  if (new URLSearchParams(location.search).has("static")) { txt.textContent = "static"; return; }   // ?static: no live refresh (screenshots, printing)
  if (!window.EventSource) return poll();
  const es = new EventSource("/api/stream");
  es.addEventListener("change", () => { live.classList.add("on"); txt.textContent = "live"; scheduleRefresh(); });
  es.onopen = () => { live.classList.add("on"); txt.textContent = "live"; };
  es.onerror = () => { live.classList.remove("on"); txt.textContent = "reconnecting"; };
}
function poll() { setInterval(render, 15000); $("#live-text").textContent = "polling"; }

// ---- Today ------------------------------------------------------------------------------
async function today() {
  const d = await api("/api/today");
  const root = h("div");
  root.append(h("h1", null, d.headline || "Today"));
  if (d.empty) { root.append(empty(d.reason || "Nothing recorded yet.")); return root; }
  const e = d.equity, r = d.risk, c = d.closed_today;
  root.append(h("div", { class: "grid" },
    card("Account value", usd(e.now), e.as_of ? "as of " + when(e.as_of) : "no snapshot yet"),
    card("Change since yesterday", e.since_yesterday === null ? dash : arrow(e.since_yesterday) + usd(e.since_yesterday, true), e.since_yesterday === null ? "needs a snapshot from before today" : null, tone(e.since_yesterday)),
    card("Closed today", c.trades + (c.trades === 1 ? " trade" : " trades"), c.trades ? usd(c.net_pnl_usd, true) + " / " + rr(c.total_r) : "none yet", tone(c.net_pnl_usd)),
    card("Risk in use", usd(r.open_risk_usd), r.heat_cap_usd ? "limit " + usd(r.heat_cap_usd) + " / " + r.open_positions + " open" : r.open_positions + " open"),
  ));
  const k = d.counts;
  root.append(h("div", { class: "grid" },
    card("Plans today", k.plan, k.veto + " blocked, " + k.declined + " declined by risk"),
    card("Risk gate accepted", k.accepted, k.fill + " fills"),
    card("Mode", r.paused ? "Paused" : "Active", "tier " + (r.tier ?? dash) + (META.agents_mode ? " / agents: " + META.agents_mode : "")),
  ));
  if (d.attention.length) {
    root.append(h("div", { class: "section" }, h("h2", null, "Needs a look"),
      h("ul", { class: "list" }, d.attention.map((a) => h("li", { class: a.level === "fault" ? "bad" : "" }, pill(a.level === "fault" ? "Fault" : "Note", a.level === "fault" ? "bad" : "info"), h("span", null, a.text))))));
  }
  root.append(h("div", { class: "section" }, h("h2", null, "What happened today")));
  if (!d.timeline.length) root.append(empty("No ledger activity yet today."));
  else root.append(h("ul", { class: "list" }, d.timeline.map((t) => {
    const li = h("li", { class: t.bad ? "bad" : "" }, h("span", { class: "t num" }, when(t.time, false)), h("span", null, t.text));
    if (t.decision_id) li.append(h("button", { class: "linkish", onclick: () => openDecision(t.decision_id) }, t.decision_id));
    return li;
  })));
  return root;
}

// ---- Positions & exposure ------------------------------------------------------------------
async function positions() {
  const at = sessionStorageGet("lookback");
  const d = await api("/api/positions" + (at ? "?at=" + encodeURIComponent(at) : ""));
  const root = h("div", null, h("h1", null, "Positions & exposure"));
  const input = h("input", { type: "datetime-local", "aria-label": "Look back to", value: at || "" });
  input.addEventListener("change", () => { sessionStorageSet("lookback", input.value ? new Date(input.value).toISOString() : ""); render(); });
  root.append(h("div", { class: "filters" }, h("span", { class: "muted" }, "Look back to"), input,
    at ? h("button", { class: "btn", onclick: () => { sessionStorageSet("lookback", ""); render(); } }, "Back to now") : null));
  if (d.empty) { root.append(empty(d.reason)); return root; }
  root.append(h("p", { class: "muted" }, (d.looking_back ? "Snapshot at or before the chosen time: " : "Latest snapshot: ") + when(d.as_of) + " (" + ago(d.as_of) + ")"));
  const hp = d.heat;
  root.append(h("div", { class: "grid" },
    card("Account value", usd(d.equity_usd), "cash " + usd(d.cash_usd)),
    h("div", { class: "card" }, h("div", { class: "label" }, "Risk budget used"), h("div", { class: "big num" }, hp.used_pct === null ? usd(hp.open_risk_usd) : pct(hp.used_pct, 0)),
      h("div", { class: "sub" }, usd(hp.open_risk_usd) + (hp.cap_usd ? " of " + usd(hp.cap_usd) : "")),
      hp.used_pct === null ? null : h("div", { class: "bar" }, h("i", { class: hp.used_pct > 80 ? "hot" : "", style: "width:" + Math.min(100, hp.used_pct) + "%" }))),
    card("Governor tier", d.limits.tier ?? dash, d.limits.paused ? "paused" : "entries allowed"),
  ));
  root.append(h("h2", null, "Open positions"));
  if (!d.positions.length) root.append(empty("No open positions."));
  else root.append(table([
    { t: "Market", f: (p) => p.symbol_label },
    { t: "Side", f: (p) => p.side },
    { t: "Size", r: 1, f: (p) => num(p.qty, 0) },
    { t: "Entry", r: 1, f: (p) => num(p.entry, 5) },
    { t: "Now", r: 1, f: (p) => (p.mark === null ? dash : num(p.mark, 5)) },
    { t: "Stop", r: 1, f: (p) => (p.has_stop ? num(p.stop, 5) : pill("NO STOP", "bad")) },
    { t: "Target", r: 1, f: (p) => (p.target === null ? dash : num(p.target, 5)) },
    { t: "Progress", r: 1, f: (p) => h("span", { class: tone(p.unrealised_r) }, p.unrealised_r === null ? dash : arrow(p.unrealised_r) + rr(p.unrealised_r)) },
    { t: "Decision", f: (p) => h("button", { class: "linkish", onclick: () => openDecision(p.decision_id) }, p.decision_id) },
  ], d.positions));
  root.append(h("div", { class: "section" }, h("h2", null, "Net exposure by currency"),
    h("p", { class: "muted" }, "US dollars at risk to each currency, long minus short, across open positions.")));
  if (!d.exposure.length) root.append(empty("No exposure."));
  else {
    const mx = Math.max(...d.exposure.map((e) => Math.abs(e.usd)), 1);
    root.append(table([
      { t: "Currency", f: (e) => e.currency },
      { t: "Net USD", r: 1, f: (e) => h("span", { class: tone(e.usd) }, usd(e.usd, true)) },
      { t: "", f: (e) => h("div", { class: "bar", style: "min-width:120px" }, h("i", { style: "width:" + (Math.abs(e.usd) / mx * 100) + "%" })) },
    ], d.exposure));
  }
  return root;
}

// small helpers for per-viewer conveniences (storage may be blocked)
function sessionStorageGet(k) { try { return sessionStorage.getItem("tradex." + k) || ""; } catch (e) { return ""; } }
function sessionStorageSet(k, v) { try { sessionStorage.setItem("tradex." + k, v); } catch (e) { /* ignore */ } }

// ---- Decisions --------------------------------------------------------------------------------
async function decisions() {
  const f = sessionStorageGet("dec_outcome");
  const d = await api("/api/decisions?limit=150" + (f ? "&outcome=" + encodeURIComponent(f) : ""));
  const sel = h("select", { "aria-label": "Filter by outcome" }, [["", "All outcomes"], ["closed", "Closed"], ["open", "Open"], ["blocked", "Blocked"], ["declined", "Declined by risk"], ["accepted", "Accepted"]].map(([v, t]) => h("option", { value: v, selected: v === f }, t)));
  sel.addEventListener("change", () => { sessionStorageSet("dec_outcome", sel.value); render(); });
  const root = h("div", null, h("h1", null, "Decisions"), h("div", { class: "filters" }, sel),
    h("p", { class: "muted" }, "Every plan the system made, including the ones it did not take. Click a row for the full reasoning."));
  if (d.empty) { root.append(empty(d.reason || "No decisions yet.")); return root; }
  root.append(table([
    { t: "Time", f: (x) => when(x.time) }, { t: "Market", f: (x) => x.side + " " + x.symbol_label },
    { t: "Outcome", f: (x) => outcomePill(x.outcome) },
    { t: "Expected", r: 1, f: (x) => rr(x.ev_r) },
    { t: "Result", r: 1, f: (x) => (x.r_multiple === null ? (x.counterfactual_r === null ? dash : h("span", { class: "muted" }, "would have been " + rr(x.counterfactual_r))) : h("span", { class: tone(x.r_multiple) }, rr(x.r_multiple))) },
    { t: "Why", f: (x) => x.why || (x.strategies || []).join(", ") },
    { t: "ID", f: (x) => x.decision_id },
  ], d.decisions, (x) => openDecision(x.decision_id)));
  return root;
}

// ---- Decision drawer -----------------------------------------------------------------------------
let lastFocus = null;
async function openDecision(id) {
  lastFocus = document.activeElement;
  const body = $("#drawer-body");
  $("#drawer-title").textContent = "Decision " + id;
  body.replaceChildren(h("p", { class: "muted" }, "Loading..."));
  $("#drawer").hidden = false; $("#scrim").hidden = false;
  $("#drawer-close").focus();
  try {
    const d = await api("/api/decisions/" + encodeURIComponent(id));
    body.replaceChildren(drawerContent(d));
    loadCandles(d);
  } catch (e) { body.replaceChildren(empty("Could not load this decision: " + e.message)); }
}
function closeDrawer() { $("#drawer").hidden = true; $("#scrim").hidden = true; if (lastFocus && lastFocus.focus) lastFocus.focus(); }
function kv(pairs) {
  return h("dl", { class: "kv" }, pairs.filter((p) => p[1] !== undefined).flatMap(([k, v]) => [h("dt", null, k), h("dd", { class: "num" }, v === null ? dash : v)]));
}
function drawerContent(d) {
  const p = d.plan, root = h("div");
  root.append(h("p", null, outcomePill(d.outcome), " ", h("strong", null, d.side + " " + d.symbol_label), " ", h("span", { class: "muted" }, when(p.time))));
  root.append(h("div", { id: "candles", class: "chart" }, h("p", { class: "muted pad" }, "Loading chart...")));
  root.append(h("h3", null, "The plan"));
  root.append(kv([["Entry", num(p.entry_price, 5) + " (" + p.entry_type + ")"], ["Stop", num(p.stop, 5)], ["Targets", p.targets.map((t) => num(t, 5)).join(", ")],
    ["Time limit", p.max_bars + " bars"], ["Expected value", rr(p.ev_r) + " after costs"], ["Reward / risk", num(p.reward_risk, 2)],
    ["Win chance used", pct01(p.p_target) + " (" + p.p_source.replace("_", " ") + ")"], ["Cost", rr(p.cost_r)], ["Invalidation", p.invalidation]]));
  root.append(h("h3", null, "What the strategies said (" + d.votes.length + ")"));
  root.append(d.votes.length ? table([
    { t: "Strategy", f: (v) => v.strategy_id }, { t: "Family", f: (v) => v.family },
    { t: "View", f: (v) => (v.direction > 0 ? "long" : v.direction < 0 ? "short" : "flat") }, { t: "Strength", r: 1, f: (v) => pct01(v.strength) },
  ], d.votes) : h("p", { class: "muted" }, "No vote rows are linked to this plan (families: " + (p.families || []).join(", ") + ")."));
  root.append(h("h3", null, "Could anything stop it?"));
  root.append(d.vetoes.length ? h("ul", { class: "list" }, d.vetoes.map((v) => h("li", null, pill("Blocked", "warn"), h("span", null, v.source + ": " + v.reason)))) : h("p", { class: "muted" }, "No context check blocked this plan."));
  root.append(h("h3", null, "Risk checks"));
  if (!d.verdict) root.append(h("p", { class: "muted" }, "The risk gate has not answered."));
  else {
    root.append(h("p", null, d.verdict.outcome === "accepted" ? pill("Accepted", "good") : pill("Declined", "bad"), " size ", num(d.verdict.qty, 0), ", risking ", usd(d.verdict.risk_usd), " (", pct(d.verdict.risk_pct, 2), " of account)"));
    if (d.verdict.reasons.length) root.append(h("ul", { class: "list" }, d.verdict.reasons.map((r) => h("li", null, r))));
    for (const c of d.checks) root.append(h("div", { class: "card", style: "margin-bottom:8px" }, h("strong", null, c.name.replace(/_/g, " ")), checkBody(c.detail)));
  }
  root.append(h("h3", null, "Agent opinions"));
  root.append(d.agent_opinions.length ? h("div", null, d.agent_opinions.map((o) => h("div", { class: "opinion" }, h("strong", null, o.agent + " " + o.action), " ", pill(o.status, o.status === "applied" ? "good" : ""), h("div", { class: "muted" }, summarizeBody(o.body))))) : h("p", { class: "muted" }, "No agent weighed in on this decision."));
  root.append(h("h3", null, "What happened"));
  root.append(h("ul", { class: "list" }, d.timeline.map((t) => h("li", null, h("span", { class: "t num" }, when(t.time)), h("span", null, t.text)))));
  if (d.counterfactual) root.append(h("p", { class: "muted" }, "If it had not been blocked by " + d.counterfactual.blocked_by + ", it would have ended at " + rr(d.counterfactual.r_multiple) + " (" + d.counterfactual.exit_reason + ")."));
  root.append(h("p", { class: "muted" }, "Recorded on code version " + (d.git_commit || dash) + ", rules " + (d.config_hash || dash) + "."));
  return root;
}
function checkBody(detail) {
  return kv(Object.entries(detail).map(([k, v]) => [k.replace(/_/g, " "), typeof v === "object" && v !== null ? JSON.stringify(v) : typeof v === "number" ? String(Math.round(v * 10000) / 10000) : String(v)]));
}
function summarizeBody(b) {
  if (!b) return "";
  const parts = [];
  if (b.result) parts.push(b.result);
  if (b.reason) parts.push(b.reason);
  if (b.would) parts.push("would " + b.would + (b.factor ? " x" + b.factor : ""));
  if (b.request && b.request.reason) parts.push(b.request.reason);
  return parts.join(" / ");
}
async function loadCandles(d) {
  const box = $("#candles");
  if (!box) return;
  try {
    const tf = d.plan.tf || "D1";
    const b = await api("/api/bars/" + encodeURIComponent(d.symbol) + "?tf=" + encodeURIComponent(tf) + "&decision_id=" + encodeURIComponent(d.decision_id));
    if (b.empty || !window.LightweightCharts) {
      box.replaceChildren(h("div", { class: "pad" }, h("p", { class: "muted" }, b.empty ? b.reason : "The chart library did not load (offline?). Price levels below."),
        b.levels && b.levels.length ? kv(b.levels.map((l) => [l.label, num(l.price, 5)])) : null));
      return;
    }
    box.replaceChildren();
    const css = getComputedStyle(document.documentElement);
    const col = (n) => css.getPropertyValue(n).trim();
    const chart = LightweightCharts.createChart(box, { height: 280, layout: { background: { color: col("--surface") }, textColor: col("--text-2") },
      grid: { vertLines: { color: col("--border") }, horzLines: { color: col("--border") } }, timeScale: { timeVisible: tf !== "D1" } });
    const s = chart.addCandlestickSeries({ upColor: col("--good"), downColor: col("--bad"), wickUpColor: col("--good"), wickDownColor: col("--bad"), borderVisible: false });
    s.setData(b.bars);
    for (const l of b.levels) s.createPriceLine({ price: l.price, color: l.label === "stop" ? col("--bad") : l.label === "entry" ? col("--accent") : col("--good"), lineWidth: 1, lineStyle: 2, title: l.label });
    const ms = b.markers.filter((m) => m.unix).map((m) => ({ time: m.unix, position: "aboveBar", shape: "circle", color: col("--accent"), text: m.label })).sort((a, c) => a.time - c.time);
    try { s.setMarkers(ms); } catch (e) { /* markers must fall on bar times; skip if not */ }
    chart.timeScale().fitContent();
  } catch (e) { box.replaceChildren(h("p", { class: "muted pad" }, "Chart unavailable: " + e.message)); }
}

// ---- Performance -----------------------------------------------------------------------------------
function lineChart(points, fmt) {
  const W = 800, H = 220, P = 34;
  const ys = points.map((p) => p.equity);
  let lo = Math.min(...ys), hi = Math.max(...ys);
  if (lo === hi) { lo -= 1; hi += 1; }
  const x = (i) => P + (points.length === 1 ? 0 : i / (points.length - 1)) * (W - P - 10);
  const y = (v) => H - 24 - ((v - lo) / (hi - lo)) * (H - 40);
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("preserveAspectRatio", "none"); svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Account value over time");
  const mk = (t, a) => { const e = document.createElementNS(ns, t); for (const k in a) e.setAttribute(k, a[k]); svg.append(e); return e; };
  for (const v of [lo, (lo + hi) / 2, hi]) { mk("line", { x1: P, x2: W - 10, y1: y(v), y2: y(v), stroke: "var(--border)", "stroke-width": 1 }); const t = mk("text", { x: 4, y: y(v) + 4, fill: "var(--muted)", "font-size": 11 }); t.textContent = Math.round(v).toLocaleString(); }
  mk("polyline", { points: points.map((p, i) => x(i) + "," + y(p.equity)).join(" "), fill: "none", stroke: "var(--accent)", "stroke-width": 2, "stroke-linejoin": "round" });
  const cross = mk("line", { y1: 10, y2: H - 24, stroke: "var(--muted)", "stroke-width": 1, visibility: "hidden" });
  const wrap = h("div", { class: "svgwrap" }, svg);
  const tip = h("div", { class: "tip", hidden: true });
  wrap.append(tip);
  svg.addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect();
    const i = Math.max(0, Math.min(points.length - 1, Math.round(((e.clientX - r.left) / r.width * W - P) / (W - P - 10) * (points.length - 1))));
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
    tip.hidden = false; tip.textContent = points[i].day + ": " + usd(points[i].equity);
    tip.style.left = Math.min(r.width - 150, Math.max(8, (x(i) / W) * r.width + 10)) + "px";
  });
  svg.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); tip.hidden = true; });
  return wrap;
}
async function performance() {
  const d = await api("/api/performance");
  const root = h("div", null, h("h1", null, "Performance"));
  if (d.empty) {
    root.append(empty(d.reason || "No results yet."));
    if (d.filters && d.filters.length) root.append(filtersTable(d.filters));
    return root;
  }
  const s = d.stats;
  if (!d.enough) root.append(h("div", { class: "banner" }, "Only " + s.trades + (s.trades === 1 ? " closed trade" : " closed trades") + " so far. Below " + d.min_trades + " the win rate and averages are mostly noise; treat them as a first look."));
  root.append(h("div", { class: "grid", style: "margin-top:12px" },
    card("Net result", usd(s.net_pnl_usd, true), s.trades + (s.trades === 1 ? " closed trade" : " closed trades") + ", after costs", tone(s.net_pnl_usd)),
    card("Win rate", pct01(s.win_rate), s.wins + " of " + s.trades + " were winners"),
    card("Average trade", rr(s.avg_r), "total " + rr(s.total_r), tone(s.avg_r)),
    card("Profit factor", s.profit_factor === null ? dash : num(s.profit_factor, 2), "winnings / losses"),
    card("Worst dip", d.max_drawdown_pct === null ? dash : pct(d.max_drawdown_pct), "peak-to-trough, daily snapshots", tone(d.max_drawdown_pct)),
    card("Costs paid", usd(d.costs_usd), "fees + spread + slippage"),
  ));
  root.append(h("h2", null, "Account value"));
  root.append(d.equity.length >= 2 ? lineChart(d.equity) : empty("Needs at least two daily snapshots."));
  root.append(h("h2", { class: "section" }, "How big were the wins and losses?"));
  const mx = Math.max(...d.r_hist.counts, 1), ed = d.r_hist.edges;
  const lab = (i) => (i === 0 ? "below " + ed[0] + "R" : i === ed.length ? ed[ed.length - 1] + "R and up" : ed[i - 1] + "R to " + ed[i] + "R");
  root.append(h("div", { class: "hist", role: "img", "aria-label": "Distribution of trade results in R" },
    d.r_hist.counts.map((c, i) => h("i", { class: i < 6 ? "neg" : i > 6 ? "pos" : "", style: "height:" + (c / mx * 100) + "%", title: lab(i) + ": " + c + " trades" }))));
  root.append(h("div", { class: "muted", style: "display:flex;justify-content:space-between;font-size:12px" }, h("span", null, "-3R or worse"), h("span", null, "0"), h("span", null, "+3R or better")));
  if (d.by_asset_class.length) {
    root.append(h("h2", { class: "section" }, "By market"));
    root.append(table([{ t: "Market", f: (x) => x.asset_class }, { t: "Trades", r: 1, f: (x) => x.trades }, { t: "Win rate", r: 1, f: (x) => pct01(x.win_rate) },
      { t: "Avg", r: 1, f: (x) => rr(x.avg_r) }, { t: "Net", r: 1, f: (x) => usd(x.net_pnl_usd, true) }], d.by_asset_class));
  }
  root.append(h("h2", { class: "section" }, "Latest closed trades"));
  root.append(table([{ t: "Closed", f: (x) => when(x.time) }, { t: "Market", f: (x) => x.symbol_label }, { t: "Result", r: 1, f: (x) => h("span", { class: tone(x.r) }, rr(x.r)) },
    { t: "P&L", r: 1, f: (x) => usd(x.pnl, true) }, { t: "Exit", f: (x) => x.reason }], d.recent, (x) => openDecision(x.decision_id)));
  root.append(h("h2", { class: "section" }, "What each filter saved or cost"));
  root.append(filtersTable(d.filters));
  return root;
}
function filtersTable(rows) {
  if (!rows.length) return empty("No blocked plans have been followed yet.");
  return table([{ t: "Blocked by", f: (x) => x.blocked_by }, { t: "Plans", r: 1, f: (x) => x.plans }, { t: "They would have averaged", r: 1, f: (x) => h("span", { class: tone(x.mean_r) }, rr(x.mean_r)) },
    { t: "Would have won", r: 1, f: (x) => pct01(x.win_rate) }], rows);
}

// ---- Strategies ------------------------------------------------------------------------------------------
async function strategies() {
  const d = await api("/api/strategies");
  const root = h("div", null, h("h1", null, "Strategies"),
    h("p", { class: "muted" }, "Backtest = out-of-sample walk-forward result from the research reports. Forward = what the strategy did on its own paper book since. Under " + d.min_trades + " forward trades is too early to judge."));
  if (d.empty) { root.append(empty(d.reason)); return root; }
  root.append(table([
    { t: "Strategy", f: (s) => s.id }, { t: "Stage", f: (s) => pill(s.status, s.status === "live" || s.status === "paper" ? "good" : s.status === "validated" ? "info" : "") },
    { t: "Backtest trades", r: 1, f: (s) => (s.backtest ? s.backtest.trades ?? dash : dash) },
    { t: "Backtest avg", r: 1, f: (s) => (s.backtest ? rr(s.backtest.expectancy_r) : dash) },
    { t: "Backtest win", r: 1, f: (s) => (s.backtest ? pct01(s.backtest.win_rate) : dash) },
    { t: "Forward trades", r: 1, f: (s) => s.forward.trades },
    { t: "Forward avg", r: 1, f: (s) => rr(s.forward.avg_r) },
    { t: "Forward win", r: 1, f: (s) => pct01(s.forward.win_rate) },
    { t: "Gap", r: 1, f: (s) => (s.delta_avg_r === null ? dash : h("span", { class: s.too_early ? "muted" : tone(s.delta_avg_r) }, rr(s.delta_avg_r))) },
    { t: "Read", f: (s) => (s.forward.trades === 0 ? pill("no forward trades", "") : s.too_early ? pill("too early", "warn") : s.delta_avg_r === null ? pill("no backtest", "") : s.delta_avg_r >= 0 ? pill("at or above backtest", "good") : pill("below backtest", "bad")) },
  ], d.strategies));
  return root;
}

// ---- Accounts & system --------------------------------------------------------------------------------------
async function system() {
  const d = await api("/api/system");
  const root = h("div", null, h("h1", null, "Accounts & system"));
  root.append(h("h2", null, "Trading accounts"));
  root.append(d.accounts.length ? table([{ t: "Account", f: (a) => a.name }, { t: "Venue", f: (a) => a.venue }, { t: "Kind", f: (a) => pill(a.environment + " (" + a.mode + ")", "info") },
    { t: "Markets", f: (a) => (a.asset_classes || []).join(", ") }], d.accounts) : empty("No accounts file found."));
  root.append(h("p", { class: "muted" }, "Account numbers are never shown here. Phase 1 trades practice and simulated accounts only."));
  if (d.empty) { root.append(empty(d.reason || "No ledger yet.")); }
  else {
    const L = d.ledger, ch = L.chain;
    root.append(h("div", { class: "grid section" },
      card("Ledger", L.last_seq + " rows", "last row " + ago(L.last_event_time) + " / " + (L.size_bytes / 1048576).toFixed(1) + " MB"),
      card("Tamper check", ch.ok === null ? dash : ch.ok ? "Intact" : "BROKEN", ch.ok ? "hash chain verified" : ch.ok === false ? "first bad row " + ch.bad_seq : "", ch.ok === false ? "bad" : ch.ok ? "good" : ""),
      card("Agent mode", d.agents_mode || dash, d.agents_mode === "shadow" ? "agent requests are recorded, not applied" : null),
      card("Last Telegram alert", d.telegram_last_alert ? ago(d.telegram_last_alert) : "none sent", null)));
    root.append(h("h2", null, "Venues (from ledger rows)"));
    root.append(table([{ t: "Venue", f: (v) => v.venue }, { t: "Last fill", f: (v) => (v.last_fill ? when(v.last_fill) + " (" + ago(v.last_fill) + ")" : "none yet") }, { t: "Closed trades", r: 1, f: (v) => v.closed_trades }], d.venues));
    root.append(h("h2", { class: "section" }, "Health checks"));
    root.append(d.health.length ? h("ul", { class: "list" }, d.health.map((x) => h("li", { class: x.ok ? "" : "bad" }, pill(x.ok ? "OK" : "FAULT", x.ok ? "good" : "bad"), h("span", null, x.check + (x.detail ? ": " + x.detail : "")), h("span", { class: "t" }, ago(x.time))))) : empty("No health rows yet."));
    const g = d.gateway;
    root.append(h("h2", { class: "section" }, "Model gateway, last 24 hours"));
    root.append(g.calls_24h ? h("div", null, h("div", { class: "grid" }, card("Calls", g.calls_24h, g.failed_24h + " failed"), card("Median wait", g.median_latency_ms === null ? dash : g.median_latency_ms + " ms"), card("Tokens", (g.tokens_in + g.tokens_out).toLocaleString(), g.tokens_in.toLocaleString() + " in / " + g.tokens_out.toLocaleString() + " out")),
      table([{ t: "Who actually answered", f: (x) => x.who }, { t: "Calls", r: 1, f: (x) => x.calls }], g.answered_by)) : empty("No agent calls in the last 24 hours."));
    root.append(h("h2", { class: "section" }, "Commands"));
    root.append(d.commands.length ? table([{ t: "When", f: (c) => when(c.time) }, { t: "From", f: (c) => c.source }, { t: "Command", f: (c) => c.command }, { t: "Result", f: (c) => (c.applied_at ? c.result : pill("waiting", "warn")) }], d.commands) : empty("No commands sent."));
    if (d.jobs.length) root.append(h("div", null, h("h2", { class: "section" }, "Agent jobs"), table([{ t: "Agent", f: (j) => j.agent }, { t: "Status", f: (j) => j.status }, { t: "Count", r: 1, f: (j) => j.n }], d.jobs)));
    if (d.recent_faults && d.recent_faults.length) root.append(h("div", null, h("h2", { class: "section" }, "Recent faults"), h("ul", { class: "list" }, d.recent_faults.map((f) => h("li", { class: "bad" }, h("span", { class: "t" }, when(f.time)), h("span", null, f.check + ": " + f.detail))))));
  }
  root.append(h("h2", { class: "section" }, "Backups and dead-man ping"));
  root.append(table([{ t: "Check", f: (o) => o.label }, { t: "Last", f: (o) => (o.last ? when(o.last) + " (" + ago(o.last) + ")" : "no stamp found") },
    { t: "Status", f: (o) => pill(o.status === "ok" ? "OK" : o.status === "stale" ? "STALE" : "unknown", o.status === "ok" ? "good" : o.status === "stale" ? "bad" : "") }], d.ops));
  return root;
}

// ---- Readiness ------------------------------------------------------------------------------------------
async function readiness() {
  const d = await api("/api/readiness");
  const root = h("div", null, h("h1", null, d.empty ? "Readiness" : d.ready ? "Every check passes" : d.passing + " of " + d.total + " go-live checks pass"));
  if (d.empty) { root.append(empty(d.reason)); return root; }
  root.append(h("p", { class: "muted" }, d.note + " Unknown means there is no real evidence yet, which is not the same as failing."));
  root.append(h("ul", { class: "list" }, d.criteria.map((c) => h("li", null,
    pill(c.status === "pass" ? "Pass" : c.status === "fail" ? "Fail" : "Unknown", c.status === "pass" ? "good" : c.status === "fail" ? "bad" : ""),
    h("span", null, c.description, c.id === "ray_go_ahead" ? h("div", { class: "muted" }, "This page is read-only in v1, so the go-ahead cannot be given from here yet.") : null, h("div", { class: "muted" }, c.status === "unknown" ? c.detail + (c.ignored ? " (" + c.ignored + " non-real item(s) ignored)" : "") : "now " + num(c.value, 0) + ", needs " + (c.threshold.min !== undefined ? "at least " + c.threshold.min : "at most " + c.threshold.max)))))));
  return root;
}

const RENDER = { today, positions, decisions, performance, strategies, system, readiness };

// ---- start ----------------------------------------------------------------------------------------------
$("#drawer-close").addEventListener("click", closeDrawer);
$("#scrim").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("#drawer").hidden) closeDrawer(); });
window.addEventListener("hashchange", show);
show().then(connect);
