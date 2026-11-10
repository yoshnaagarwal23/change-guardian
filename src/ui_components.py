"""HTML building blocks for the Streamlit console (pure functions, unit-tested).

Visual references: NVIDIA Brev (dark grid, mono labels, lime "go" color), metopipharma.com
(navy-black + warm amber), kamoku.tech (terminal-style security tone), MagicUI (subtle glow
and pulse effects, recreated in CSS). Icons: Font Awesome Free 6.7.2 (CC BY 4.0), inlined
from static/icons so the demo works offline. All alert text is HTML-escaped.
"""
from __future__ import annotations

from functools import lru_cache
from html import escape
from pathlib import Path

from alert_trust import badge

ICONS = Path(__file__).resolve().parents[1] / "static" / "icons"

AMBER, GREEN, RED, BLUE, MUTED = "#FFB224", "#7BD88F", "#F25F5C", "#5AA9E6", "#8B98A5"
SEV_COLOR = {"SEV-1": RED, "SEV-2": AMBER, "SEV-3": "#FFD166", "SEV-4": BLUE, "SEV-5": MUTED}
PRIORITY_KIND = {"Critical": "crit", "High": "high", "Medium": "med", "Low": "low"}
RISK = {"diagnostic": ("magnifying-glass", "diagnostic", "ok"),
        "change": ("wrench", "change", "high"),
        "security": ("user-shield", "security", "crit")}

CSS = """
<style>
.stApp {
  background:
    radial-gradient(900px 420px at 12% -8%, rgba(255,178,36,.10), transparent 60%),
    radial-gradient(700px 380px at 95% 0%, rgba(90,169,230,.07), transparent 60%),
    linear-gradient(rgba(255,255,255,.022) 1px, transparent 1px) 0 0/44px 44px,
    linear-gradient(90deg, rgba(255,255,255,.022) 1px, transparent 1px) 0 0/44px 44px,
    #0A0F14;
}
header[data-testid="stHeader"] {background: transparent;}
.ic {display:inline-block; width:1em; height:1em; flex:none; vertical-align:-0.125em;}
.block-container {padding-top: 1.4rem; max-width: 1480px;}
.cg-eyebrow {font: 600 10.5px JetBrainsMono, monospace; letter-spacing: .08em; text-transform: uppercase;
  color: #8B98A5; margin: 0 0 8px 0; display:flex; align-items:center; gap:7px; min-width:0;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.cg-card {background: rgba(17,24,33,.82); border: 1px solid #1E2A36; border-radius: 14px;
  padding: 16px 18px; backdrop-filter: blur(6px); transition: border-color .2s ease;}
.cg-card:hover {border-color: #2B3B4C;}
.cg-hero {display:flex; justify-content:space-between; align-items:flex-end; gap:24px; margin-bottom: 6px;}
.cg-title {font-size: 2.1rem; font-weight: 750; letter-spacing: -.02em; margin: 0; line-height: 1.1;
  background: linear-gradient(90deg, #FFFFFF 0%, #FFE2A8 55%, #FFB224 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;}
.cg-sub {color:#8B98A5; margin: 6px 0 0 0; font-size: .95rem;}
.cg-status {display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end;}
.cg-dot {width:8px; height:8px; border-radius:50%; background:#7BD88F; box-shadow:0 0 0 0 rgba(123,216,143,.6);
  animation: cg-pulse 2s infinite; display:inline-block;}
@keyframes cg-pulse {0%{box-shadow:0 0 0 0 rgba(123,216,143,.55)} 70%{box-shadow:0 0 0 8px rgba(123,216,143,0)}
  100%{box-shadow:0 0 0 0 rgba(123,216,143,0)}}
.chip {display:inline-flex; align-items:center; gap:5px; padding:1px 8px; border-radius:6px; line-height:18px;
  font: 500 11px Inter, sans-serif; border:1px solid #243140; color:#AAB6C2; background:transparent;
  white-space:nowrap; max-width:100%; overflow:hidden; text-overflow:ellipsis; vertical-align:middle;}
.chip .ic {width:11px; height:11px;}
.chip[title] {cursor:help;}
.chip.mono {font-family: JetBrainsMono, monospace; font-size: 10.5px;}
/* Only "crit" is filled: strong color is reserved for what needs attention. */
.chip.crit {color:#FF8A87; border-color:rgba(242,95,92,.45); background:rgba(242,95,92,.10);}
.chip.high {color:#F2BE63; border-color:rgba(255,178,36,.30);}
.chip.med  {color:#D9C68E; border-color:rgba(255,209,102,.22);}
.chip.low  {color:#9FB4C6;}
.chip.ok   {color:#8FCF9C; border-color:rgba(123,216,143,.22);}
.chip.muted{color:#8B98A5;}
.cg-id {font: 600 1.45rem JetBrainsMono, monospace; letter-spacing:-.01em; margin:0;}
.cg-chips {display:flex; gap:6px; flex-wrap:wrap; margin: 6px 0 4px 0;}
.cg-banner {display:flex; gap:9px; align-items:flex-start; padding:7px 12px; border-radius:8px;
  font-size:.84rem; line-height:1.4; margin: 4px 0;}
.cg-banner .ic {width:13px; height:13px; margin-top:3px;}
.cg-note {display:flex; gap:6px; align-items:center; font-size:.78rem; color:#8B98A5; margin:2px 0 8px 0;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.cg-note .ic {width:12px; height:12px; flex:none;}
.cg-note[title] {cursor:help;}
.cg-notes {display:flex; flex-wrap:wrap; gap:4px 20px; margin:0 0 6px 0;}
.cg-notes .cg-note {margin:0;}
.cg-text {font-size:.92rem; line-height:1.5; color:#E6EDF3; margin:0 0 10px 0;}
.cg-banner.crit {background:rgba(242,95,92,.08); border:1px solid rgba(242,95,92,.35); color:#FFB3B1;}
.cg-banner.warn {background:rgba(255,178,36,.07); border:1px solid rgba(255,178,36,.30); color:#FFD48A;}
.tl {position:relative; margin: 4px 0 0 6px; padding-left: 22px; border-left: 1px dashed #2B3B4C;}
.tl-item {position:relative; padding: 2px 0 14px 0;}
.tl-node {position:absolute; left:-31px; top:2px; width:18px; height:18px; border-radius:50%;
  display:flex; align-items:center; justify-content:center; background:#0A0F14; border:1px solid #2B3B4C;}
.tl-node .ic {width:10px; height:10px;}
.tl-item.cause .tl-node {border-color:#FFB224; color:#FFB224; box-shadow:0 0 12px rgba(255,178,36,.35);}
.tl-item.symptom .tl-node {color:#8B98A5;}
.tl-item.gap .tl-node {color:#5AA9E6; border-style:dashed;}
.tl-head {display:flex; flex-wrap:nowrap; gap:6px; align-items:center; font-size:.72rem; color:#8B98A5;
  font-family: JetBrainsMono, monospace; white-space:nowrap; margin-top:2px; min-width:0; overflow:hidden;}
.tl-head .ent {overflow:hidden; text-overflow:ellipsis; min-width:0;}
.tl-label {font: 600 .68rem JetBrainsMono, monospace; letter-spacing:.1em; text-transform:uppercase;
  color:#8B98A5;}
.cg-guard {list-style:none; padding:0; margin:0; display:flex; flex-direction:column; gap:8px;}
.cg-guard li {display:flex; gap:8px; align-items:flex-start; font-size:.84rem; color:#AAB6C2; line-height:1.35;}
.cg-guard .ic {color:#7BD88F; width:13px; height:13px; margin-top:3px;}
.tl-item.cause .tl-label {color:#FFB224;} .tl-item.gap .tl-label {color:#5AA9E6;}
.tl-text {margin-top:3px; font-size:.9rem; line-height:1.4; color:#E6EDF3;}
.tl-item.symptom .tl-text, .tl-item.gap .tl-text {color:#AAB6C2;}
.rb {display:flex; flex-direction:column; gap:8px;}
.rb-step {display:grid; grid-template-columns: 26px 1fr auto; gap:10px; align-items:start;
  padding:8px 10px; border:1px solid #1E2A36; border-radius:10px; background:rgba(255,255,255,.015);}
.rb-step.locked {border-color: rgba(255,178,36,.28);}
.rb-num {font: 600 12px JetBrainsMono, monospace; color:#8B98A5; width:24px; height:24px; border-radius:7px;
  display:flex; align-items:center; justify-content:center; border:1px solid #2B3B4C;}
.rb-text {font-size:.9rem; line-height:1.4;}
.rb-tags {display:flex; gap:4px; align-items:center; padding-top:1px;}
.pc {display:grid; grid-template-columns: minmax(0, 1fr); gap:5px; padding:9px 0; border-bottom:1px solid #1A2430;}
.pc:last-child {border-bottom:none;}
.pc-id {font: 600 11.5px JetBrainsMono, monospace; color:#C9D4DE; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis;}
.pc-cause {grid-column: 1 / -1; font-size:.88rem; line-height:1.4; display:-webkit-box; -webkit-line-clamp:3;
  -webkit-box-orient:vertical; overflow:hidden;}
.pc-fix {grid-column: 1 / -1; font-size:.8rem; line-height:1.4; color:#8B98A5; display:-webkit-box;
  -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;}
.pc details summary {font-size:.78rem; color:#8B98A5; cursor:pointer; list-style:none;}
.pc details summary::before {content:"▸ "; font-size:.7rem;}
.pc details[open] summary::before {content:"▾ ";}
.pc details ul {margin:4px 0 0 0; padding-left:16px; font-size:.8rem; color:#8B98A5;}
.meter {height:6px; border-radius:99px; background:#1A2430; overflow:hidden; margin-top:6px;}
.meter > span {display:block; height:100%; border-radius:99px;
  background: linear-gradient(90deg, #FFB224, #7BD88F);}
.kv {display:grid; grid-template-columns: auto 1fr; gap:4px 14px; font-size:.8rem;}
.kv dt {color:#8B98A5;} .kv dd {margin:0; font-family: JetBrainsMono, monospace; text-align:right;}
div[data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}
.stTabs [data-baseweb="tab-list"] {gap: 4px;}
footer {visibility: hidden;}
/* Narrow widths: let column rows wrap instead of squeezing cards to one word per line. */
@media (max-width: 1180px) {
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] {flex-wrap: wrap;}
  [data-testid="stMain"] [data-testid="stColumn"] {min-width: 220px;}
  .st-key-cg-actions [data-testid="stColumn"] {min-width: 0;}
  [data-testid="stMain"] [data-testid="stColumn"]:has([data-testid="stPlotlyChart"]) {min-width: 100%;}
}
[data-testid="stMetric"], [data-testid="stCaptionContainer"] {white-space: normal;}
.stExpander summary p {white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
</style>
"""


@lru_cache(maxsize=None)
def icon(name: str) -> str:
    """Font Awesome icon as a CSS mask, tinted with the surrounding text color.

    st.html sanitizes inline <svg>, so the SVG is served from static/ (Streamlit static
    serving) and used as a mask over a currentColor background.
    """
    if not (ICONS / f"{name}.svg").exists():
        raise FileNotFoundError(f"missing icon {name}")
    url = f"app/static/icons/{name}.svg"
    return (f'<i class="ic" style="-webkit-mask:url({url}) center/contain no-repeat;'
            f'mask:url({url}) center/contain no-repeat;background-color:currentColor"></i>')


def chip(text: str, kind: str = "", icon_name: str | None = None, mono: bool = False,
         title: str | None = None) -> str:
    ico = icon(icon_name) if icon_name else ""
    tip = f' title="{escape(title)}"' if title else ""
    return f'<span class="chip {kind}{" mono" if mono else ""}"{tip}>{ico}{escape(str(text))}</span>'


def eyebrow(text: str, icon_name: str | None = None) -> str:
    return f'<div class="cg-eyebrow">{icon(icon_name) if icon_name else ""}{escape(text)}</div>'


def hero(engine: str, chain_ok: bool, alerts: int, pages: int) -> str:
    chain = chip("audit chain intact", "ok", "link") if chain_ok else chip("audit chain BROKEN", "crit", "link")
    return f"""
<div class="cg-hero">
  <div>
    <div class="cg-eyebrow">{icon("shield-halved")} Mastercard BizOps · UGA FinTech Capstone</div>
    <h1 class="cg-title">Change Guardian</h1>
    <p class="cg-sub">AI incident triage with a human in the loop · {alerts} alerts in, {pages} pages out</p>
  </div>
  <div class="cg-status">
    <span class="chip ok"><span class="cg-dot"></span>live</span>
    {chip(engine, "", "microchip")}
    {chain}
  </div>
</div>"""


def trust_badge(trust: float, n: int) -> str:
    """Short on the card; the full label and sample size live in the tooltip."""
    label, kind = badge(trust)
    short = label.split(" ")[0].lower()
    return chip(f"{short} {trust:.0%}", kind, "gauge-high", title=f"{label} · Trust {trust:.0%} (n={n})")


def _tl_item(kind: str, label: str, icon_name: str, alert: dict, trust: dict | None = None) -> str:
    t = (trust or {}).get(alert["root_cause"])
    badge_html = f'<div style="margin-top:4px">{trust_badge(*t)}</div>' if t else ""
    return f"""
<div class="tl-item {kind}">
  <div class="tl-node">{icon(icon_name)}</div>
  <div class="tl-label">{label}</div>
  <div class="tl-head"><span>{escape(str(alert["timestamp"]))[11:16]}</span><span>·</span>
    <span class="ent" title="{escape(alert["entity_name"])}">{escape(alert["entity_name"])}</span><span>·</span>
    <span style="color:{SEV_COLOR.get(alert["severity"], MUTED)}">{escape(alert["severity"])}</span></div>
  <div class="tl-text">{escape(alert["root_cause"])}</div>{badge_html}
</div>"""


def guardrail_list(items: list[str]) -> str:
    rows = "".join(f'<li>{icon("circle-check")}<span>{escape(t)}</span></li>' for t in items)
    return f'<ul class="cg-guard">{rows}</ul>'


def timeline(card: dict, trust: dict | None = None) -> str:
    """trust: alert name -> (score, n); shown as a badge, never used to hide an alert."""
    items = [_tl_item("cause", "Probable cause", "crosshairs", card["probable_cause"], trust)]
    items += [_tl_item("symptom", "Suppressed symptom", "bell", a, trust) for a in card.get("symptoms", [])]
    items += [_tl_item("gap", "Telemetry gap", "eye-slash", a, trust) for a in card.get("telemetry_gaps", [])]
    return f'<div class="tl">{"".join(items)}</div>'


def runbook(step_plan: list[dict]) -> str:
    rows = []
    for i, s in enumerate(step_plan, 1):
        ico, label, kind = RISK[s["risk"]]
        # One badge per step: the gate, colored by risk; the risk type is in the tooltip.
        if s["needs_approval"]:
            gate = chip("needs approval", "crit" if s["risk"] == "security" else "high", "lock",
                        title=f"{label} step · a human must approve it")
        else:
            gate = chip("read-only" if s["risk"] == "diagnostic" else label, "muted", ico, title=f"{label} step")
        rows.append(f"""
<div class="rb-step{' locked' if s['needs_approval'] else ''}">
  <div class="rb-num">{i:02d}</div>
  <div class="rb-text">{escape(s["step"])}</div>
  <div class="rb-tags">{gate}</div>
</div>""")
    return f'<div class="rb">{"".join(rows)}</div>'


def precedents(items: list[dict]) -> str:
    if not items:
        return '<div class="cg-sub">No precedent tickets for this event type.</div>'
    rows = []
    for p in items:
        tag = (chip("matches this alert", "ok", "circle-check") if p.get("matches_alert")
               else chip("same event type · different cause", "muted"))
        rows.append(f"""
<div class="pc">
  <div class="pc-id">{escape(p["incident_id"])} <span style="color:#8B98A5">· seen {p["precedent_count"]}x ·
    {escape(p.get("status", ""))}</span></div>
  <div>{tag}</div>
  <div class="pc-cause" title="{escape(p["root_cause"])}">{escape(p["root_cause"])}</div>
  <div class="pc-fix" title="{escape(p["remediation"])}">Remediation: {escape(p["remediation"])}</div>
</div>""")
    return "".join(rows)


LEVEL_KIND = {"high": "crit", "medium": "med", "low": "muted"}


def change_risk(changes: list[dict], has_log: bool = True) -> str:
    """Recent deployments on the problem's entities, highest risk first, with the reasons."""
    if not changes:
        msg = ("No deployments on these entities in the 24 h before the problem." if has_log else
               "No deploy log loaded. Add data/changes/deployments.csv "
               "(change_id, timestamp, entity, description) to score recent changes.")
        return f'<div class="cg-sub">{escape(msg)}</div>'
    rows = []
    for c in changes:
        lvl = c.get("risk_level", "low")
        tag = chip(f"{lvl} risk · {c.get('risk_score', 0):.2f}", LEVEL_KIND.get(lvl, "muted"), "bolt",
                   title="recency × (0.25 + 0.75 × fit) · high ≥ 0.55 · medium ≥ 0.30")
        mins = c.get("minutes_before")
        when = f"{mins:.0f} min before" if mins is not None and mins >= 0 else "after the problem started"
        reasons = "".join(f"<li>{escape(r)}</li>" for r in c.get("reasons", []))
        rows.append(f"""
<div class="pc">
  <div class="pc-id">{tag} {escape(str(c.get("change_id", "")))} <span style="color:#8B98A5">·
    {escape(str(c.get("entity", "")))} · {escape(when)} · {escape(c.get("kind", ""))}</span></div>
  <div class="pc-cause">{escape(str(c.get("description", "")))}</div>
  <details><summary>Why this score</summary><ul>{reasons}</ul></details>
</div>""")
    return "".join(rows)


GATE_KIND = {"LOW": ("muted", "circle-check"), "MEDIUM": ("med", "triangle-exclamation"), "HIGH": ("crit", "lock")}


def gate_badge(level: str, score: float) -> str:
    kind, ico = GATE_KIND[level]
    return chip(f"{level} · {score:.2f}", kind, ico)


def note(text: str, tooltip: str | None = None, icon_name: str = "circle-check") -> str:
    """One quiet line; the longer explanation goes in the hover tooltip."""
    tip = f' title="{escape(tooltip)}"' if tooltip else ""
    return f'<div class="cg-note"{tip}>{icon(icon_name)}<span>{escape(text)}</span></div>'


def notes(*items: str) -> str:
    """Several note() lines on one row."""
    return f'<div class="cg-notes">{"".join(items)}</div>'


def para(body: str) -> str:
    return f'<p class="cg-text">{escape(body)}</p>'


def banner(text: str, kind: str = "warn", icon_name: str = "triangle-exclamation") -> str:
    return f'<div class="cg-banner {kind}">{icon(icon_name)}<div>{escape(text)}</div></div>'


def meter(value: float) -> str:
    pct = max(0, min(100, round(100 * value)))
    return f'<div class="meter"><span style="width:{pct}%"></span></div>'


def kv(pairs: list[tuple[str, str]]) -> str:
    body = "".join(f"<dt>{escape(k)}</dt><dd>{escape(str(v))}</dd>" for k, v in pairs)
    return f'<dl class="kv">{body}</dl>'
