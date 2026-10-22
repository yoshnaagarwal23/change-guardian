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
.block-container {padding-top: 1.4rem; max-width: 1480px;}
.cg-eyebrow {font: 600 11px JetBrainsMono, monospace; letter-spacing: .14em; text-transform: uppercase;
  color: #8B98A5; margin: 0 0 8px 0; display:flex; align-items:center; gap:8px;}
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
.chip {display:inline-flex; align-items:center; gap:6px; padding:3px 10px; border-radius:999px;
  font: 500 12px Inter, sans-serif; border:1px solid #2B3B4C; color:#C9D4DE; background:rgba(255,255,255,.03);
  white-space:nowrap;}
.chip svg {width:12px; height:12px;}
.chip.mono {font-family: JetBrainsMono, monospace; font-size: 11.5px;}
.chip.crit {color:#FF8A87; border-color:rgba(242,95,92,.45); background:rgba(242,95,92,.10);}
.chip.high {color:#FFC65C; border-color:rgba(255,178,36,.45); background:rgba(255,178,36,.10);}
.chip.med  {color:#FFDD8A; border-color:rgba(255,209,102,.35); background:rgba(255,209,102,.08);}
.chip.low  {color:#8CC4F0; border-color:rgba(90,169,230,.40); background:rgba(90,169,230,.08);}
.chip.ok   {color:#9BE4AA; border-color:rgba(123,216,143,.40); background:rgba(123,216,143,.08);}
.chip.muted{color:#8B98A5;}
.cg-id {font: 600 1.45rem JetBrainsMono, monospace; letter-spacing:-.01em; margin:0;}
.cg-chips {display:flex; gap:6px; flex-wrap:wrap; margin: 8px 0 4px 0;}
.cg-banner {display:flex; gap:10px; align-items:flex-start; padding:10px 14px; border-radius:10px;
  font-size:.9rem; margin: 6px 0;}
.cg-banner svg {width:15px; height:15px; margin-top:2px; flex:none;}
.cg-banner.crit {background:rgba(242,95,92,.08); border:1px solid rgba(242,95,92,.35); color:#FFB3B1;}
.cg-banner.warn {background:rgba(255,178,36,.07); border:1px solid rgba(255,178,36,.30); color:#FFD48A;}
.tl {position:relative; margin: 4px 0 0 6px; padding-left: 22px; border-left: 1px dashed #2B3B4C;}
.tl-item {position:relative; padding: 2px 0 14px 0;}
.tl-node {position:absolute; left:-31px; top:2px; width:18px; height:18px; border-radius:50%;
  display:flex; align-items:center; justify-content:center; background:#0A0F14; border:1px solid #2B3B4C;}
.tl-node svg {width:10px; height:10px;}
.tl-item.cause .tl-node {border-color:#FFB224; color:#FFB224; box-shadow:0 0 12px rgba(255,178,36,.35);}
.tl-item.symptom .tl-node {color:#8B98A5;}
.tl-item.gap .tl-node {color:#5AA9E6; border-style:dashed;}
.tl-head {display:flex; gap:8px; align-items:center; font-size:.78rem; color:#8B98A5;
  font-family: JetBrainsMono, monospace;}
.tl-label {font-weight:600; letter-spacing:.06em; text-transform:uppercase; font-size:.7rem;}
.tl-item.cause .tl-label {color:#FFB224;} .tl-item.gap .tl-label {color:#5AA9E6;}
.tl-text {margin-top:3px; font-size:.95rem; color:#E6EDF3;}
.tl-item.symptom .tl-text, .tl-item.gap .tl-text {color:#AAB6C2;}
.rb {display:flex; flex-direction:column; gap:8px;}
.rb-step {display:grid; grid-template-columns: 28px 1fr auto; gap:12px; align-items:start;
  padding:10px 12px; border:1px solid #1E2A36; border-radius:10px; background:rgba(255,255,255,.015);}
.rb-step.locked {border-color: rgba(255,178,36,.28);}
.rb-num {font: 600 12px JetBrainsMono, monospace; color:#8B98A5; width:24px; height:24px; border-radius:7px;
  display:flex; align-items:center; justify-content:center; border:1px solid #2B3B4C;}
.rb-text {font-size:.92rem; line-height:1.4;}
.rb-tags {display:flex; flex-direction:column; gap:4px; align-items:flex-end;}
.pc {display:grid; grid-template-columns: 1fr auto; gap:6px 14px; padding:10px 0; border-bottom:1px solid #1A2430;}
.pc:last-child {border-bottom:none;}
.pc-id {font: 600 12.5px JetBrainsMono, monospace; color:#C9D4DE;}
.pc-cause {grid-column: 1 / -1; font-size:.9rem;}
.pc-fix {grid-column: 1 / -1; font-size:.85rem; color:#8B98A5;}
.meter {height:6px; border-radius:99px; background:#1A2430; overflow:hidden; margin-top:6px;}
.meter > span {display:block; height:100%; border-radius:99px;
  background: linear-gradient(90deg, #FFB224, #7BD88F);}
.kv {display:grid; grid-template-columns: auto 1fr; gap:6px 14px; font-size:.86rem;}
.kv dt {color:#8B98A5;} .kv dd {margin:0; font-family: JetBrainsMono, monospace; text-align:right;}
div[data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}
.stTabs [data-baseweb="tab-list"] {gap: 4px;}
footer {visibility: hidden;}
</style>
"""


@lru_cache(maxsize=None)
def icon(name: str) -> str:
    """Inline Font Awesome SVG that inherits the surrounding text color."""
    svg = (ICONS / f"{name}.svg").read_text()
    svg = svg.replace(svg[svg.index("<!--"):svg.index("-->") + 3], "") if "<!--" in svg else svg
    return svg.replace("<svg ", '<svg fill="currentColor" style="width:1em;height:1em" ', 1)


def chip(text: str, kind: str = "", icon_name: str | None = None, mono: bool = False) -> str:
    ico = icon(icon_name) if icon_name else ""
    return f'<span class="chip {kind}{" mono" if mono else ""}">{ico}{escape(str(text))}</span>'


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


def _tl_item(kind: str, label: str, icon_name: str, alert: dict) -> str:
    return f"""
<div class="tl-item {kind}">
  <div class="tl-node">{icon(icon_name)}</div>
  <div class="tl-head"><span class="tl-label">{label}</span>
    <span>{escape(str(alert["timestamp"]))[11:16]}</span><span>·</span>
    <span>{escape(alert["entity_name"])}</span>
    <span style="color:{SEV_COLOR.get(alert["severity"], MUTED)}">{escape(alert["severity"])}</span></div>
  <div class="tl-text">{escape(alert["root_cause"])}</div>
</div>"""


def timeline(card: dict) -> str:
    items = [_tl_item("cause", "Probable cause", "crosshairs", card["probable_cause"])]
    items += [_tl_item("symptom", "Suppressed symptom", "bell", a) for a in card.get("symptoms", [])]
    items += [_tl_item("gap", "Telemetry gap", "eye-slash", a) for a in card.get("telemetry_gaps", [])]
    return f'<div class="tl">{"".join(items)}</div>'


def runbook(step_plan: list[dict]) -> str:
    rows = []
    for i, s in enumerate(step_plan, 1):
        ico, label, kind = RISK[s["risk"]]
        gate = chip("needs approval", "high", "lock") if s["needs_approval"] else chip("read-only", "muted")
        rows.append(f"""
<div class="rb-step{' locked' if s['needs_approval'] else ''}">
  <div class="rb-num">{i:02d}</div>
  <div class="rb-text">{escape(s["step"])}</div>
  <div class="rb-tags">{chip(label, kind, ico)}{gate}</div>
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
  <div class="pc-cause">{escape(p["root_cause"])}</div>
  <div class="pc-fix">Remediation: {escape(p["remediation"])}</div>
</div>""")
    return "".join(rows)


def banner(text: str, kind: str = "warn", icon_name: str = "triangle-exclamation") -> str:
    return f'<div class="cg-banner {kind}">{icon(icon_name)}<div>{escape(text)}</div></div>'


def meter(value: float) -> str:
    pct = max(0, min(100, round(100 * value)))
    return f'<div class="meter"><span style="width:{pct}%"></span></div>'


def kv(pairs: list[tuple[str, str]]) -> str:
    body = "".join(f"<dt>{escape(k)}</dt><dd>{escape(str(v))}</dd>" for k, v in pairs)
    return f'<dl class="kv">{body}</dl>'
