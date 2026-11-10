"""Change Guardian - live demo console.

  uv run streamlit run app.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit_shadcn_ui as sui

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).parent / ".env")  # ANTHROPIC_API_KEY etc.; real env vars still win

import ui_components as ui  # noqa: E402
from agent import TriageAgent  # noqa: E402
from alert_trust import ACTION, NO_ACTION, badge, disposition, load_history, trust_lookup, trust_table  # noqa: E402
from guardrails import AuditLog  # noqa: E402
from kpis import compute_kpis  # noqa: E402
from llm import Generation, cost_label, get_backend, token_cost_usd, usd  # noqa: E402
from notify import send_risk_alert  # noqa: E402
from predeploy_risk import label_culprits, llm_summary, load_inputs, score_all  # noqa: E402
from tools import CHANGES, TriageTools  # noqa: E402

AUDIT = Path("outputs/audit_log.jsonl")
REPLAY = Path("outputs/demo_cache.jsonl")
ADAPTER = Path("models/lora-mac")
COMPARISON = Path("reports/model_comparison.json")
# Real change records win; the invented demo log is only a fallback (scripts/make_demo_changes.py).
DEMO_CHANGES = Path("data/changes/deployments_demo.csv")
DEMO_MODE = not CHANGES.exists() and DEMO_CHANGES.exists()

st.set_page_config(page_title="Change Guardian", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")
st.html(ui.CSS)


# ----------------------------------------------------------------------------- data + engines
@st.cache_resource
def load_tools() -> TriageTools:
    return TriageTools.from_disk(changes_csv=DEMO_CHANGES if DEMO_MODE else CHANGES)


@st.cache_resource(show_spinner="Loading local model…")
def load_backend(mode: str):
    if mode == "Local model + LoRA":
        return get_backend("hf", adapter_path=str(ADAPTER))
    if mode == "Local model (base)":
        return get_backend("hf")
    if mode == "Claude API":
        return get_backend("anthropic")
    return None


class MeteredBackend:
    """Wraps an engine so every model call (triage or summary) adds to the session total in the sidebar."""

    def __init__(self, inner, on_call):
        self.inner, self.name, self.on_call = inner, inner.name, on_call

    def generate(self, messages: list[dict]) -> Generation:
        gen = self.inner.generate(messages)
        self.on_call(gen)
        return gen


# Engines that need optional packages. Missing ones are listed as "not configured" instead of offered.
ENGINE_DEPS = {"Local model + LoRA": ["torch", "transformers", "peft"],
               "Local model (base)": ["torch", "transformers"],
               "Claude API": ["anthropic"]}
ENGINE_FIX = {"Local model + LoRA": "uv sync --extra ml", "Local model (base)": "uv sync --extra ml",
              "Claude API": "uv sync"}


def missing_deps(mode: str) -> list[str]:
    return [m for m in ENGINE_DEPS.get(mode, []) if importlib.util.find_spec(m) is None]


@st.cache_data
def load_replay() -> dict:
    if not REPLAY.exists():
        return {}
    return {c["problem_id"]: c for c in map(json.loads, REPLAY.read_text().splitlines())}


@st.cache_data
def alert_history() -> pd.DataFrame:
    return load_history()


def current_trust() -> pd.DataFrame:
    return trust_table(alert_history(), st.session_state.get("dispositions", []))


@st.cache_data
def batch_cards() -> list[dict]:
    rules = TriageAgent(load_tools())
    return [rules.triage(p) for p in load_tools().problems["problem_id"]]


@st.cache_data(show_spinner="Scoring pending changes…")
def pending_scores() -> list[dict]:
    pending, tiers, freezes = load_inputs()
    if pending.empty:
        return []
    return score_all(pending, tiers, freezes, label_culprits(load_tools()))


tools = load_tools()
candidates = ["Rules only", "Local model (base)"]
if (ADAPTER / "adapter_config.json").exists():
    candidates.insert(1, "Local model + LoRA")
if os.getenv("ANTHROPIC_API_KEY"):
    candidates.append("Claude API")
if REPLAY.exists():
    candidates.append("Replay (offline)")
unavailable = {m: missing_deps(m) for m in candidates if missing_deps(m)}
modes = [m for m in candidates if m not in unavailable]

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.html(ui.eyebrow("Engine", "microchip"))
    mode = st.selectbox("Engine", modes, index=1 if "Local model + LoRA" in modes else 0,
                        label_visibility="collapsed")
    if unavailable:
        st.html(ui.note("Not configured: " + ", ".join(unavailable),
                        " · ".join(f"{m}: missing {', '.join(d)} (run `{ENGINE_FIX[m]}`)"
                                   for m, d in unavailable.items()), "triangle-exclamation"))

    # Load the engine here so a failure shows as a small warning, never a crash; fall back to rules.
    failed = st.session_state.setdefault("engine_failed", {})
    backend = None
    if mode not in {"Rules only", "Replay (offline)"}:
        if mode not in failed:
            try:
                backend = load_backend(mode)
            except Exception as exc:  # noqa: BLE001 - any load failure falls back to rules
                failed[mode] = f"{type(exc).__name__}: {exc}"
        if mode in failed:
            st.html(ui.banner(f"{mode} failed to load, using Rules only.", "warn"))
            st.html(ui.note("Why it failed", failed[mode], "magnifying-glass"))
            mode = "Rules only"

    usage = st.session_state.setdefault("llm_usage", {"calls": 0, "tokens_in": 0, "tokens_out": 0, "usd": 0.0})
    usage_slot = st.empty()

    def show_usage() -> None:
        n = usage["calls"]
        usage_slot.html(ui.note(f"Session: {n} call{'s' if n != 1 else ''} · {usd(usage['usd'])}",
                                f"{usage['tokens_in']:,} tokens in · {usage['tokens_out']:,} tokens out · "
                                "model calls this browser session (triage + summaries), at list prices",
                                "gauge-high"))

    def record_call(gen: Generation) -> None:
        usage["calls"] += 1
        usage["tokens_in"] += gen.tokens_in
        usage["tokens_out"] += gen.tokens_out
        usage["usd"] += token_cost_usd(backend.name, gen.tokens_in, gen.tokens_out)
        show_usage()

    show_usage()
    if backend is not None:
        backend = MeteredBackend(backend, record_call)
    st.caption({"Rules only": "Deterministic: correlation, routing, SOP branch, precedents. $0, ~10 ms.",
                "Local model + LoRA": "Qwen2.5-0.5B + our LoRA adapter, on this laptop. $0, ~1 s.",
                "Local model (base)": "Untuned Qwen2.5-0.5B, for comparison.",
                "Claude API": "Hosted comparison backend.",
                "Replay (offline)": "Pre-computed cards: stage-safe fallback."}.get(mode, ""))
    st.html(ui.eyebrow("Reviewer", "user-shield"))
    reviewer = st.text_input("Reviewer", "on-call SRE", label_visibility="collapsed")
    st.divider()
    st.html(ui.eyebrow("Guardrails", "lock"))
    st.html(ui.guardrail_list(["Production changes need approval", "Steps must trace to an SOP or ticket",
                               "Summary must name the alert that fired", "Every decision is hash-chained"]))

agent = TriageAgent(tools, backend)


def triage(pid: str) -> dict:
    cache = st.session_state.setdefault("cards", {})
    key = (mode, pid)
    if key not in cache:
        replayed = load_replay().get(pid) if mode == "Replay (offline)" else None
        cache[key] = replayed or (TriageAgent(tools).triage(pid) if mode == "Replay (offline)"
                                  else agent.triage(pid))
    return cache[key]


# ----------------------------------------------------------------------------- hero + KPI cards
log = AuditLog(AUDIT)
n_alerts, n_pages = len(tools.alerts), tools.problems["problem_id"].nunique()
st.html(ui.hero(mode, log.verify(), n_alerts, n_pages))

k1, k2, k3, k4 = st.columns(4)
with k1:
    sui.metric_card("Alerts received", n_alerts, description="Dynatrace · Oct 12–15", key="m1")
with k2:
    sui.metric_card("Pages after correlation", n_pages,
                    description=f"-{100 * (1 - n_pages / n_alerts):.0f}% noise", key="m2")
with k3:
    covered = sum(tools.find_sop(e) is not None for e in tools.problems["event_type"].unique())
    sui.metric_card("Event types with an SOP", f"{covered} / 6", description="gap: Monitoring",
                    key="m3")
with k4:
    sui.metric_card("Decisions logged", len(log.entries()),
                    description="chain verified" if log.verify() else "chain broken", key="m4")

tab = sui.tabs(["Triage", "Insights", "Alert Trust", "Pending Changes", "Audit trail"], value="Triage", key="tabs")

# ----------------------------------------------------------------------------- triage
if tab == "Triage":
    left, right = st.columns([1.2, 1.8], gap="large")

    with left:
        st.html(ui.eyebrow("Problem feed", "bell"))
        sev = st.pills("Severity", sorted(tools.problems["severity"].unique()), selection_mode="multi",
                       default=["SEV-1", "SEV-2"], label_visibility="collapsed")
        evt = st.pills("Event", sorted(tools.problems["event_type"].unique()), selection_mode="multi",
                       format_func=lambda e: {"Monitoring Unavailable": "Monitoring", "Custom Alert": "Custom"}.get(e, e),
                       label_visibility="collapsed")
        feed = tools.problems
        if sev:
            feed = feed[feed["severity"].isin(sev)]
        if evt:
            feed = feed[feed["event_type"].isin(evt)]
        feed = feed.sort_values(["severity", "first_seen"]).reset_index(drop=True)
        view = pd.DataFrame({
            "Sev": feed["severity"],
            "Problem": feed["problem_id"],
            "Event": feed["event_type"],
            "Entity": feed["entities"].str.split("; ").str[0],
        })
        styled = view.style.map(lambda s: f"color:{ui.SEV_COLOR.get(s, ui.MUTED)};font-weight:600", subset=["Sev"])
        picked = st.dataframe(styled, hide_index=True, height=620, on_select="rerun",
                              selection_mode="single-row", key="feed", width="stretch",
                              column_config={"Sev": st.column_config.TextColumn(width=52),
                                             "Problem": st.column_config.TextColumn(width=78),
                                             "Event": st.column_config.TextColumn(width=92)})
        st.caption(f"{len(feed)} of {len(tools.problems)} problems · click a row")

    if feed.empty:
        right.info("No problems match the filters.")
        st.stop()
    rows = picked.selection.rows
    pid = feed.iloc[rows[0] if rows else 0]["problem_id"]

    with right:
        with st.spinner("Triaging…"):
            card = triage(pid)

        st.html(f'<div class="cg-eyebrow">{ui.icon("crosshairs")}Incident</div>'
                f'<p class="cg-id">{pid} <span style="color:#8B98A5;font-weight:400">· {card["event_type"]}</span></p>')
        chips = [ui.chip(f"Priority {card['priority']}", ui.PRIORITY_KIND.get(card["priority"], ""), "bolt"),
                 ui.chip(card["assigned_group"], "", "list-check", mono=True),
                 ui.chip(card["sop_id"] or "no SOP", "ok" if card["sop_id"] else "crit", "book", mono=True),
                 ui.chip(f"{card['backend']} · {card['latency_ms']:.0f} ms", "muted", "microchip", mono=True)]
        if card["notify_secops"]:
            chips.append(ui.chip("notify SecOps", "crit", "user-shield"))
        st.html(f'<div class="cg-chips">{"".join(chips)}</div>')

        if card["escalate"]:
            st.html(ui.banner("Escalated: no SOP covers this event type. Steps come from past tickets only "
                              "(documentation gap).", "crit"))
        for w in card["warnings"]:
            if "No SOP" not in w and "Security signal" not in w:
                st.html(ui.banner(w, "warn"))

        c1, c2 = st.columns([1.35, 1], gap="medium")
        with c1, st.container(border=True):
            n = 1 + len(card["symptoms"]) + len(card["telemetry_gaps"])
            st.html(ui.eyebrow(f"Correlation · {n} alert{'s' if n > 1 else ''} → 1 page", "layer-group"))
            st.html(ui.timeline(card, trust_lookup(current_trust())))
            if card["related_problems"]:
                st.html(ui.note(f"{len(card['related_problems'])} related problem(s) on this entity",
                                "Same entity within 60 min: " + ", ".join(card["related_problems"]), "link"))
        with c2, st.container(border=True):
            st.html(ui.eyebrow("Root-cause summary", "magnifying-glass"))
            st.html(ui.para(card["root_cause_summary"]))
            st.html(ui.eyebrow("Confidence", "gauge-high") + ui.meter(card["confidence"]))
            st.html(ui.kv([("confidence", f"{card['confidence']:.2f}"),
                           ("tokens in/out", f"{card['tokens_in']}/{card['tokens_out']}"),
                           ("est. cost", cost_label(card["backend"], card["tokens_in"], card["tokens_out"]))]))

        with st.container(border=True):
            locked = sum(s["needs_approval"] for s in card["step_plan"])
            st.html(ui.eyebrow(f"Runbook · {len(card['step_plan'])} steps · {locked} need approval", "list-check"))
            st.html(ui.runbook(card["step_plan"]))

        with st.container(border=True):
            changes = card.get("recent_changes") or []
            level = (card.get("change_risk") or {}).get("level", "none")
            st.html(ui.eyebrow(f"Change risk · {len(changes)} deploy{'s' if len(changes) != 1 else ''} · {level}",
                               "bolt"))
            if DEMO_MODE:
                st.html(ui.note("demo deploy log · invented", "Invented DEMO-CHG deploys from "
                                "scripts/make_demo_changes.py. Real records in data/changes/deployments.csv "
                                "take precedence.", "triangle-exclamation"))
            st.html(ui.change_risk(changes, has_log=tools.changes is not None))

        p1, p2 = st.columns([1.35, 1], gap="medium")
        with p1, st.container(border=True):
            precs = card.get("precedents") or tools.find_similar_tickets(
                f"{card['event_type']} {card['probable_cause']['root_cause']}", card["event_type"])
            st.html(ui.eyebrow("Precedent tickets", "clock-rotate-left"))
            st.html(ui.precedents(precs))
        with p2, st.container(border=True):
            st.html(ui.eyebrow("Draft Remedy ticket", "circle-nodes"))
            edit = st.toggle("Edit before approving", key=f"edit-{pid}")
            summary, steps = card["root_cause_summary"], card["next_steps"]
            if edit:
                summary = st.text_area("Summary", summary, height=100, key=f"sum-{pid}")
                steps = [s for s in st.text_area("Steps (one per line)", "\n".join(steps), height=140,
                                                 key=f"steps-{pid}").splitlines() if s.strip()]
            final = {**card, "root_cause_summary": summary, "next_steps": steps}
            if card["sop_id"]:
                with st.expander(f"{card['sop_id']} full text"):
                    st.markdown(Path(f"data/sops/{card['sop_id']}.md").read_text())
            with st.expander("Ticket JSON"):
                st.json(tools.draft_ticket(final))

        actions = st.container(key="cg-actions")
        b1, b2, b3, b4, _ = actions.columns([1, 1, 1, 1.5, 0.7])
        with b1:
            approve = sui.button("Approve", key=f"ap-{pid}", width="stretch")
        with b2:
            escalate = sui.button("Escalate", variant="secondary", key=f"es-{pid}", width="stretch")
        with b3:
            reject = sui.button("Reject", variant="destructive", key=f"rj-{pid}", width="stretch")
        with b4:
            no_action = sui.button("No action needed", variant="outline", key=f"na-{pid}", width="stretch")
        decision = ("edited" if edit else "approved") if approve else "escalated" if escalate else \
            "rejected" if reject else None
        alerts = [card["probable_cause"], *card["symptoms"], *card["telemetry_gaps"]]
        if decision:
            rec = AuditLog(AUDIT).append({"problem_id": pid, "decision": decision, "user": reviewer,
                                          "backend": card["backend"], "ticket": tools.draft_ticket(final)})
            if decision != "rejected":  # approve / edit / escalate: the probable cause needed action
                st.session_state.setdefault("dispositions", []).append(
                    disposition(card["probable_cause"], ACTION, reviewer))
            st.toast(f"{decision.title()} by {reviewer} · hash {rec['hash'][:10]}…", icon="✅")
        if no_action:
            st.session_state.setdefault("dispositions", []).extend(
                disposition(a, NO_ACTION, reviewer) for a in alerts)
            rec = AuditLog(AUDIT).append({"problem_id": pid, "decision": NO_ACTION, "user": reviewer,
                                          "backend": card["backend"],
                                          "alerts": [a["root_cause"] for a in alerts]})
            st.toast(f"{len(alerts)} alert(s) marked no action needed by {reviewer} · "
                     f"hash {rec['hash'][:10]}…", icon="🔕")

# ----------------------------------------------------------------------------- insights
elif tab == "Insights":
    cards = batch_cards()
    k = compute_kpis(cards, tools.alerts)
    a, b, c, d = st.columns(4)
    with a:
        sui.metric_card("SOP coverage", f"{k['sop_coverage_pct']}%", description="have a runbook", key="i1")
    with b:
        sui.metric_card("Escalated", f"{k['escalation_rate_pct']}%", description="documentation gaps", key="i2")
    with c:
        sui.metric_card("Steps gated", f"{k['steps_needing_approval_pct']}%", description="need approval",
                        key="i3")
    with d:
        sui.metric_card("SecOps flags", k["secops_notifications"], description="auth signals",
                        key="i4")

    layout = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#C9D4DE"),
                  margin=dict(l=10, r=10, t=40, b=10), height=330)
    g1, g2 = st.columns(2, gap="large")
    with g1, st.container(border=True):
        misrouted = sum(c["probable_cause"]["event_type"] != c["event_type"] for c in cards)
        fig = go.Figure(go.Funnel(
            y=["Alerts", "Pages (correlated)", "Pages with an SOP", "Routed to the probable-cause team"],
            x=[k["alerts_in"], k["pages_after_correlation"], round(len(cards) * k["sop_coverage_pct"] / 100),
               len(cards) - misrouted],
            marker=dict(color=["#6B7785", "#5E6A78", "#52606E", ui.BLUE]), textinfo="value+percent initial"))
        fig.update_layout(title="From alert noise to actionable pages", **layout)
        st.plotly_chart(fig, width="stretch")
    with g2, st.container(border=True):
        counts = pd.DataFrame(cards)["event_type"].value_counts()
        fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h",
                               marker=dict(color=[ui.RED if e == "Monitoring Unavailable" else ui.MUTED
                                                  for e in counts.index])))
        fig.update_layout(title="Problems by event type (red = no SOP)", yaxis=dict(autorange="reversed"), **layout)
        st.plotly_chart(fig, width="stretch")

    with st.container(border=True):
        st.html(ui.eyebrow("Model evaluation · held-out alert types the model never saw", "microchip"))
        if COMPARISON.exists():
            comp = pd.DataFrame(json.loads(COMPARISON.read_text())).T
            metrics = {"alert_branch_hit_rate": "Right remedy branch", "alert_consistency": "Names the alert",
                       "alert_grounding": "Grounded steps", "alert_json_valid_rate": "Valid output"}
            fig = go.Figure()
            for color, (name, row) in zip([ui.MUTED, ui.RED, ui.AMBER, ui.GREEN], comp.iterrows()):
                fig.add_bar(name=name, x=list(metrics.values()), y=[row.get(m) for m in metrics],
                            marker_color=color)
            fig.update_layout(barmode="group", yaxis=dict(range=[0, 1.05], tickformat=".0%"),
                              legend=dict(orientation="h", y=1.15), **{**layout, "height": 360})
            st.plotly_chart(fig, width="stretch")
            st.caption("v1 trained on tickets only learned to recite the precedent ticket (0% right branch on "
                       "unseen alerts). The alert-consistency guardrail now catches that failure for every model.")
        else:
            st.info("Run `uv run python src/compare_models.py models/lora-mac-v1 models/lora-mac` to populate.")

# ----------------------------------------------------------------------------- alert trust
elif tab == "Alert Trust":
    table = current_trust()
    info = ui.note("Informational only · never hides an alert",
                   "A low trust score never hides or suppresses an alert. It flags alert definitions a human "
                   "may want to tune.")
    if table.empty:
        st.html(info)
        st.info("Run `uv run python scripts/make_alert_dispositions.py` to create the demo history.")
    else:
        session_n = len(st.session_state.get("dispositions", []))
        st.html(ui.notes(info, ui.note(
            f"{int(table['firings'].sum())} dispositions · {session_n} this session · demo history",
            "Trust = share of past firings that needed action. Trusted ≥ 60%, Mixed 30–59%, Noisy < 30%. "
            "Demo history: data/alert_dispositions.csv (invented).", "gauge-high")))
        with st.container(border=True):
            colors = [ui.MUTED if t >= 0.6 else ui.AMBER if t >= 0.3 else ui.RED for t in table["trust"]]
            names = table["alert_name"].str.rstrip(".")
            fig = go.Figure(go.Bar(x=table["trust"], y=[n if len(n) <= 44 else n[:43] + "…" for n in names],
                                   orientation="h", marker=dict(color=colors),
                                   text=[f"{t:.0%}" for t in table["trust"]], textposition="outside",
                                   hovertext=[f"{n} · n={f}" for n, f in zip(names, table["firings"])],
                                   hoverinfo="text"))
            fig.update_layout(title=dict(text="Trust by alert · red = noisy", font=dict(size=13)),
                              xaxis=dict(range=[0, 1.1], tickformat=".0%"), yaxis=dict(autorange="reversed"),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(family="Inter", color="#C9D4DE"), margin=dict(l=10, r=10, t=40, b=10),
                              height=max(330, 24 * len(table)))
            st.plotly_chart(fig, width="stretch")
        view = pd.DataFrame({"Alert": table["alert_name"], "Type": table["alert_type"],
                             "Firings": table["firings"], "% no action": table["no_action_pct"].round(0),
                             "Trust": (100 * table["trust"]).round(0),
                             "Status": [badge(t)[0].split(" ")[0] for t in table["trust"]]})
        st.dataframe(view, hide_index=True, width="stretch",
                     column_config={"% no action": st.column_config.NumberColumn(format="%d%%", width=90),
                                    "Trust": st.column_config.NumberColumn(format="%d%%", width=64),
                                    "Firings": st.column_config.NumberColumn(width=64),
                                    "Status": st.column_config.TextColumn(
                                        width=80, help="Trusted ≥ 60% · Mixed 30–59% · Noisy < 30% (review threshold)")})

# ----------------------------------------------------------------------------- pending changes
elif tab == "Pending Changes":
    scored = pending_scores()
    info = ui.note("Recommendation only · humans decide",
                   "Change Guardian never deploys or blocks a change. A human can always approve with a "
                   "justification, and every override is audited.")
    if not scored:
        st.html(info)
        st.info("Run `uv run python scripts/make_pending_changes.py` to create the demo pending changes.")
        st.stop()
    row = [info, ui.note("demo pending changes · invented", "Generated by scripts/make_pending_changes.py",
                         "triangle-exclamation")]
    overrides = st.session_state.setdefault("overrides", {})

    # Webhook for HIGH changes, once per change per session; never raises.
    alerted = st.session_state.setdefault("risk_alerts", {})
    for c in scored:
        if c["level"] == "HIGH" and c["change_id"] not in alerted:
            alerted[c["change_id"]] = send_risk_alert(c["change_id"], c["entity"], c["owner"], c["risk_score"],
                                                      c["level"], c["reasons"])[1]
    if alerted:
        row.append(ui.note(f"Risk webhook · {len(alerted)} HIGH", next(iter(alerted.values())), "bell"))
    st.html(ui.notes(*row))

    SHORT_GATE = {("LOW", False): "Safe to land", ("LOW", True): "Safe to land",
                  ("MEDIUM", False): "Extra review", ("MEDIUM", True): "Blocked · freeze",
                  ("HIGH", False): "Blocked · senior", ("HIGH", True): "Blocked · senior"}

    def status(c: dict) -> str:
        if c["change_id"] in overrides:
            return f"Approved · {overrides[c['change_id']]['user']}"
        return f"Awaiting {c['gate']['approver']}" if c["gate"]["blocked"] else "—"

    view = pd.DataFrame({"Change": [c["change_id"] for c in scored], "Entity": [c["entity"] for c in scored],
                         "Planned": [c["planned_time"].strftime("%a %m-%d %H:%M") for c in scored],
                         "Score": [c["risk_score"] for c in scored], "Level": [c["level"] for c in scored],
                         "Freeze": ["❄" if c["freeze"] else "" for c in scored],
                         "Gate": [SHORT_GATE[(c["level"], bool(c["freeze"]))] for c in scored],
                         "Status": [status(c) for c in scored]})
    st.dataframe(view.style.map(lambda v: f"color:{ {'HIGH': ui.RED, 'MEDIUM': ui.AMBER, 'LOW': ui.MUTED}[v]};"
                                          "font-weight:600", subset=["Level"]),
                 hide_index=True, width="stretch",
                 column_config={"Score": st.column_config.NumberColumn(format="%.2f", width=56),
                                "Level": st.column_config.TextColumn(width=72),
                                "Freeze": st.column_config.TextColumn(width=56, help="Planned inside a code freeze"),
                                "Planned": st.column_config.TextColumn(width=110),
                                "Gate": st.column_config.TextColumn(
                                    help="LOW < 0.35 safe to land · MEDIUM needs review (blocked in a freeze) · "
                                         "HIGH ≥ 0.65 needs senior approval")})

    summaries = st.session_state.setdefault("risk_summaries", {})
    for c in scored:
        desc = c["description"] if len(c["description"]) <= 44 else c["description"][:43] + "…"
        with st.expander(f"{c['change_id']} · {c['level']} {c['risk_score']:.2f} · {c['entity']} · {desc}"):
            chips = [ui.gate_badge(c["level"], c["risk_score"]),
                     ui.chip(SHORT_GATE[(c["level"], bool(c["freeze"]))], "muted", title=c["gate"]["label"]),
                     ui.chip(c["owner"], "muted", "user-shield", mono=True, title="owner")]
            if c["freeze"]:
                chips.append(ui.chip("freeze", "muted", "lock", title=c["freeze"]))
            if c["off_hours"]:
                chips.append(ui.chip(c["off_hours"], "muted", "clock-rotate-left"))
            st.html(f'<div class="cg-chips">{"".join(chips)}</div>')

            st.html(ui.eyebrow("Why · contribution per signal", "list-check"))
            contrib = [x for x in c["contributions"] if x["contribution"]][::-1]
            fig = go.Figure(go.Bar(x=[x["contribution"] for x in contrib], y=[x["signal"] for x in contrib],
                                   orientation="h", text=[f"{x['contribution']:+.2f}" for x in contrib],
                                   textposition="outside", hovertext=[x["reason"] for x in contrib],
                                   hoverinfo="text",
                                   marker=dict(color=["#C0605D" if x["contribution"] > 0 else ui.GREEN
                                                      for x in contrib])))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(family="Inter", color="#C9D4DE"), margin=dict(l=10, r=10, t=10, b=10),
                              height=30 + 26 * len(contrib), font_size=12,
                              xaxis=dict(zeroline=True, zerolinecolor="#2B3B4C", range=[-0.25, 0.3]))
            st.plotly_chart(fig, width="stretch", key=f"why-{c['change_id']}")

            key = (c["change_id"], getattr(backend, "name", None))
            if backend is None:
                st.html(ui.note("Model summary: pick an engine in the sidebar",
                                "An optional 2-sentence summary from the reasons above. It never changes the score."))
            elif key in summaries:
                st.info(summaries[key])
            elif st.button("Summarize with model", key=f"sum-{c['change_id']}"):
                summaries[key] = llm_summary(backend, c)
                st.rerun()

            if c["change_id"] in overrides:
                o = overrides[c["change_id"]]
                st.caption(f"Approved by {o['user']} · “{o['justification']}” · hash {o['hash'][:10]}…")
            else:
                why = st.text_input("Justification", key=f"just-{c['change_id']}",
                                    placeholder="Why is it safe to land anyway?")
                if st.button("Approve with justification", key=f"ov-{c['change_id']}", disabled=not why.strip()):
                    rec = AuditLog(AUDIT).append({"change_id": c["change_id"], "decision": "override_approved",
                                                  "user": reviewer, "backend": "rules", "risk_score": c["risk_score"],
                                                  "level": c["level"], "gate": c["gate"]["label"],
                                                  "justification": why.strip()})
                    overrides[c["change_id"]] = {"user": reviewer, "justification": why.strip(), "hash": rec["hash"]}
                    st.toast(f"{c['change_id']} approved by {reviewer} · hash {rec['hash'][:10]}…", icon="✅")
                    st.rerun()

# ----------------------------------------------------------------------------- audit
else:
    entries = AuditLog(AUDIT).entries()
    ok = AuditLog(AUDIT).verify()
    if ok:
        st.html(ui.note("Hash chain verified", "No decision has been edited after the fact.", "link"))
    else:
        st.html(ui.banner("Hash chain broken: the log was edited.", "crit", "triangle-exclamation"))
    if entries:
        df = pd.DataFrame(entries)
        df["hash"] = df["hash"].str[:16] + "…"
        st.dataframe(df.reindex(columns=["ts", "problem_id", "change_id", "decision", "user", "backend", "hash"]), hide_index=True,
                     width="stretch", height=520)
    else:
        st.caption("No decisions yet. Approve, escalate, or reject a card on the Triage tab.")
