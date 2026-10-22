"""Change Guardian - live demo console.

  uv run streamlit run app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit_shadcn_ui as sui

sys.path.insert(0, str(Path(__file__).parent / "src"))

import ui_components as ui  # noqa: E402
from agent import TriageAgent  # noqa: E402
from guardrails import AuditLog  # noqa: E402
from kpis import compute_kpis  # noqa: E402
from llm import get_backend  # noqa: E402
from tools import TriageTools  # noqa: E402

AUDIT = Path("outputs/audit_log.jsonl")
REPLAY = Path("outputs/demo_cache.jsonl")
ADAPTER = Path("models/lora-mac")
COMPARISON = Path("reports/model_comparison.json")

st.set_page_config(page_title="Change Guardian", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")
st.html(ui.CSS)


# ----------------------------------------------------------------------------- data + engines
@st.cache_resource
def load_tools() -> TriageTools:
    return TriageTools.from_disk()


@st.cache_resource(show_spinner="Loading local model…")
def load_backend(mode: str):
    if mode == "Local model + LoRA":
        return get_backend("hf", adapter_path=str(ADAPTER))
    if mode == "Local model (base)":
        return get_backend("hf")
    if mode == "Claude API":
        return get_backend("anthropic")
    return None


@st.cache_data
def load_replay() -> dict:
    if not REPLAY.exists():
        return {}
    return {c["problem_id"]: c for c in map(json.loads, REPLAY.read_text().splitlines())}


@st.cache_data
def batch_cards() -> list[dict]:
    rules = TriageAgent(load_tools())
    return [rules.triage(p) for p in load_tools().problems["problem_id"]]


tools = load_tools()
modes = ["Rules only", "Local model (base)"]
if (ADAPTER / "adapter_config.json").exists():
    modes.insert(1, "Local model + LoRA")
if os.getenv("ANTHROPIC_API_KEY"):
    modes.append("Claude API")
if REPLAY.exists():
    modes.append("Replay (offline)")

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.html(ui.eyebrow("Engine", "microchip"))
    mode = st.selectbox("Engine", modes, index=1 if "Local model + LoRA" in modes else 0,
                        label_visibility="collapsed")
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

backend = None if mode in {"Rules only", "Replay (offline)"} else load_backend(mode)
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
    sui.metric_card("Alerts received", n_alerts, description="Dynatrace sample, Oct 12–15", key="m1")
with k2:
    sui.metric_card("Pages after correlation", n_pages,
                    description=f"-{100 * (1 - n_pages / n_alerts):.0f}% noise · one page per problem", key="m2")
with k3:
    covered = sum(tools.find_sop(e) is not None for e in tools.problems["event_type"].unique())
    sui.metric_card("Event types with an SOP", f"{covered} / 6", description="gap: Monitoring Unavailable",
                    key="m3")
with k4:
    sui.metric_card("Decisions logged", len(log.entries()),
                    description="hash chain verified" if log.verify() else "chain broken", key="m4")

tab = sui.tabs(["Triage", "Insights", "Audit trail"], value="Triage", key="tabs")

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
                              column_config={"Sev": st.column_config.TextColumn(width=58),
                                             "Problem": st.column_config.TextColumn(width=86),
                                             "Event": st.column_config.TextColumn(width=110)})
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
            st.html(ui.timeline(card))
            if card["related_problems"]:
                st.caption("Same entity within 60 min: " + ", ".join(card["related_problems"]))
        with c2, st.container(border=True):
            st.html(ui.eyebrow("Root-cause summary", "magnifying-glass"))
            st.markdown(card["root_cause_summary"])
            st.html(ui.eyebrow("Confidence", "gauge-high") + ui.meter(card["confidence"]))
            st.html(ui.kv([("confidence", f"{card['confidence']:.2f}"),
                           ("tokens in/out", f"{card['tokens_in']}/{card['tokens_out']}"),
                           ("cost", "$0.00" if not card["tokens_in"] or card["backend"].startswith("hf")
                            else "see KPIs")]))

        with st.container(border=True):
            locked = sum(s["needs_approval"] for s in card["step_plan"])
            st.html(ui.eyebrow(f"Runbook · {len(card['step_plan'])} steps · {locked} need approval", "list-check"))
            st.html(ui.runbook(card["step_plan"]))

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

        b1, b2, b3, _ = st.columns([1, 1, 1, 2.2])
        with b1:
            approve = sui.button("Approve", key=f"ap-{pid}", width="stretch")
        with b2:
            escalate = sui.button("Escalate", variant="secondary", key=f"es-{pid}", width="stretch")
        with b3:
            reject = sui.button("Reject", variant="destructive", key=f"rj-{pid}", width="stretch")
        decision = ("edited" if edit else "approved") if approve else "escalated" if escalate else \
            "rejected" if reject else None
        if decision:
            rec = AuditLog(AUDIT).append({"problem_id": pid, "decision": decision, "user": reviewer,
                                          "backend": card["backend"], "ticket": tools.draft_ticket(final)})
            st.toast(f"{decision.title()} by {reviewer} · hash {rec['hash'][:10]}…", icon="✅")

# ----------------------------------------------------------------------------- insights
elif tab == "Insights":
    cards = batch_cards()
    k = compute_kpis(cards, tools.alerts)
    a, b, c, d = st.columns(4)
    with a:
        sui.metric_card("SOP coverage", f"{k['sop_coverage_pct']}%", description="problems with a runbook", key="i1")
    with b:
        sui.metric_card("Escalated", f"{k['escalation_rate_pct']}%", description="documentation gaps", key="i2")
    with c:
        sui.metric_card("Steps gated", f"{k['steps_needing_approval_pct']}%", description="need human approval",
                        key="i3")
    with d:
        sui.metric_card("SecOps flags", k["secops_notifications"], description="auth / brute-force signals",
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
            marker=dict(color=[ui.AMBER, "#E0A030", ui.GREEN, ui.BLUE]), textinfo="value+percent initial"))
        fig.update_layout(title="From alert noise to actionable pages", **layout)
        st.plotly_chart(fig, width="stretch")
    with g2, st.container(border=True):
        counts = pd.DataFrame(cards)["event_type"].value_counts()
        fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h",
                               marker=dict(color=[ui.RED if e == "Monitoring Unavailable" else ui.AMBER
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

# ----------------------------------------------------------------------------- audit
else:
    entries = AuditLog(AUDIT).entries()
    ok = AuditLog(AUDIT).verify()
    st.html(ui.banner("Hash chain verified: no decision has been edited after the fact." if ok else
                      "Hash chain broken: the log was edited.", "warn" if ok else "crit",
                      "link" if ok else "triangle-exclamation"))
    if entries:
        df = pd.DataFrame(entries)
        df["hash"] = df["hash"].str[:16] + "…"
        st.dataframe(df[["ts", "problem_id", "decision", "user", "backend", "hash"]], hide_index=True,
                     width="stretch", height=520)
    else:
        st.caption("No decisions yet. Approve, escalate, or reject a card on the Triage tab.")
