"""Change Guardian - live demo UI.

  uv run streamlit run app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent import TriageAgent  # noqa: E402
from guardrails import AuditLog  # noqa: E402
from kpis import compute_kpis  # noqa: E402
from llm import get_backend  # noqa: E402
from tools import TriageTools  # noqa: E402

AUDIT = Path("outputs/audit_log.jsonl")
REPLAY = Path("outputs/demo_cache.jsonl")
ADAPTER = Path("models/lora-mac")
RISK_BADGE = {"diagnostic": "🔎 diagnostic", "change": "🛠️ change", "security": "🛡️ security"}

st.set_page_config(page_title="Change Guardian", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
  .pill {display:inline-block;padding:2px 10px;border-radius:999px;font-size:0.82rem;
         margin-right:6px;border:1px solid rgba(128,128,128,.35)}
  .crit {background:#d6453520;border-color:#d64535} .high {background:#f28c2820;border-color:#f28c28}
  .med {background:#e0b10020;border-color:#e0b100} .low {background:#3a8f5c20;border-color:#3a8f5c}
  .cause {border-left:4px solid #f28c28;padding:6px 12px;margin:6px 0}
  .symptom {border-left:4px solid rgba(128,128,128,.5);padding:4px 12px;margin:4px 0;opacity:.85}
</style>""", unsafe_allow_html=True)
PRIORITY_CLASS = {"Critical": "crit", "High": "high", "Medium": "med", "Low": "low"}


# ----------------------------------------------------------------------------- data + models
@st.cache_resource
def load_tools() -> TriageTools:
    return TriageTools.from_disk()


@st.cache_resource(show_spinner="Loading model...")
def load_backend(mode: str):
    if mode == "Local model + LoRA (fine-tuned)":
        return get_backend("hf", adapter_path=str(ADAPTER))
    if mode == "Local model (base)":
        return get_backend("hf")
    if mode == "Claude (Anthropic API)":
        return get_backend("anthropic")
    return None


@st.cache_data
def load_replay() -> dict:
    if not REPLAY.exists():
        return {}
    return {c["problem_id"]: c for c in map(json.loads, REPLAY.read_text().splitlines()) if c}


tools = load_tools()
modes = ["Rules only (no model)", "Local model (base)"]
if (ADAPTER / "adapter_config.json").exists():
    modes.insert(1, "Local model + LoRA (fine-tuned)")
if os.getenv("ANTHROPIC_API_KEY"):
    modes.append("Claude (Anthropic API)")
if REPLAY.exists():
    modes.append("Replay (pre-computed, offline-safe)")

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🛡️ Change Guardian")
    st.caption("AI incident triage with a human in the loop")
    mode = st.selectbox("Engine", modes, index=1 if len(modes) > 2 else 0)
    reviewer = st.text_input("Reviewer", value="on-call SRE")
    sev_filter = st.multiselect("Severity", sorted(tools.problems["severity"].unique()),
                                default=sorted(tools.problems["severity"].unique()))
    evt_filter = st.multiselect("Event type", sorted(tools.problems["event_type"].unique()),
                                default=sorted(tools.problems["event_type"].unique()))
    st.divider()
    st.caption("Production changes always need human approval. Every decision is written to a "
               "hash-chained audit log.")

backend = None if mode.startswith(("Rules", "Replay")) else load_backend(mode)
agent = TriageAgent(tools, backend)


def triage(pid: str) -> dict:
    key = (mode, pid)
    cache = st.session_state.setdefault("cards", {})
    if key not in cache:
        if mode.startswith("Replay"):
            cache[key] = load_replay().get(pid) or TriageAgent(tools).triage(pid)
        else:
            cache[key] = agent.triage(pid)
    return cache[key]


# ----------------------------------------------------------------------------- header KPIs
noise = tools.alerts.groupby("problem_id").size()
log = AuditLog(AUDIT)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Alerts received", len(tools.alerts))
c2.metric("Pages after correlation", len(noise), f"-{100 * (1 - len(noise) / len(tools.alerts)):.0f}% noise",
          delta_color="inverse")
c3.metric("Event types with an SOP", f"{sum(tools.find_sop(e) is not None for e in tools.problems['event_type'].unique())}/6")
c4.metric("Decisions in audit log", len(log.entries()), "chain intact" if log.verify() else "TAMPERED",
          delta_color="normal" if log.verify() else "inverse")

tab_triage, tab_kpi, tab_audit = st.tabs(["Triage", "KPIs & model evaluation", "Audit trail"])

# ----------------------------------------------------------------------------- triage tab
with tab_triage:
    feed = tools.problems[tools.problems["severity"].isin(sev_filter)
                          & tools.problems["event_type"].isin(evt_filter)].sort_values(["severity", "first_seen"])
    if feed.empty:
        st.info("No problems match the filters.")
        st.stop()
    left, right = st.columns([1, 2.4], gap="large")
    with left:
        st.subheader(f"Problem feed ({len(feed)})")
        labels = {r.problem_id: f"{r.severity} · {r.problem_id} · {r.event_type} · {r.entities.split('; ')[0]}"
                  for r in feed.itertuples()}
        pid = st.radio("Open problems", list(labels), format_func=labels.get, label_visibility="collapsed")

    with right:
        with st.spinner("Triaging..."):
            card = triage(pid)
        pills = (f"<span class='pill {PRIORITY_CLASS.get(card['priority'], '')}'>Priority {card['priority']}</span>"
                 f"<span class='pill'>{card['assigned_group']}</span>"
                 f"<span class='pill'>{card['sop_id'] or 'NO SOP'}</span>"
                 f"<span class='pill'>confidence {card['confidence']:.2f}</span>"
                 f"<span class='pill'>{card['backend']} · {card['latency_ms']:.0f} ms</span>")
        if card["notify_secops"]:
            pills += "<span class='pill crit'>🛡️ notify SecOps</span>"
        st.subheader(f"{card['problem_id']} · {card['event_type']}")
        st.markdown(pills, unsafe_allow_html=True)
        if card["escalate"]:
            st.error("Escalated: no SOP covers this event type. Steps below come from past tickets only.")
        for w in card["warnings"]:
            if "No SOP" not in w:
                st.warning(w)

        st.markdown("##### Correlation: 1 page instead of "
                    f"{len(card['symptoms']) + len(card['telemetry_gaps']) + 1}")
        pc = card["probable_cause"]
        st.markdown(f"<div class='cause'><b>Probable cause</b> · {pc['entity_name']} · {pc['severity']}<br>"
                    f"{pc['root_cause']}<br><small>{pc['timestamp']}</small></div>", unsafe_allow_html=True)
        for a in card["symptoms"]:
            st.markdown(f"<div class='symptom'><b>Suppressed symptom</b> · {a['entity_name']} · {a['severity']}"
                        f"<br>{a['root_cause']}</div>", unsafe_allow_html=True)
        for a in card["telemetry_gaps"]:
            st.markdown(f"<div class='symptom'><b>Telemetry gap</b> · {a['entity_name']}<br>{a['root_cause']}</div>",
                        unsafe_allow_html=True)
        if card["related_problems"]:
            st.caption("Same entity within 60 min: " + ", ".join(card["related_problems"]))

        st.markdown("##### Proposed analysis")
        summary = st.text_area("Root-cause summary", card["root_cause_summary"], height=90)
        for s in card["step_plan"]:
            lock = "🔒 needs approval" if s["needs_approval"] else "auto-OK (read-only)"
            st.markdown(f"- {s['step']}  \n  <small>{RISK_BADGE[s['risk']]} · {lock}</small>",
                        unsafe_allow_html=True)
        edited_steps = st.text_area("Edit steps (one per line)", "\n".join(card["next_steps"]), height=120)

        with st.expander(f"Precedent tickets ({len(card['similar_tickets'])})"):
            for p in tools.find_similar_tickets(f"{card['event_type']} {pc['root_cause']}", card["event_type"]):
                st.markdown(f"**{p['incident_id']}** · seen {p['precedent_count']}x · {p['status']}  \n"
                            f"{p['root_cause']}  \n_Remediation:_ {p['remediation']}")
        if card["sop_id"]:
            with st.expander(f"{card['sop_id']} (full text)"):
                st.markdown(Path(f"data/sops/{card['sop_id']}.md").read_text())
        final = {**card, "root_cause_summary": summary,
                 "next_steps": [s for s in edited_steps.splitlines() if s.strip()]}
        with st.expander("Draft Remedy ticket"):
            st.json(tools.draft_ticket(final))

        b1, b2, b3, b4 = st.columns(4)
        changed = summary != card["root_cause_summary"] or final["next_steps"] != card["next_steps"]
        decision = None
        if b1.button("✅ Approve", type="primary", width="stretch"):
            decision = "edited" if changed else "approved"
        if b2.button("⬆️ Escalate", width="stretch"):
            decision = "escalated"
        if b3.button("❌ Reject", width="stretch"):
            decision = "rejected"
        if decision:
            rec = AuditLog(AUDIT).append({"problem_id": pid, "decision": decision, "user": reviewer,
                                          "backend": card["backend"], "ticket": tools.draft_ticket(final)})
            st.success(f"{decision.title()} by {reviewer}. Audit hash {rec['hash'][:12]}…")

# ----------------------------------------------------------------------------- KPI tab
with tab_kpi:
    st.subheader("Batch triage of all problems (rules engine)")
    if st.button("Run batch triage (150 problems)"):
        rules = TriageAgent(tools)
        st.session_state["batch"] = [rules.triage(p) for p in tools.problems["problem_id"]]
    if "batch" in st.session_state:
        cards = st.session_state["batch"]
        k = compute_kpis(cards, tools.alerts)
        cols = st.columns(4)
        cols[0].metric("SOP coverage", f"{k['sop_coverage_pct']}%")
        cols[1].metric("Escalated (documentation gaps)", f"{k['escalation_rate_pct']}%")
        cols[2].metric("Steps needing approval", f"{k['steps_needing_approval_pct']}%")
        cols[3].metric("SecOps notifications", k["secops_notifications"])
        st.bar_chart(pd.DataFrame(cards)["event_type"].value_counts())
        with st.expander("All KPI values"):
            st.json(k)

    st.subheader("Model evaluation (held-out alert types the model never saw)")
    comparison = Path("reports/model_comparison.json")
    if comparison.exists():
        table = pd.DataFrame(json.loads(comparison.read_text())).T
        keep = [c for c in ["alert_consistency", "alert_branch_hit_rate", "alert_grounding",
                            "alert_json_valid_rate", "ticket_rouge_l"] if c in table]
        st.dataframe(table[keep].style.format("{:.2f}"), width="stretch")
        st.caption("consistency = summary names the alert that fired · branch hit = picked the right SOP "
                   "remedy · grounding = steps traceable to SOP/tickets")
    else:
        st.info("Run `uv run python src/compare_models.py models/lora-mac` to compare models.")

# ----------------------------------------------------------------------------- audit tab
with tab_audit:
    entries = AuditLog(AUDIT).entries()
    st.subheader("Tamper-evident decision log")
    (st.success if AuditLog(AUDIT).verify() else st.error)(
        "Hash chain verified" if AuditLog(AUDIT).verify() else "Hash chain broken: the log was edited")
    if entries:
        st.dataframe(pd.DataFrame(entries)[["ts", "problem_id", "decision", "user", "backend", "hash"]],
                     width="stretch")
    else:
        st.caption("No decisions yet.")
