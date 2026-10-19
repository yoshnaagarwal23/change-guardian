# Change Guardian

AI incident triage assistant for the UGA FinTech Capstone with Mastercard.
A Dynatrace problem comes in → the agent finds the matching SOP and similar past tickets → drafts a Remedy ticket → a human approves.

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/load_data.py           # writes data/processed/*.csv
```

## Folder layout

```
data/raw/          client sample files (do not edit)
data/processed/    cleaned files from load_data.py (Yoshna)
sops/              one Markdown file per SOP (Andrew adds SOP-004..006)
src/load_data.py   cleaning + grouping (Yoshna)
src/tools.py       agent tools (Swap)
src/agent.py       LLM agent (Swap)
src/evaluate.py    scores agent output vs. real tickets (Yoshna)
app.py             Streamlit page (Swap)
```

## Agreed format (the contract between data and agent)

**Input: `data/processed/problems.csv`**, one row per Problem ID:

| column | example |
|---|---|
| problem_id | P-261000 |
| first_seen / last_seen | 2026-10-12 08:00:00 |
| alert_count | 2 |
| event_type | Slowdown (event type of the worst alert) |
| severity / severity_label | SEV-2 / SEV-2 (Major) |
| entities | B2B Checkout App; host-catalog-db-01 |
| root_causes | text from Dynatrace, separated by " \| " |

Also available: `alerts.csv` (each alert) and `tickets.csv` (past Remedy tickets with `remediation`).

**Output: the agent returns exactly this JSON for each problem:**

```json
{
  "problem_id": "P-261000",
  "event_type": "Slowdown",
  "severity": "SEV-2",
  "priority": "High",
  "assigned_group": "ITOM-Slowdown-SRE",
  "sop_id": "SOP-002",
  "root_cause_summary": "One or two sentences.",
  "next_steps": ["Step 1", "Step 2", "Step 3"],
  "similar_tickets": ["INC00000143202"],
  "escalate": false
}
```

- `sop_id` is `null` and `escalate` is `true` when no SOP covers the event type.
- The agent must cite the SOP and similar tickets and never invent numbers.

## Facts from the data

- Severity → priority is fixed: SEV-1 Critical, SEV-2 High, SEV-3 Medium, SEV-4 Low (100% of tickets).
- Assigned group is always `ITOM-<event type>-SRE` (100% of tickets).
- So priority and routing are rules (`load_data.py` has them). The LLM's job is the **root cause summary and next steps**.
- Only 6% of tickets link to a Problem ID in the alerts file, so match on event type + root cause, not Problem ID.
- SOPs exist only for Availability, Slowdown, and Error.

## Week of Oct 19 checklist

- [ ] Yoshna: `evaluate.py` (priority, group, SOP match + a 1–5 rating sheet for summaries)
- [ ] Swap: `tools.py` → `get_problem`, `find_sop`, `find_similar_tickets`, `draft_ticket`, `escalate`
- [ ] Swap: `agent.py` working on 1 problem in the terminal by Wed Oct 21
- [ ] Swap: `app.py` with problem feed → analysis → Approve / Edit / Escalate
- [ ] Both: 5 problems end to end by Sat Oct 25
