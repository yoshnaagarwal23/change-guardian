"""Load and clean the client sample data.

Run from the repo root:  python src/load_data.py
Outputs (in data/processed/):
  alerts.csv    - one row per Dynatrace alert, cleaned
  problems.csv  - one row per Problem ID (alerts grouped)
  tickets.csv   - one row per Remedy ticket, cleaned
"""
from pathlib import Path
import re

import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")
SOPS = Path("data/sops")

# SEV-1 is the most severe. Used to pick the "highest" severity in a problem.
SEVERITY_RANK = {"SEV-1": 1, "SEV-2": 2, "SEV-3": 3, "SEV-4": 4, "SEV-5": 5}

# Confirmed from the tickets file: severity -> priority is a fixed 1-to-1 mapping.
SEVERITY_TO_PRIORITY = {
    "SEV-1": "Critical",
    "SEV-2": "High",
    "SEV-3": "Medium",
    "SEV-4": "Low",
    "SEV-5": "Low",  # not in tickets; assumed
}

# Which SOP covers which event type. None = no SOP yet.
# SOP-005 (Synthetic & RUM) is user-experience monitoring, not OneAgent/ActiveGate
# connectivity, so "Monitoring Unavailable" stays uncovered until Andrew writes one.
EVENT_TO_SOP = {
    "Availability": "SOP-001",
    "Slowdown": "SOP-002",
    "Error": "SOP-003",
    "Resource": "SOP-004",
    "Monitoring Unavailable": None,
    "Custom Alert": "SOP-006",
}


def sev_code(label: str) -> str:
    """'SEV-2 (Major)' -> 'SEV-2'"""
    return label.split(" ")[0].strip()


def load_alerts() -> pd.DataFrame:
    df = pd.read_csv(RAW / "dynatrace_alerts.csv")
    df = df.rename(columns={
        "Problem ID": "problem_id",
        "Timestamp (UTC)": "timestamp",
        "Event Type": "event_type",
        "Severity Level": "severity_label",
        "Entity Type": "entity_type",
        "Entity Name": "entity_name",
        "Root Cause Diagnosis": "root_cause",
        "Log Detail Evidence": "log_evidence",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%y %H:%M")
    df["severity"] = df["severity_label"].map(sev_code)
    df["severity_rank"] = df["severity"].map(SEVERITY_RANK)
    return df.sort_values("timestamp").reset_index(drop=True)


def build_problems(alerts: pd.DataFrame) -> pd.DataFrame:
    """Group alerts by Problem ID. The problem's severity is its worst alert."""
    rows = []
    for pid, g in alerts.groupby("problem_id"):
        worst = g.loc[g["severity_rank"].idxmin()]
        rows.append({
            "problem_id": pid,
            "first_seen": g["timestamp"].min(),
            "last_seen": g["timestamp"].max(),
            "alert_count": len(g),
            "event_type": worst["event_type"],      # event type of the worst alert
            "all_event_types": "; ".join(sorted(g["event_type"].unique())),
            "severity": worst["severity"],
            "severity_label": worst["severity_label"],
            "entities": "; ".join(sorted(g["entity_name"].unique())),
            "root_causes": " | ".join(g["root_cause"].unique()),
        })
    return pd.DataFrame(rows).sort_values("first_seen").reset_index(drop=True)


def load_tickets() -> pd.DataFrame:
    df = pd.read_excel(RAW / "remedy_tickets.xlsx")
    df = df.rename(columns={
        "Incident ID": "incident_id",
        "Dynatrace Problem ID": "problem_id",
        "Timestamp (UTC)": "timestamp",
        "Incident Summary": "summary",
        "Remedy Priority": "priority",
        "Dynatrace Severity": "severity_label",
        "Status": "status",
        "Assigned Group": "assigned_group",
        "Root Cause Details": "root_cause",
        "End User / Analyst Remediation Action": "remediation",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["severity"] = df["severity_label"].map(sev_code)
    # Summary starts with "[Event Type] ..."
    df["event_type"] = df["summary"].str.extract(r"^\[(.*?)\]", expand=False)
    return df


def split_sops(md_path: Path) -> None:
    """Split the SOP markdown (converted from the .docx) into one file per SOP."""
    text = md_path.read_text()
    parts = re.split(r"(?m)^# (?=SOP-\d+)", text)
    SOPS.mkdir(parents=True, exist_ok=True)
    for part in parts:
        if not part.strip().startswith("SOP-"):
            continue
        sop_id = part.split(":")[0].strip()
        (SOPS / f"{sop_id}.md").write_text("# " + part.strip() + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    alerts = load_alerts()
    problems = build_problems(alerts)
    tickets = load_tickets()

    alerts.to_csv(OUT / "alerts.csv", index=False)
    problems.to_csv(OUT / "problems.csv", index=False)
    tickets.to_csv(OUT / "tickets.csv", index=False)

    print(f"alerts:   {len(alerts)} rows")
    print(f"problems: {len(problems)} rows")
    print(f"tickets:  {len(tickets)} rows")
    linked = tickets["problem_id"].isin(alerts["problem_id"]).mean()
    print(f"tickets linked to an alert problem: {linked:.0%}")
