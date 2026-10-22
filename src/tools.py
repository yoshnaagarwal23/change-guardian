"""Agent tools: the deterministic half of the agent.

Everything a script can do reliably lives here (Acid Test 2): look up the problem,
correlate its alerts, route it, find the SOP and precedents, draft the Remedy ticket.
The language model only writes the root-cause summary and next steps on top of this.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from correlate import correlate_problem, related_problems
from knowledge import Remedy, Sop, TicketIndex, load_sops, rank_remedies
from load_data import EVENT_TO_SOP, SEVERITY_TO_PRIORITY

CHANGES = Path("data/changes/deployments.csv")


class TriageTools:
    def __init__(self, alerts: pd.DataFrame, problems: pd.DataFrame, tickets: pd.DataFrame,
                 sops: dict[str, Sop], changes: pd.DataFrame | None = None):
        self.alerts = alerts
        self.problems = problems
        self.tickets = tickets
        self.sops = sops
        self.changes = changes
        self.ticket_index = TicketIndex(tickets)

    @classmethod
    def from_disk(cls, processed_dir: Path = Path("data/processed"),
                  sops_dir: Path = Path("data/sops"), changes_csv: Path = CHANGES) -> "TriageTools":
        processed_dir = Path(processed_dir)
        alerts = pd.read_csv(processed_dir / "alerts.csv", parse_dates=["timestamp"])
        problems = pd.read_csv(processed_dir / "problems.csv", parse_dates=["first_seen", "last_seen"])
        tickets = pd.read_csv(processed_dir / "tickets.csv", parse_dates=["timestamp"])
        changes = pd.read_csv(changes_csv, parse_dates=["timestamp"]) if Path(changes_csv).exists() else None
        return cls(alerts, problems, tickets, load_sops(sops_dir), changes)

    # --- lookups -------------------------------------------------------------
    def get_problem(self, problem_id: str) -> dict:
        rows = self.problems[self.problems["problem_id"] == problem_id]
        if rows.empty:
            raise KeyError(f"Unknown problem {problem_id}")
        problem = rows.iloc[0].to_dict()
        problem["entities"] = str(problem["entities"]).split("; ")
        problem["alerts"] = self.problem_alerts(problem_id)
        return problem

    def problem_alerts(self, problem_id: str) -> pd.DataFrame:
        return self.alerts[self.alerts["problem_id"] == problem_id]

    def correlate(self, problem_id: str) -> dict:
        result = correlate_problem(self.problem_alerts(problem_id))
        result["related_problems"] = related_problems(self.alerts, problem_id)
        return result

    # --- routing (fixed rules, 100% consistent with the ticket history) -------
    @staticmethod
    def route(event_type: str, severity: str) -> dict:
        return {"priority": SEVERITY_TO_PRIORITY[severity], "assigned_group": f"ITOM-{event_type}-SRE"}

    # --- knowledge -------------------------------------------------------------
    def find_sop(self, event_type: str) -> Sop | None:
        sop_id = EVENT_TO_SOP.get(event_type)
        return self.sops.get(sop_id) if sop_id else None

    def rank_remedies(self, sop: Sop, query: str, k: int = 2) -> list[tuple[Remedy, float]]:
        return rank_remedies(sop, query, k)

    def find_similar_tickets(self, query: str, event_type: str | None = None, k: int = 3,
                             exclude: set[str] | None = None) -> list[dict]:
        return self.ticket_index.search(query, event_type, k, exclude)

    def find_recent_changes(self, entities: list[str], before: pd.Timestamp,
                            window_hours: int = 24) -> list[dict]:
        """Hook for the change-risk model (Topic 1). Empty until a deploy log exists."""
        if self.changes is None:
            return []
        c = self.changes
        mask = (c["entity"].isin(entities)
                & c["timestamp"].between(before - pd.Timedelta(hours=window_hours), before))
        return c[mask].sort_values("timestamp", ascending=False).to_dict("records")

    # --- outputs ---------------------------------------------------------------
    @staticmethod
    def draft_ticket(card: dict) -> dict:
        """Remedy-shaped ticket, same columns as the client's ticket export."""
        cause = card.get("probable_cause") or {}
        return {
            "Dynatrace Problem ID": card["problem_id"],
            "Incident Summary": f"[{card['event_type']}] {cause.get('root_cause', '')} "
                                f"on {cause.get('entity_name', 'unknown entity')}".strip(),
            "Remedy Priority": card["priority"],
            "Dynatrace Severity": card["severity"],
            "Status": "Pending Approval",
            "Assigned Group": card["assigned_group"],
            "Root Cause Details": card["root_cause_summary"],
            "End User / Analyst Remediation Action": " ".join(card["next_steps"]),
        }

    @staticmethod
    def escalate(card: dict, reason: str) -> dict:
        return {"problem_id": card["problem_id"], "escalate": True, "reason": reason,
                "assigned_group": card["assigned_group"]}
