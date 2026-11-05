"""Change-risk scoring: the right deployment is flagged, and the card says why."""
import pandas as pd

from agent import TriageAgent
from risk_model import assess_changes, change_kind, recency, score_change
from schema import TriageCard
from tools import TriageTools

T0 = pd.Timestamp("2026-10-12 08:00:00")
CAUSE = {"entity_name": "host-catalog-db-01",
         "root_cause": "Prometheus metric queue_depth exceeded threshold > 500."}
SYMPTOMS = [{"entity_name": "B2B Checkout App"}]
DIAG = [CAUSE["root_cause"]]


def _change(cid, minutes_before, entity, description):
    return {"change_id": cid, "timestamp": T0 - pd.Timedelta(minutes=minutes_before),
            "entity": entity, "description": description}


def test_change_kind_ranks_infrastructure_above_docs():
    assert change_kind("Database schema migration for orders")[0] == "infrastructure"
    assert change_kind("Raise queue threshold in config")[0] == "configuration"
    assert change_kind("Deploy payment-api v2.4.1")[0] == "code release"
    assert change_kind("Update Grafana dashboard")[0] == "observability"
    assert change_kind("Rollback payment-api to v2.4.0")[0] == "rollback"


def test_recency_halves_every_two_hours_and_ignores_later_changes():
    assert recency(T0, T0)[0] == 1.0
    assert abs(recency(T0 - pd.Timedelta(hours=2), T0)[0] - 0.5) < 1e-9
    later = score_change(_change("CHG-9", -30, CAUSE["entity_name"], "Deploy v2"),
                         T0, CAUSE["entity_name"], set(), DIAG)
    assert later["risk_score"] == 0.0 and later["risk_level"] == "low"


def test_recent_config_change_on_cause_entity_is_the_high_risk_suspect():
    changes = [
        _change("CHG-1", 20, "host-catalog-db-01", "Lower queue_depth threshold in consumer config"),
        _change("CHG-2", 600, "B2B Checkout App", "Update dashboard tiles"),
    ]
    risk = assess_changes(changes, T0, CAUSE, SYMPTOMS, DIAG)
    assert risk["suspect"]["change_id"] == "CHG-1"
    assert risk["level"] == "high"
    assert [c["change_id"] for c in risk["changes"]] == ["CHG-1", "CHG-2"]
    assert any("probable-cause entity" in r for r in risk["suspect"]["reasons"])
    assert risk["changes"][1]["risk_level"] == "low"


def test_old_minor_change_is_not_a_suspect():
    risk = assess_changes([_change("CHG-3", 22 * 60, "B2B Checkout App", "Update docs")],
                          T0, CAUSE, SYMPTOMS, DIAG)
    assert risk["suspect"] is None and risk["level"] == "low"
    assert assess_changes([], T0, CAUSE, SYMPTOMS, DIAG) == {"changes": [], "suspect": None, "level": "none"}


def test_agent_card_flags_a_suspect_change_and_asks_for_approved_rollback(tools):
    problem = tools.problems.iloc[0]
    alerts = tools.problem_alerts(problem["problem_id"])
    cause_entity = alerts.sort_values("timestamp").iloc[0]["entity_name"]
    log = pd.DataFrame([{"change_id": "CHG-42",
                         "timestamp": problem["first_seen"] - pd.Timedelta(minutes=15),
                         "entity": cause_entity, "description": "Database index migration"}])
    with_log = TriageTools(tools.alerts, tools.problems, tools.tickets, tools.sops, log)

    card = TriageAgent(with_log).triage(problem["problem_id"])
    TriageCard.model_validate(card)
    assert card["change_risk"]["suspect"]["change_id"] == "CHG-42"
    assert card["recent_changes"][0]["risk_level"] in {"medium", "high"}
    step = next(s for s in card["step_plan"] if "CHG-42" in s["step"])
    assert step["risk"] == "change"
    assert any("CHG-42" in w for w in card["warnings"])


def test_card_without_deploy_log_has_no_change_risk(tools):
    card = TriageAgent(tools).triage(tools.problems.iloc[0]["problem_id"])
    assert card["recent_changes"] == []
    assert card["change_risk"]["level"] == "none"
