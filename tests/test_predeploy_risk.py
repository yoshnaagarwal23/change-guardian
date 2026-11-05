"""Pre-deploy risk: explainable score, Meta-style gates, recommendation only."""
import pandas as pd

from llm import Generation
from notify import send_risk_alert
from predeploy_risk import (CHANGE_TYPE_WEIGHTS, failure_rate, gate, level, llm_summary, load_inputs,
                            off_hours, score_all, score_pending)
from risk_model import change_kind

TIERS = pd.DataFrame([("Checkout", 1, 20), ("Docs", 3, 0)], columns=["entity", "tier", "downstream_dependents"])
FREEZE = pd.DataFrame([{"name": "peak", "start": "2026-10-15 00:00", "end": "2026-10-19 08:00"}])
HISTORY = pd.DataFrame({"entity": ["Checkout"] * 4 + ["Docs"] * 4,
                        "change_type": ["infrastructure"] * 4 + ["observability"] * 4,
                        "culprit": [True, True, True, False] + [False] * 4})


def _change(**kw):
    base = {"change_id": "C-1", "planned_time": "2026-10-20 10:00", "entity": "Docs",
            "change_type": "observability", "description": "x", "owner": "dev", "lines_changed": 10,
            "has_rollback_plan": True, "tested_in_canary": True}
    return {**base, **kw}


def test_change_type_weights_are_reused_from_the_post_incident_model():
    assert CHANGE_TYPE_WEIGHTS["infrastructure"] == change_kind("schema migration")[1] == 1.0
    assert CHANGE_TYPE_WEIGHTS["observability"] < CHANGE_TYPE_WEIGHTS["code release"]


def test_failure_rate_is_smoothed_away_from_0_and_1():
    rate, k, n = failure_rate(HISTORY, "Checkout", "infrastructure")
    assert (k, n) == (3, 4) and 0.5 < rate < 0.75
    unseen, _, n0 = failure_rate(HISTORY, "Docs", "code release")
    assert n0 == 0 and 0 < unseen < 1


def test_safe_change_lands_and_risky_change_in_freeze_is_blocked():
    safe = score_pending(_change(), TIERS, FREEZE, HISTORY)
    assert safe["level"] == "LOW" and safe["gate"]["label"] == "Safe to land" and not safe["gate"]["blocked"]
    risky = score_pending(_change(entity="Checkout", change_type="infrastructure", lines_changed=1500,
                                  has_rollback_plan=False, tested_in_canary=False,
                                  planned_time="2026-10-16 21:00"), TIERS, FREEZE, HISTORY)
    assert risky["level"] == "HIGH" and risky["gate"]["blocked"]
    assert risky["freeze"] == "peak" and risky["off_hours"] == "Friday evening"
    assert sum(c["contribution"] for c in risky["contributions"]) >= risky["risk_score"]  # clipped at 1
    assert all(r[0] in "+-" for r in risky["reasons"])


def test_safety_controls_lower_the_score():
    with_controls = score_pending(_change(entity="Checkout", change_type="code release"), TIERS, FREEZE, HISTORY)
    without = score_pending(_change(entity="Checkout", change_type="code release", has_rollback_plan=False,
                                    tested_in_canary=False), TIERS, FREEZE, HISTORY)
    assert round(without["risk_score"] - with_controls["risk_score"], 3) == 0.2


def test_gates_match_the_three_meta_style_levels():
    assert (level(0.34), level(0.35), level(0.64), level(0.65)) == ("LOW", "MEDIUM", "MEDIUM", "HIGH")
    assert gate("MEDIUM", None)["blocked"] is False
    assert gate("MEDIUM", "peak")["blocked"] is True and "freeze" in gate("MEDIUM", "peak")["label"]
    assert gate("HIGH", None)["label"] == "Blocked — requires senior approval"


def test_off_hours():
    assert off_hours(pd.Timestamp("2026-10-16 18:00")) == "Friday evening"
    assert off_hours(pd.Timestamp("2026-10-20 23:30")) == "overnight"
    assert off_hours(pd.Timestamp("2026-10-20 10:00")) is None


def test_demo_pending_changes_cover_all_three_levels():
    pending, tiers, freezes = load_inputs()
    history = pd.DataFrame(columns=["entity", "change_type", "culprit"])
    scored = score_all(pending, tiers, freezes, history)
    assert 14 <= len(pending) <= 16 and {c["level"] for c in scored} == {"LOW", "MEDIUM", "HIGH"}


def test_demo_high_changes_spread_out_and_not_all_in_the_freeze():
    from tools import TriageTools
    from predeploy_risk import DEPLOY_LOG, label_culprits
    pending, tiers, freezes = load_inputs()
    scored = score_all(pending, tiers, freezes, label_culprits(TriageTools.from_disk(changes_csv=DEPLOY_LOG)))
    high = [c for c in scored if c["level"] == "HIGH"]
    assert max(c["risk_score"] for c in scored) <= 0.95
    assert sum(c["freeze"] is None and c["off_hours"] is None for c in high) >= 2
    assert len({round(c["risk_score"], 2) for c in high}) == len(high)


def test_llm_summary_is_two_sentences_and_never_touches_the_score():
    class Fake:
        name = "fake"

        def generate(self, messages):
            return Generation("One. Two. Three, and it is actually LOW.", 10, 5)

    c = score_pending(_change(), TIERS, FREEZE, HISTORY)
    before = (c["risk_score"], c["level"])
    assert llm_summary(Fake(), c) == "One. Two."
    assert (c["risk_score"], c["level"]) == before


def test_webhook_fails_gracefully(monkeypatch):
    monkeypatch.delenv("CG_RISK_WEBHOOK_URL", raising=False)
    sent, msg = send_risk_alert("C-1", "Checkout", "dev", 0.9, "HIGH", ["a", "b", "c", "d"])
    assert sent is False and "no webhook" in msg
    sent, msg = send_risk_alert("C-1", "Checkout", "dev", 0.9, "HIGH", ["a"], url="http://127.0.0.1:9/hook",
                                timeout=0.5)
    assert sent is False and msg.startswith("webhook failed")
