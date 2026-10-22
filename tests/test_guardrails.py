"""Human-in-the-loop policy, grounding checks, and the tamper-evident audit trail."""
import json

import pytest

from guardrails import (CONSISTENCY_MIN, AuditLog, alert_consistency, approval_plan, classify_step,
                        grounding_score, is_production)


@pytest.mark.parametrize("step,expected", [
    ("Review the Failure Rate Analysis dashboard for the problem.", "diagnostic"),
    ("Restart the application process with updated memory limits.", "change"),
    ("Roll back the recent configuration patch.", "change"),
    ("Block the offending IP range at the perimeter firewall.", "security"),
    ("Correct expired tokens or credential entries.", "security"),
])
def test_classify_step(step, expected):
    assert classify_step(step) == expected


def test_everything_is_production_unless_clearly_non_production():
    assert is_production("host-payment-prod-01")
    assert is_production("PaymentService")  # default-deny for a payment network
    assert not is_production("staging-artifactory-02")


def test_state_changing_steps_on_production_need_approval():
    plan = approval_plan(
        ["Check GC metrics in Dynatrace.", "Restart the worker process."], ["host-worker-03"])
    assert [p["needs_approval"] for p in plan] == [False, True]


def test_grounding_score_rewards_steps_copied_from_sources():
    sources = ["Truncate or archive non-essential logs, clear temporary files."]
    assert grounding_score("Archive non-essential logs and clear temporary files.", sources) > 0.6
    assert grounding_score("Reinstall the kernel and email the CEO.", sources) < 0.3


def test_alert_consistency_catches_template_recall():
    alert = "Disk space capacity remaining less than 5% on /var/log."
    recalled = ("Extensive garbage collection (GC) cycles taking up 95% of active CPU cycles "
                "due to small JVM heap allocation settings.")
    paraphrase = "The /var/log volume on host-payment-prod-01 has under 5% disk space left."
    assert alert_consistency(paraphrase, alert) >= CONSISTENCY_MIN
    assert alert_consistency(recalled, alert) < CONSISTENCY_MIN


def test_audit_log_is_hash_chained_and_detects_tampering(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append({"problem_id": "P-1", "decision": "approved", "user": "swap"})
    log.append({"problem_id": "P-2", "decision": "escalated", "user": "swap"})
    assert log.verify()

    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    record = json.loads(lines[0])
    record["decision"] = "rejected"  # someone edits history
    lines[0] = json.dumps(record)
    (tmp_path / "audit.jsonl").write_text("\n".join(lines) + "\n")
    assert not AuditLog(tmp_path / "audit.jsonl").verify()
