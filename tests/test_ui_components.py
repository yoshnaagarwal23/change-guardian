"""The HTML building blocks of the console."""
import ui_components as ui

CARD = {
    "probable_cause": {"timestamp": "2026-10-12 09:11:00", "event_type": "Availability", "severity": "SEV-1",
                       "entity_name": "Mobile API Gateway", "root_cause": "Health check timeout <script>x</script>"},
    "symptoms": [{"timestamp": "2026-10-12 09:20:00", "event_type": "Slowdown", "severity": "SEV-3",
                  "entity_name": "api-2", "root_cause": "P95 degraded"}],
    "telemetry_gaps": [{"timestamp": "2026-10-12 09:30:00", "event_type": "Monitoring Unavailable",
                        "severity": "SEV-4", "entity_name": "host-frontend-05", "root_cause": "heartbeat missing"}],
}


def test_icons_are_masks_that_follow_text_color_and_exist_on_disk():
    # st.html strips inline <svg>, so icons are CSS masks over static files tinted with currentColor.
    html = ui.icon("lock")
    assert "app/static/icons/lock.svg" in html and "currentColor" in html
    for name in ["shield-halved", "crosshairs", "eye-slash", "bell", "lock", "wrench", "user-shield"]:
        assert (ui.ICONS / f"{name}.svg").exists()


def test_timeline_orders_cause_symptoms_gaps_and_escapes_alert_text():
    html = ui.timeline(CARD)
    assert html.index("Probable cause") < html.index("Suppressed symptom") < html.index("Telemetry gap")
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_runbook_marks_steps_that_need_approval():
    html = ui.runbook([{"step": "Check GC metrics.", "risk": "diagnostic", "needs_approval": False},
                       {"step": "Restart the worker.", "risk": "change", "needs_approval": True}])
    assert html.count("needs approval") == 1 and html.count("read-only") == 1


def test_precedents_say_whether_they_match_the_alert():
    html = ui.precedents([
        {"incident_id": "INC1", "precedent_count": 5, "status": "Resolved", "root_cause": "OOM",
         "remediation": "Restart.", "matches_alert": True},
        {"incident_id": "INC2", "precedent_count": 9, "status": "Closed", "root_cause": "GC",
         "remediation": "Tune.", "matches_alert": False}])
    assert "matches this alert" in html and "different cause" in html
