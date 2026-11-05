"""Alert trust score: share of past firings that needed action, per alert name."""
import pandas as pd

import ui_components as ui
from alert_trust import ACTION, NO_ACTION, badge, disposition, load_history, trust_lookup, trust_table

ALERT = {"root_cause": "Disk full.", "event_type": "Resource"}


def _hist(name, actions, noise):
    return [{"alert_name": name, "alert_type": "Resource", "disposition": ACTION}] * actions + \
           [{"alert_name": name, "alert_type": "Resource", "disposition": NO_ACTION}] * noise


def test_trust_is_action_share_sorted_least_trusted_first():
    hist = pd.DataFrame(_hist("Disk full.", 1, 9) + _hist("NPE.", 7, 3))
    table = trust_table(hist)
    assert list(table["alert_name"]) == ["Disk full.", "NPE."]
    assert table.loc[0, "trust"] == 0.1 and table.loc[0, "firings"] == 10 and table.loc[0, "no_action_pct"] == 90


def test_session_dispositions_move_the_score():
    hist = pd.DataFrame(_hist("Disk full.", 1, 1))
    session = [disposition(ALERT, NO_ACTION, "sre"), disposition(ALERT, NO_ACTION, "sre")]
    assert trust_lookup(trust_table(hist, session))["Disk full."] == (0.25, 4)


def test_badge_thresholds():
    assert badge(0.6) == ("Trusted", "ok")
    assert badge(0.3)[0] == "Mixed" and badge(0.59)[0] == "Mixed"
    assert badge(0.29) == ("Noisy — review threshold", "crit")


def test_demo_history_is_mostly_noise_but_varies_by_alert():
    hist = load_history()
    assert 450 <= len(hist) <= 550
    assert 0.75 <= (hist["disposition"] == NO_ACTION).mean() <= 0.85
    levels = {badge(t)[0] for t in trust_table(hist)["trust"]}
    assert levels == {"Trusted", "Mixed", "Noisy — review threshold"}


def test_timeline_shows_badge_but_never_drops_a_noisy_alert():
    card = {"probable_cause": {"timestamp": "2026-10-12 09:11:00", "severity": "SEV-1", "entity_name": "db",
                               "root_cause": "Disk full."}, "symptoms": [], "telemetry_gaps": []}
    html = ui.timeline(card, {"Disk full.": (0.1, 84)})
    assert "Disk full." in html and "Noisy — review threshold · Trust 10% (n=84)" in html
