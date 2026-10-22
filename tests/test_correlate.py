"""Alert correlation: one page per fault instead of one page per symptom."""
import pandas as pd

from correlate import correlate_problem, noise_summary, related_problems


def _alerts(rows):
    df = pd.DataFrame(rows, columns=["problem_id", "timestamp", "event_type", "severity",
                                     "severity_rank", "entity_name", "root_cause"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_telemetry_gap_is_never_picked_as_probable_cause():
    # Arrange: the monitoring gap fires first, the real signal second.
    df = _alerts([
        ["P-1", "2026-10-12 08:00", "Monitoring Unavailable", "SEV-4", 4, "host-a", "heartbeat missing"],
        ["P-1", "2026-10-12 08:05", "Availability", "SEV-1", 1, "svc-b", "OOM crash"],
    ])
    # Act
    result = correlate_problem(df)
    # Assert
    assert result["probable_cause"]["root_cause"] == "OOM crash"
    assert [a["root_cause"] for a in result["telemetry_gaps"]] == ["heartbeat missing"]
    assert result["pages_before"] == 2 and result["pages_after"] == 1


def test_earliest_signal_is_probable_cause_and_later_ones_are_symptoms():
    df = _alerts([
        ["P-2", "2026-10-12 08:00", "Resource", "SEV-3", 3, "db-1", "memory 96%"],
        ["P-2", "2026-10-12 08:12", "Slowdown", "SEV-2", 2, "api-1", "P95 degraded"],
    ])
    result = correlate_problem(df)
    assert result["probable_cause"]["root_cause"] == "memory 96%"
    assert [a["root_cause"] for a in result["symptoms"]] == ["P95 degraded"]


def test_related_problems_share_an_entity_inside_the_window():
    df = _alerts([
        ["P-1", "2026-10-12 08:00", "Error", "SEV-2", 2, "host-x", "a"],
        ["P-2", "2026-10-12 08:30", "Slowdown", "SEV-3", 3, "host-x", "b"],
        ["P-3", "2026-10-12 08:31", "Slowdown", "SEV-3", 3, "host-y", "c"],
        ["P-4", "2026-10-12 14:00", "Slowdown", "SEV-3", 3, "host-x", "d"],
    ])
    assert related_problems(df, "P-1", window_minutes=60) == ["P-2"]


def test_noise_summary_on_client_data(tools):
    summary = noise_summary(tools.alerts)
    assert summary["alerts"] == 300
    assert summary["problems"] == 150
    assert summary["page_reduction_pct"] == 50.0
