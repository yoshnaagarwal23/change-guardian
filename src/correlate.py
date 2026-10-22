"""Alert correlation: collapse a problem's alerts into one actionable incident.

The sponsor's top pain is cascading noise: one slow database pages engineers once per
dependent service. Dynatrace already groups alerts into Problems; this module decides,
inside a problem, which alert is the probable cause and which are symptoms, and links
problems that hit the same entity close together in time (a possible alert storm).
"""
from __future__ import annotations

import pandas as pd

# Losing telemetry is a gap in visibility, not a cause of user impact.
TELEMETRY_GAP = "Monitoring Unavailable"

_ALERT_FIELDS = ["timestamp", "event_type", "severity", "entity_name", "root_cause"]


def _records(df: pd.DataFrame) -> list[dict]:
    out = df[_ALERT_FIELDS].copy()
    out["timestamp"] = out["timestamp"].astype(str)
    return out.to_dict("records")


def correlate_problem(alerts: pd.DataFrame) -> dict:
    """Pick the probable cause: the earliest real signal (severity breaks ties).

    In a cascade the first alert is usually closest to the fault; later alerts on
    dependent services are symptoms. Telemetry gaps are reported separately.
    """
    alerts = alerts.sort_values(["timestamp", "severity_rank"])
    gaps = alerts[alerts["event_type"] == TELEMETRY_GAP]
    signals = alerts[alerts["event_type"] != TELEMETRY_GAP]
    pool = signals if len(signals) else gaps
    cause = pool.iloc[[0]]
    symptoms = signals.drop(cause.index, errors="ignore")
    gaps = gaps.drop(cause.index, errors="ignore")
    return {
        "probable_cause": _records(cause)[0],
        "symptoms": _records(symptoms),
        "telemetry_gaps": _records(gaps),
        "pages_before": int(len(alerts)),
        "pages_after": 1,
    }


def related_problems(alerts: pd.DataFrame, problem_id: str, window_minutes: int = 60) -> list[str]:
    """Other problems on the same entities within +/- window of this problem."""
    mine = alerts[alerts["problem_id"] == problem_id]
    if mine.empty:
        return []
    window = pd.Timedelta(minutes=window_minutes)
    start, end = mine["timestamp"].min() - window, mine["timestamp"].max() + window
    others = alerts[(alerts["problem_id"] != problem_id)
                    & alerts["entity_name"].isin(mine["entity_name"])
                    & alerts["timestamp"].between(start, end)]
    return sorted(others["problem_id"].unique().tolist())


def noise_summary(alerts: pd.DataFrame) -> dict:
    n_alerts, n_problems = len(alerts), alerts["problem_id"].nunique()
    return {
        "alerts": int(n_alerts),
        "problems": int(n_problems),
        "page_reduction_pct": round(100 * (1 - n_problems / n_alerts), 1) if n_alerts else 0.0,
        "telemetry_gap_alerts": int((alerts["event_type"] == TELEMETRY_GAP).sum()),
    }
