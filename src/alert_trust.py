"""Alert trust score: how often an alert type actually needed action.

  trust = action_needed / total firings, per alert name (the alert definition a team tunes)

History comes from data/alert_dispositions.csv (demo data, scripts/make_alert_dispositions.py)
plus the dispositions recorded in this session. Informational only: a low score marks a
noisy alert worth a threshold review. Nothing is ever suppressed or hidden because of it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HISTORY = Path("data/alert_dispositions.csv")
ACTION, NO_ACTION = "action_needed", "no_action_needed"
TRUSTED, MIXED = 0.60, 0.30


def load_history(path: Path = HISTORY) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=["timestamp", "alert_id", "alert_name", "alert_type", "disposition", "engineer"])
    return pd.read_csv(path, parse_dates=["timestamp"])


def trust_table(history: pd.DataFrame, session: list[dict] | None = None) -> pd.DataFrame:
    """One row per alert name, least trusted first."""
    df = pd.concat([history, pd.DataFrame(session or [])], ignore_index=True)
    if df.empty:
        return pd.DataFrame(columns=["alert_name", "alert_type", "firings", "no_action_pct", "trust"])
    df["action"] = df["disposition"] == ACTION
    out = df.groupby("alert_name").agg(alert_type=("alert_type", "first"), firings=("action", "size"),
                                       actions=("action", "sum")).reset_index()
    out["trust"] = out["actions"] / out["firings"]
    out["no_action_pct"] = 100 * (1 - out["trust"])
    return out.drop(columns="actions").sort_values(["trust", "firings"], ascending=[True, False]) \
        .reset_index(drop=True)


def trust_lookup(table: pd.DataFrame) -> dict[str, tuple[float, int]]:
    return {r.alert_name: (r.trust, int(r.firings)) for r in table.itertuples()}


def badge(trust: float) -> tuple[str, str]:
    """(label, chip kind) for a trust score."""
    if trust >= TRUSTED:
        return "Trusted", "ok"
    if trust >= MIXED:
        return "Mixed", "med"
    return "Noisy — review threshold", "crit"


def disposition(alert: dict, decision: str, engineer: str) -> dict:
    """A session record shaped like a history row."""
    return {"timestamp": pd.Timestamp.now(), "alert_id": alert.get("alert_id", ""),
            "alert_name": alert["root_cause"], "alert_type": alert.get("event_type", ""),
            "disposition": decision, "engineer": engineer}
