"""Pre-deploy change risk (modeled on Meta's Diff Risk Score): score a change BEFORE it ships.

Separate from risk_model.score_change, which looks back after an incident. This looks
forward at a pending change and recommends a gate. It never deploys or blocks anything on
its own: a human can always approve with a written justification, and that is audited.

  score = clip(  W_HISTORY x failure rate     (smoothed share of past deploys on this entity +
                                              change type that were incident culprits)
               + W_TIER    x tier risk        (1 = critical service)
               + W_BLAST   x blast radius     (downstream dependents / BLAST_CAP)
               + W_TYPE    x change type      (risk_model's change-type weights)
               + W_SIZE    x change size      (log-scaled lines changed)
               - ROLLBACK_CREDIT - CANARY_CREDIT          (safety controls)
               + FREEZE_PENALTY + OFF_HOURS_PENALTY, 0, 1)  (timing)

  uv run python src/predeploy_risk.py      # print scores for the demo pending changes
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd

from risk_model import _DEFAULT_KIND, _KINDS, assess_changes, change_kind

# --------------------------------------------------------------------------- weights + thresholds
W_HISTORY, W_TIER, W_BLAST, W_TYPE, W_SIZE = 0.20, 0.20, 0.15, 0.15, 0.15
ROLLBACK_CREDIT, CANARY_CREDIT = 0.10, 0.10
FREEZE_PENALTY, OFF_HOURS_PENALTY = 0.20, 0.10
TIER_RISK = {1: 1.0, 2: 0.5, 3: 0.1}
UNKNOWN_TIER = 1                  # default-deny: an entity with no tier is treated as critical
BLAST_CAP = 20                    # dependents at which blast radius maxes out
SIZE_CAP_LINES = 2000             # lines at which change size maxes out (log scale)
HISTORY_CAP = 0.5                 # failure rate at which the history signal maxes out
PRIOR_STRENGTH = 3.0              # pseudo-deploys of smoothing toward the global culprit rate
OFF_HOURS = {"friday_from": 16, "night_from": 22, "night_to": 6}
LOW_MAX, MEDIUM_MAX = 0.35, 0.65  # < 0.35 LOW · 0.35-0.65 MEDIUM · >= 0.65 HIGH
CHANGE_TYPE_WEIGHTS = {name: w for name, _, w in _KINDS} | {_DEFAULT_KIND[0]: _DEFAULT_KIND[1]}

GATES = {"LOW": "Safe to land", "MEDIUM": "Needs extra review", "HIGH": "Blocked — requires senior approval"}
DIR = Path("data/changes")
PENDING, TIERS, FREEZES, DEPLOY_LOG = (DIR / "pending_demo.csv", DIR / "service_tiers.csv",
                                       DIR / "freeze_windows.csv", DIR / "deployments_demo.csv")


# --------------------------------------------------------------------------- history
def label_culprits(tools) -> pd.DataFrame:
    """Past deploys with their change type and whether the post-incident scorer made them a
    HIGH suspect for some problem (the demo log has no outcome column, so this is the label)."""
    culprits: set[str] = set()
    for p in tools.problems.itertuples():
        corr = tools.correlate(p.problem_id)
        cause = corr["probable_cause"]
        texts = [a["root_cause"] for a in [cause, *corr["symptoms"], *corr["telemetry_gaps"]]]
        changes = tools.find_recent_changes(str(p.entities).split("; "), p.first_seen)
        risk = assess_changes(changes, p.first_seen, cause, corr["symptoms"], texts)
        if risk["level"] == "high":
            culprits.add(risk["suspect"]["change_id"])
    log = tools.changes.copy() if tools.changes is not None else pd.DataFrame(
        columns=["change_id", "timestamp", "entity", "description"])
    log["change_type"] = [change_kind(str(d))[0] for d in log["description"]]
    log["culprit"] = log["change_id"].isin(culprits)
    return log


def failure_rate(history: pd.DataFrame, entity: str, change_type: str) -> tuple[float, int, int]:
    """(smoothed rate, culprits, deploys) for this entity + change type."""
    prior = float(history["culprit"].mean()) if len(history) else 0.15
    same = history[(history["entity"] == entity) & (history["change_type"] == change_type)]
    k, n = int(same["culprit"].sum()), len(same)
    return (k + PRIOR_STRENGTH * prior) / (n + PRIOR_STRENGTH), k, n


# --------------------------------------------------------------------------- timing
def in_freeze(when: pd.Timestamp, freezes: pd.DataFrame) -> str | None:
    for f in freezes.itertuples():
        if pd.Timestamp(f.start) <= when < pd.Timestamp(f.end):
            return f.name
    return None


def off_hours(when: pd.Timestamp) -> str | None:
    if when.day_of_week == 4 and when.hour >= OFF_HOURS["friday_from"]:
        return "Friday evening"
    if when.day_of_week >= 5:
        return "weekend"
    if when.hour >= OFF_HOURS["night_from"] or when.hour < OFF_HOURS["night_to"]:
        return "overnight"
    return None


# --------------------------------------------------------------------------- score + gate
def level(score: float) -> str:
    return "LOW" if score < LOW_MAX else "MEDIUM" if score < MEDIUM_MAX else "HIGH"


def gate(lvl: str, freeze: str | None) -> dict:
    """The recommendation only. blocked=True means "a human must approve", never "auto-stopped"."""
    if lvl == "HIGH":
        return {"label": GATES["HIGH"], "blocked": True, "approver": "senior engineer"}
    if lvl == "MEDIUM" and freeze:
        return {"label": f"{GATES['MEDIUM']} · blocked during freeze unless approved", "blocked": True,
                "approver": "reviewer"}
    if lvl == "MEDIUM":
        return {"label": GATES["MEDIUM"], "blocked": False, "approver": "second reviewer"}
    return {"label": GATES["LOW"], "blocked": False, "approver": None}


def score_pending(change: dict, tiers: pd.DataFrame, freezes: pd.DataFrame, history: pd.DataFrame) -> dict:
    entity, ctype = change["entity"], change["change_type"]
    when = pd.Timestamp(change["planned_time"])
    row = tiers[tiers["entity"] == entity]
    tier = int(row["tier"].iloc[0]) if len(row) else UNKNOWN_TIER
    dependents = int(row["downstream_dependents"].iloc[0]) if len(row) else BLAST_CAP
    rate, k, n = failure_rate(history, entity, ctype)
    lines = int(change["lines_changed"])
    freeze, off = in_freeze(when, freezes), off_hours(when)

    signals = [  # (signal, contribution, plain-English reason)
        ("history", W_HISTORY * min(1.0, rate / HISTORY_CAP),
         f"{rate:.0%} smoothed failure rate for {ctype} changes on {entity} ({k} culprit of {n} past)"),
        ("tier", W_TIER * TIER_RISK.get(tier, 1.0), f"tier-{tier} service" + (" (critical)" if tier == 1 else "")),
        ("blast radius", W_BLAST * min(1.0, dependents / BLAST_CAP), f"{dependents} downstream dependents"),
        ("change type", W_TYPE * CHANGE_TYPE_WEIGHTS.get(ctype, _DEFAULT_KIND[1]), f"{ctype} change"),
        ("size", W_SIZE * min(1.0, math.log10(1 + lines) / math.log10(1 + SIZE_CAP_LINES)),
         f"{lines} lines changed"),
        ("rollback plan", -ROLLBACK_CREDIT if change["has_rollback_plan"] else 0.0,
         "has a rollback plan" if change["has_rollback_plan"] else "no rollback plan"),
        ("canary", -CANARY_CREDIT if change["tested_in_canary"] else 0.0,
         "tested in canary" if change["tested_in_canary"] else "not tested in canary"),
        ("freeze", FREEZE_PENALTY if freeze else 0.0, f"inside freeze: {freeze}" if freeze else "outside any freeze"),
        ("timing", OFF_HOURS_PENALTY if off else 0.0, f"planned {off}" if off else "planned in business hours"),
    ]
    score = round(min(1.0, max(0.0, sum(c for _, c, _ in signals))), 3)
    lvl = level(score)
    contributions = [{"signal": s, "contribution": round(c, 3), "reason": f"{c:+.2f} {r}"} for s, c, r in signals]
    ranked = sorted((c for c in contributions if c["contribution"] > 0), key=lambda c: -c["contribution"])
    return {**change, "planned_time": when, "tier": tier, "freeze": freeze, "off_hours": off,
            "risk_score": score, "level": lvl, "gate": gate(lvl, freeze),
            "contributions": contributions, "reasons": [c["reason"] for c in ranked]}


def load_inputs(directory: Path = DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = Path(directory)
    pending = pd.read_csv(d / PENDING.name) if (d / PENDING.name).exists() else pd.DataFrame()
    tiers = pd.read_csv(d / TIERS.name) if (d / TIERS.name).exists() else pd.DataFrame(
        columns=["entity", "tier", "downstream_dependents"])
    freezes = pd.read_csv(d / FREEZES.name) if (d / FREEZES.name).exists() else pd.DataFrame(
        columns=["name", "start", "end"])
    return pending, tiers, freezes


def score_all(pending: pd.DataFrame, tiers: pd.DataFrame, freezes: pd.DataFrame,
              history: pd.DataFrame) -> list[dict]:
    scored = [score_pending(c, tiers, freezes, history) for c in pending.to_dict("records")]
    return sorted(scored, key=lambda c: -c["risk_score"])


# --------------------------------------------------------------------------- optional LLM summary
def summary_messages(scored: dict) -> list[dict]:
    return [{"role": "system", "content":
             "You explain a pre-deploy change risk score to an engineer. Write exactly two plain-English "
             "sentences from the facts given. Do not change, recompute, or dispute the score or level, and "
             "do not add facts."},
            {"role": "user", "content":
             f"Change {scored['change_id']} on {scored['entity']}: {scored['description']}. "
             f"Score {scored['risk_score']:.2f}, level {scored['level']} ({scored['gate']['label']}). "
             "Signals: " + "; ".join(c["reason"] for c in scored["contributions"])}]


def llm_summary(backend, scored: dict) -> str:
    """Two sentences from the computed reasons. Text only: the score and level are never touched."""
    text = backend.generate(summary_messages(scored)).text.strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:2])


def main() -> None:
    from tools import TriageTools
    pending, tiers, freezes = load_inputs()
    history = label_culprits(TriageTools.from_disk(changes_csv=DEPLOY_LOG))
    print(f"history: {len(history)} past deploys, {int(history['culprit'].sum())} culprits")
    for c in score_all(pending, tiers, freezes, history):
        flags = " ".join(x for x in [c["freeze"] and "FREEZE", c["off_hours"]] if x)
        print(f"{c['change_id']}  {c['risk_score']:.2f}  {c['level']:<6}  {c['entity']:<24} "
              f"{c['description'][:46]:<46}  {flags}")


if __name__ == "__main__":
    main()
