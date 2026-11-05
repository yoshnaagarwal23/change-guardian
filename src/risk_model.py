"""Change-risk scoring (Topic 1): which recent deployment most likely caused this problem.

Input is what TriageTools.find_recent_changes returns (rows of the deploy log
`data/changes/deployments.csv`: change_id, timestamp, entity, description) plus the
correlated problem. Each change gets a 0-1 suspicion score, a level, and the reasons
behind it, so the reviewer sees *why* a change was flagged, not just that it was.

It is deliberately a transparent weighted rule, like the routing rules (Acid Test 2):
there are no change records with outcomes to train on yet. Once the sponsors share
change records linked to incidents, these weights are the baseline a model must beat.

  score = recency x (0.25 + 0.75 x fit)
    recency  exponential decay since the change, half-life 2 h (a gate: old changes stay low)
    fit      0.4 entity     1.0 on the probable-cause entity, 0.5 on a symptom entity
           + 0.3 kind       how disruptive the change type is (infra/data > config > code)
           + 0.3 relevance  share of the description's words that appear in the alerts

Recency multiplies instead of adding because deploys repeat on the same few entities:
as a plain sum, a 10-hour-old migration on the right host still scored "medium".
"""
from __future__ import annotations

import math
import re

import pandas as pd

from guardrails import grounding_score

FIT_WEIGHTS = {"entity": 0.4, "kind": 0.3, "relevance": 0.3}
FLOOR = 0.25  # share of the score a recent change gets even with no fit
HALF_LIFE_HOURS = 2.0
HIGH, MEDIUM = 0.55, 0.3

# First match wins, so the most specific / most disruptive kinds come first.
_KINDS: list[tuple[str, str, float]] = [
    ("rollback", r"roll ?back|revert", 0.4),
    ("infrastructure", r"schema|migration|database|\bdb\b|index|network|firewall|dns|"
                       r"load ?balancer|certificate|\bcert\b|tls|kernel|os patch|storage|disk|"
                       r"capacity|scal(e|ing)|failover|cluster", 1.0),
    ("configuration", r"config|feature flag|flag|toggle|threshold|timeout|pool size|"
                      r"\benv\b|environment variable|parameter|limit", 0.8),
    ("code release", r"deploy|release|upgrade|hotfix|patch|version|\bv\d|build", 0.6),
    ("observability", r"dashboard|monitor|alert rule|logging|log level|docs?\b|documentation", 0.1),
]
_DEFAULT_KIND = ("other", 0.5)


def change_kind(description: str) -> tuple[str, float]:
    low = description.lower()
    for name, pattern, weight in _KINDS:
        if re.search(pattern, low):
            return name, weight
    return _DEFAULT_KIND


def recency(change_ts: pd.Timestamp, first_seen: pd.Timestamp) -> tuple[float, float]:
    """(signal, minutes before the problem). A change after first_seen can't be the cause."""
    minutes = (pd.Timestamp(first_seen) - pd.Timestamp(change_ts)).total_seconds() / 60
    if minutes < 0:
        return 0.0, minutes
    return math.exp(-math.log(2) * minutes / (60 * HALF_LIFE_HOURS)), minutes


def level(score: float) -> str:
    return "high" if score >= HIGH else "medium" if score >= MEDIUM else "low"


def _ago(minutes: float) -> str:
    return f"{minutes:.0f} min" if minutes < 120 else f"{minutes / 60:.1f} h"


def score_change(change: dict, first_seen: pd.Timestamp, cause_entity: str | None,
                 symptom_entities: set[str], diagnoses: list[str]) -> dict:
    description = str(change.get("description", ""))
    entity = change.get("entity")
    reasons: list[str] = []

    rec, minutes = recency(change["timestamp"], first_seen)
    if minutes < 0:
        reasons.append("deployed after the problem started, so it is not the trigger")
    else:
        reasons.append(f"deployed {_ago(minutes)} before the problem started")

    if entity == cause_entity:
        ent = 1.0
        reasons.append(f"touches the probable-cause entity {entity}")
    elif entity in symptom_entities:
        ent = 0.5
        reasons.append(f"touches {entity}, which only shows downstream symptoms")
    else:
        ent = 0.0

    kind, kind_w = change_kind(description)
    reasons.append(f"{kind} change")

    rel = grounding_score(description, diagnoses) if diagnoses else 0.0
    if rel >= 0.3:
        reasons.append(f"description overlaps the alert diagnosis ({rel:.0%} of its words)")

    signals = {"recency": rec, "entity": ent, "kind": kind_w, "relevance": rel}
    fit = sum(FIT_WEIGHTS[k] * signals[k] for k in FIT_WEIGHTS)
    score = round(rec * (FLOOR + (1 - FLOOR) * fit), 3)
    return {**change, "kind": kind, "minutes_before": round(minutes, 1),
            "signals": {k: round(v, 3) for k, v in signals.items()},
            "risk_score": score, "risk_level": level(score), "reasons": reasons}


def assess_changes(changes: list[dict], first_seen: pd.Timestamp, probable_cause: dict | None,
                   symptoms: list[dict], diagnoses: list[str]) -> dict:
    """Score every recent change; the top one is the suspect if it is medium or high risk."""
    cause_entity = (probable_cause or {}).get("entity_name")
    symptom_entities = {s["entity_name"] for s in symptoms} - {cause_entity}
    scored = sorted((score_change(c, first_seen, cause_entity, symptom_entities, diagnoses)
                     for c in changes), key=lambda c: c["risk_score"], reverse=True)
    suspect = scored[0] if scored and scored[0]["risk_level"] != "low" else None
    return {"changes": scored, "suspect": suspect,
            "level": suspect["risk_level"] if suspect else ("low" if scored else "none")}
