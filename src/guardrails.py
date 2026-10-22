"""Governance controls (Area 6): what the agent may suggest, and proof of what humans decided.

Preventative: every state-changing step on production needs human approval.
Detective:    every suggested step must be grounded in an SOP or a past ticket.
Evidence:     decisions are written to a hash-chained, tamper-evident audit log.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

_SECURITY = r"firewall|\bblock\b|credential|token|secops|ip range|certificate|authentication"
_CHANGE = (r"restart|roll ?back|rollback|reboot|\bkill\b|scale|scaling|increase|adjust|deploy|"
           r"hotfix|truncate|archive|clear|re-apply|reapply|toggle|update|route traffic|drain|"
           r"disable|failover|expand|expansion|purge|delete|patch")
_NON_PROD = r"stag|\bdev\b|-dev|test|sandbox|nonprod|non-prod|qa\b"
_STOP = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "if", "any",
         "is", "are", "be", "by", "from", "via", "at", "as", "it", "this", "that", "into",
         "less", "than", "more", "has", "have", "was", "were", "due"}
CONSISTENCY_MIN = 0.3


def classify_step(step: str) -> str:
    """'security' > 'change' > 'diagnostic' (the most restrictive label wins)."""
    low = step.lower()
    if re.search(_SECURITY, low):
        return "security"
    if re.search(_CHANGE, low):
        return "change"
    return "diagnostic"


def is_production(entity: str) -> bool:
    """Default-deny: on a payment network anything not clearly non-prod is prod."""
    return not re.search(_NON_PROD, entity.lower())


def approval_plan(steps: list[str], entities: list[str]) -> list[dict]:
    prod = any(is_production(e) for e in entities) if entities else True
    plan = []
    for step in steps:
        risk = classify_step(step)
        plan.append({"step": step, "risk": risk, "needs_approval": risk != "diagnostic" and prod})
    return plan


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9/_%.-]+", text.lower().rstrip(".")) if w not in _STOP}


def grounding_score(step: str, sources: list[str]) -> float:
    """Share of a step's content words that appear in the trusted sources."""
    words = {w.strip(".,") for w in _words(step)} - {""}
    if not words:
        return 0.0
    vocab = {w.strip(".,") for s in sources for w in _words(s)}
    return len(words & vocab) / len(words)


def alert_consistency(summary: str, diagnosis: str) -> float:
    """Share of the firing alert's content words that the summary mentions.

    Grounding alone can't catch a model that recites a real but unrelated precedent
    (every word is "grounded"). This checks the answer is about the alert that fired.
    """
    return grounding_score(diagnosis, [summary])


class AuditLog:
    """Append-only JSONL where each record carries the hash of the previous one."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    @staticmethod
    def _digest(prev_hash: str, body: dict) -> str:
        payload = prev_hash + json.dumps(body, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def append(self, event: dict) -> dict:
        records = self._records()
        prev = records[-1]["hash"] if records else "GENESIS"
        body = {"ts": datetime.now(timezone.utc).isoformat(), **event}
        record = {**body, "prev_hash": prev, "hash": self._digest(prev, body)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return record

    def verify(self) -> bool:
        prev = "GENESIS"
        for record in self._records():
            body = {k: v for k, v in record.items() if k not in {"prev_hash", "hash"}}
            if record.get("prev_hash") != prev or record.get("hash") != self._digest(prev, body):
                return False
            prev = record["hash"]
        return True

    def entries(self) -> list[dict]:
        return self._records()
