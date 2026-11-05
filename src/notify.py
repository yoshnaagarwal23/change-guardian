"""Outbound notifications. Never raises: a missing or failing webhook must not break triage."""
from __future__ import annotations

import json
import os
import urllib.request

WEBHOOK_ENV = "CG_RISK_WEBHOOK_URL"


def send_risk_alert(change_id: str, entity: str, owner: str, score: float, level: str,
                    reasons: list[str], url: str | None = None, timeout: float = 5.0) -> tuple[bool, str]:
    """POST a high-risk change to a webhook (Slack-compatible `text` + structured fields).

    Returns (sent, message). With no URL configured it does nothing and says so.
    """
    url = url or os.getenv(WEBHOOK_ENV)
    if not url:
        return False, f"no webhook configured (set {WEBHOOK_ENV})"
    payload = {"text": f"[{level}] {change_id} on {entity} (owner {owner}) scored {score:.2f}: "
                       + "; ".join(reasons[:3]),
               "change_id": change_id, "entity": entity, "owner": owner, "score": round(score, 3),
               "level": level, "top_reasons": reasons[:3]}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300, f"webhook HTTP {resp.status}"
    except Exception as exc:  # noqa: BLE001 - notification failures are reported, not raised
        return False, f"webhook failed: {exc}"
