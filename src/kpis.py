"""KPIs for the business case (feeds Akhila's ROI estimate and Gabrielle's KPI table)."""
from __future__ import annotations

import statistics

import pandas as pd

from correlate import noise_summary
from llm import token_cost_usd


def compute_kpis(cards: list[dict], alerts: pd.DataFrame) -> dict:
    n = len(cards) or 1
    steps = [s for c in cards for s in c.get("step_plan", [])]
    noise = noise_summary(alerts)
    return {
        "problems_triaged": len(cards),
        "alerts_in": noise["alerts"],
        "pages_after_correlation": noise["problems"],
        "page_reduction_pct": noise["page_reduction_pct"],
        "sop_coverage_pct": round(100 * sum(c["sop_id"] is not None for c in cards) / n, 1),
        "escalation_rate_pct": round(100 * sum(c["escalate"] for c in cards) / n, 1),
        "secops_notifications": sum(c.get("notify_secops", False) for c in cards),
        "steps_suggested": len(steps),
        "steps_needing_approval_pct": round(100 * sum(s["needs_approval"] for s in steps) / max(1, len(steps)), 1),
        "llm_fallback_rate_pct": round(100 * sum(any("fallback" in w for w in c.get("warnings", []))
                                                 for c in cards) / n, 1),
        "median_latency_ms": round(statistics.median(c.get("latency_ms", 0) for c in cards), 1) if cards else 0.0,
        "tokens_in": sum(c.get("tokens_in", 0) for c in cards),
        "tokens_out": sum(c.get("tokens_out", 0) for c in cards),
        "est_llm_cost_usd": round(sum(token_cost_usd(c.get("backend", "rules"), c.get("tokens_in", 0),
                                                     c.get("tokens_out", 0)) for c in cards), 4),
    }
