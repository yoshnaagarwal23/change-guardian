"""The output contract (README "Agreed format"), enforced with Pydantic.

Required fields are the ones Yoshna's evaluate.py scores. Everything else is
context for the human reviewer and the KPI dashboard.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TriageCard(BaseModel):
    model_config = ConfigDict(extra="allow")

    problem_id: str
    event_type: str
    severity: str
    priority: str
    assigned_group: str
    sop_id: str | None
    root_cause_summary: str = Field(min_length=1)
    next_steps: list[str] = Field(min_length=1, max_length=8)
    similar_tickets: list[str]
    escalate: bool

    # Reviewer context
    probable_cause: dict | None = None
    symptoms: list[dict] = []
    telemetry_gaps: list[dict] = []
    related_problems: list[str] = []
    recent_changes: list[dict] = []
    notify_secops: bool = False
    step_plan: list[dict] = []
    confidence: float = 0.0
    warnings: list[str] = []

    # Cost / latency telemetry for the ROI case
    backend: str = "rules"
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


class ModelAnswer(BaseModel):
    """The only part the language model is allowed to write."""
    root_cause_summary: str = Field(min_length=1)
    next_steps: list[str] = Field(min_length=1, max_length=8)
