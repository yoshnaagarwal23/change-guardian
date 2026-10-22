"""HTTP API so a ticketing tool (Remedy, ServiceNow) or a webhook could call the agent.

  uv run uvicorn api:create_app --factory --app-dir src --port 8000   # docs at /docs
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent import TriageAgent
from guardrails import AuditLog
from tools import TriageTools

AUDIT_PATH = Path("outputs/audit_log.jsonl")


class Decision(BaseModel):
    problem_id: str
    decision: Literal["approved", "edited", "escalated", "rejected"]
    user: str
    notes: str = ""
    card: dict | None = None


def create_app(tools: TriageTools | None = None, backend=None) -> FastAPI:
    tools = tools or TriageTools.from_disk()
    agent = TriageAgent(tools, backend)
    app = FastAPI(title="Change Guardian", version="0.2.0",
                  description="Human-in-the-loop incident triage for payment operations.")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "backend": getattr(backend, "name", "rules")}

    @app.get("/problems")
    def problems() -> list[dict]:
        df = tools.problems.sort_values(["severity", "first_seen"])
        cols = ["problem_id", "first_seen", "event_type", "severity_label", "entities", "alert_count"]
        return df[cols].astype(str).to_dict("records")

    @app.post("/triage/{problem_id}")
    def triage(problem_id: str) -> dict:
        try:
            return agent.triage(problem_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown problem {problem_id}")

    @app.post("/decision")
    def decision(d: Decision) -> dict:
        log = AuditLog(AUDIT_PATH)
        record = log.append(d.model_dump())
        return {"record": record, "audit_chain_ok": log.verify()}

    @app.get("/audit")
    def audit() -> dict:
        log = AuditLog(AUDIT_PATH)
        return {"entries": log.entries(), "audit_chain_ok": log.verify()}

    return app
