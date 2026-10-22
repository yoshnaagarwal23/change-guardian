"""Change Guardian triage agent.

Pipeline for one Dynatrace problem:
  1. tools: look up the problem, correlate its alerts, route it (deterministic)
  2. tools: find the SOP, rank its remediation branches, find precedent tickets
  3. model (optional): write the root-cause summary + next steps from that context only
  4. guardrails: drop ungrounded steps, fall back to the rules draft if nothing survives,
     tag every step with its risk and whether a human must approve it

Run from the repo root:
  uv run python src/agent.py --limit 5                  # rules only, no model
  uv run python src/agent.py --backend hf --limit 5     # local Hugging Face model
  uv run python src/agent.py --backend hf --adapter models/lora-mac
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from guardrails import CONSISTENCY_MIN, alert_consistency, approval_plan, grounding_score
from knowledge import precedent_matches, rank_texts
from llm import build_messages, get_backend, parse_model_output
from schema import TriageCard
from tools import TriageTools

RESULTS = Path("outputs/agent_results.jsonl")
GROUNDING_MIN = 0.5
_SECURITY_SIGNAL = r"authentication|signature|brute[- ]force|failed login|credential"


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=\.)\s+(?=[A-Z])", text.strip()) if s.strip()]


class TriageAgent:
    def __init__(self, tools: TriageTools, backend=None):
        self.tools = tools
        self.backend = backend

    # ------------------------------------------------------------------ context
    def build_context(self, problem_id: str) -> dict:
        t = self.tools
        problem = t.get_problem(problem_id)
        corr = t.correlate(problem_id)
        cause = corr["probable_cause"]
        sop = t.find_sop(problem["event_type"])
        query = f"{problem['event_type']} {cause['root_cause']}"
        remedies = t.rank_remedies(sop, cause["root_cause"]) if sop else []
        precedents = [{**p, "matches_alert": precedent_matches(cause["root_cause"], p["root_cause"])}
                      for p in t.find_similar_tickets(query, problem["event_type"])]
        changes = t.find_recent_changes(problem["entities"], problem["first_seen"])
        return {
            "problem_id": problem_id,
            "event_type": problem["event_type"],
            "severity": problem["severity"],
            "severity_label": problem["severity_label"],
            "entities": problem["entities"],
            **corr,
            "recent_changes": changes,
            "sop": None if sop is None else {
                "sop_id": sop.sop_id, "title": sop.title,
                "diagnostic_steps": sop.diagnostic_steps,
                "remedies": [r.text for r, _ in remedies]},
            # The two diagnostic steps most relevant to this alert, kept in SOP order.
            "focus_steps": [s for s in sop.diagnostic_steps
                            if s in {t for t, _ in rank_texts(sop.diagnostic_steps, cause["root_cause"], 2)}]
            if sop else [],
            "sop_all_text": sop.all_text if sop else [],
            "remedy_score": remedies[0][1] if remedies else 0.0,
            "precedents": precedents,
            "alert_texts": [a["root_cause"] for a in [cause, *corr["symptoms"], *corr["telemetry_gaps"]]],
        }

    # ------------------------------------------------------------ rules draft
    @staticmethod
    def rules_draft(ctx: dict) -> dict:
        cause, sop, precedents = ctx["probable_cause"], ctx["sop"], ctx["precedents"]
        summary = f"{cause['root_cause'].rstrip('.')} on {cause['entity_name']} is the probable cause."
        if ctx["symptoms"]:
            on = ", ".join(sorted({a["entity_name"] for a in ctx["symptoms"]}))
            summary += f" {len(ctx['symptoms'])} later alert(s) on {on} look like downstream symptoms."
        match = next((p for p in precedents if p.get("matches_alert")), None)
        if match:
            summary += f" Matches precedent {match['incident_id']} (seen {match['precedent_count']}x): {match['root_cause']}"

        steps: list[str] = []
        for c in ctx["recent_changes"][:1]:
            steps.append(f"Check change {c.get('change_id', '')} ({c.get('description', '')}); "
                         "if the problem started after it, request a rollback.")
        if sop:
            steps += ctx["focus_steps"]
            steps += [r.split(": ", 1)[-1] for r in sop["remedies"][:1]]
        elif precedents:
            steps += split_sentences(precedents[0]["remediation"])[:3]
        if not sop:
            steps.append(f"Escalate to the {ctx['event_type']} domain owner: no SOP covers this event "
                         "type yet (documentation gap).")
        return {"root_cause_summary": summary, "next_steps": steps[:6]}

    # ------------------------------------------------------------------ triage
    def triage(self, problem_id: str) -> dict:
        start = time.perf_counter()
        ctx = self.build_context(problem_id)
        warnings: list[str] = []
        answer = self.rules_draft(ctx)
        backend_name, tokens_in, tokens_out = "rules", 0, 0

        if self.backend is not None:
            gen = self.backend.generate(build_messages(ctx))
            tokens_in, tokens_out = gen.tokens_in, gen.tokens_out
            sources = (ctx["sop_all_text"] + ctx["alert_texts"]
                       + [p["remediation"] + " " + p["root_cause"] for p in ctx["precedents"]])
            try:
                proposed = parse_model_output(gen.text)
                diagnosis = ctx["probable_cause"]["root_cause"]
                if alert_consistency(proposed["root_cause_summary"], diagnosis) < CONSISTENCY_MIN:
                    raise ValueError("model summary does not match the alert (possible template recall)")
                kept = [s for s in proposed["next_steps"] if grounding_score(s, sources) >= GROUNDING_MIN]
                if len(kept) < len(proposed["next_steps"]):
                    warnings.append(f"dropped {len(proposed['next_steps']) - len(kept)} ungrounded step(s)")
                if not kept:
                    raise ValueError("no grounded steps")
                answer = {"root_cause_summary": proposed["root_cause_summary"], "next_steps": kept}
                backend_name = self.backend.name
            except ValueError as exc:
                warnings.append(f"llm_fallback to rules: {exc}")

        sop_id = ctx["sop"]["sop_id"] if ctx["sop"] else None
        if sop_id is None:
            warnings.append(f"No SOP covers '{ctx['event_type']}' alerts; escalated (documentation gap).")
        notify_secops = any(re.search(_SECURITY_SIGNAL, t.lower()) for t in ctx["alert_texts"])
        if notify_secops:
            warnings.append("Security signal: notify SecOps before any remediation.")

        top_prec = ctx["precedents"][0]["score"] if ctx["precedents"] else 0.0
        confidence = round(0.4 * (sop_id is not None) + 0.3 * min(1.0, 2 * ctx["remedy_score"])
                           + 0.3 * min(1.0, 2 * top_prec), 2)

        card = {
            "problem_id": problem_id,
            "event_type": ctx["event_type"],
            "severity": ctx["severity"],
            **self.tools.route(ctx["event_type"], ctx["severity"]),
            "sop_id": sop_id,
            **answer,
            "similar_tickets": [p["incident_id"] for p in ctx["precedents"]],
            "precedents": ctx["precedents"],
            "escalate": sop_id is None,
            "probable_cause": ctx["probable_cause"],
            "symptoms": ctx["symptoms"],
            "telemetry_gaps": ctx["telemetry_gaps"],
            "related_problems": ctx["related_problems"],
            "recent_changes": ctx["recent_changes"],
            "notify_secops": notify_secops,
            "step_plan": approval_plan(answer["next_steps"], ctx["entities"]),
            "confidence": confidence,
            "warnings": warnings,
            "backend": backend_name,
            "latency_ms": round(1000 * (time.perf_counter() - start), 1),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }
        return TriageCard.model_validate(card).model_dump()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="rules", choices=["rules", "hf", "anthropic"])
    parser.add_argument("--model", help="HF model id or Claude model id")
    parser.add_argument("--adapter", help="path to a LoRA adapter (hf backend)")
    parser.add_argument("--problem", action="append", help="problem id (repeatable)")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()

    kwargs = {}
    if args.backend == "hf":
        kwargs = {k: v for k, v in {"model_id": args.model, "adapter_path": args.adapter}.items() if v}
    elif args.backend == "anthropic" and args.model:
        kwargs = {"model": args.model}

    tools = TriageTools.from_disk()
    agent = TriageAgent(tools, get_backend(args.backend, **kwargs))
    ids = args.problem or tools.problems["problem_id"].head(args.limit).tolist()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for pid in ids:
            card = agent.triage(pid)
            f.write(json.dumps(card, default=str) + "\n")
            flag = "ESCALATE" if card["escalate"] else card["sop_id"]
            print(f"{pid}  {card['priority']:<8} {card['assigned_group']:<32} {flag:<9} "
                  f"{card['latency_ms']:>7.0f} ms  {card['backend']}")
    print(f"\nwrote {len(ids)} cards -> {args.out}")


if __name__ == "__main__":
    main()
