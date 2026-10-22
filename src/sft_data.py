"""Build supervised fine-tuning data from the client's real Remedy tickets.

Each ticket becomes one chat example:
  input  = the exact prompt the agent sends at run time (llm.build_messages), built from
           the ticket's event type, severity, and summary plus the SOP and precedents,
           with the ticket itself excluded from its own precedents
  target = the JSON the agent must return, taken from what the analyst actually wrote
           (Root Cause Details -> root_cause_summary, Remediation Action -> next_steps)

Run:  uv run python src/sft_data.py          -> data/sft/train.jsonl, data/sft/test.jsonl

Caveat for the report: the tickets are templated (one root cause per event type), so the
held-out split measures format/grounding skill more than generalization. Rate real
Dynatrace problems with evaluate.py's rating sheet for the honest quality number.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from agent import split_sentences
from llm import build_messages
from tools import TriageTools

OUT = Path("data/sft")


def split_steps(text: str) -> list[str]:
    return split_sentences(text)


def label_mask(prompt_ids: list[int], full_ids: list[int]) -> list[int]:
    """Loss only on the answer tokens: prompt positions get -100 (ignored by the loss)."""
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("full sequence must start with the prompt")
    return [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]


def ticket_context(tools: TriageTools, t: dict) -> dict:
    title = t["summary"].split("] ", 1)[-1]
    entity = title.rsplit(" on ", 1)[-1] if " on " in title else "unknown entity"
    cause = {"timestamp": str(t["timestamp"]), "event_type": t["event_type"],
             "severity": t["severity"], "entity_name": entity, "root_cause": title}
    sop = tools.find_sop(t["event_type"])
    remedies = tools.rank_remedies(sop, title) if sop else []
    precedents = tools.find_similar_tickets(f"{t['event_type']} {title}", t["event_type"],
                                            exclude={t["incident_id"]})
    return {
        "problem_id": t["problem_id"], "event_type": t["event_type"], "severity": t["severity"],
        "severity_label": t["severity_label"], "probable_cause": cause, "symptoms": [],
        "telemetry_gaps": [], "recent_changes": [], "precedents": precedents,
        "sop": None if sop is None else {"sop_id": sop.sop_id, "title": sop.title,
                                         "diagnostic_steps": sop.diagnostic_steps,
                                         "remedies": [r.text for r, _ in remedies]},
        "sources": (sop.all_text if sop else []) + [p["remediation"] for p in precedents],
    }


def ticket_example(tools: TriageTools, t: dict) -> dict:
    ctx = ticket_context(tools, t)
    gold = {"root_cause_summary": t["root_cause"], "next_steps": split_steps(t["remediation"])}
    messages = build_messages(ctx) + [{"role": "assistant", "content": json.dumps(gold)}]
    return {"incident_id": t["incident_id"], "event_type": t["event_type"],
            "messages": messages, "gold": gold, "sources": ctx["sources"]}


def build_dataset(tools: TriageTools, holdout_frac: float = 0.2, seed: int = 42):
    rows = tools.tickets.to_dict("records")
    random.Random(seed).shuffle(rows)
    n_test = int(len(rows) * holdout_frac)
    examples = [ticket_example(tools, t) for t in rows]
    return examples[n_test:], examples[:n_test]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    train, test = build_dataset(TriageTools.from_disk())
    write_jsonl(train, OUT / "train.jsonl")
    write_jsonl(test, OUT / "test.jsonl")
    print(f"train: {len(train)}  test: {len(test)}  -> {OUT}/")
