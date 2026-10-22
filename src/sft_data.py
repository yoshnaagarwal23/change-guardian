"""Build supervised fine-tuning data for the local model.

Two kinds of examples, both using the exact prompt the agent sends at run time
(llm.build_messages), so the adapter is trained on what it will see in production:

  ticket  One per Remedy ticket. Target = what the analyst actually wrote
          (Root Cause Details -> summary, Remediation Action -> steps). The ticket is
          excluded from its own precedents.
  alert   One per Dynatrace problem. Target = the rules engine's alert-specific draft
          (names the firing alert, the most relevant SOP steps, the matching remedy branch).

Why both: v1 trained on tickets only. Tickets are templated (one root cause per event
type), so the model learned to recite the precedent ticket whatever the alert said
(a disk alert got the GC/CPU story). Alert examples teach it to answer about the alert.

Split: tickets 80/20 at random; alerts by *diagnosis*, so whole alert types the model
never saw in training are held out. That measures generalization, not memorization.

Run:  uv run python src/sft_data.py          -> data/sft/train.jsonl, data/sft/test.jsonl
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from agent import TriageAgent, split_sentences
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


def _example(kind: str, ex_id: str, ctx: dict, gold: dict, diagnosis: str,
             key_step: str | None, sources: list[str]) -> dict:
    messages = build_messages(ctx) + [{"role": "assistant", "content": json.dumps(gold)}]
    return {"kind": kind, "id": ex_id, "event_type": ctx["event_type"], "diagnosis": diagnosis,
            "key_step": key_step, "messages": messages, "gold": gold, "sources": sources}


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
    return _example("ticket", t["incident_id"], ctx, gold, ctx["probable_cause"]["root_cause"],
                    None, ctx["sources"])


def alert_example(agent: TriageAgent, problem_id: str) -> dict:
    ctx = agent.build_context(problem_id)
    gold = agent.rules_draft(ctx)
    key_step = ctx["sop"]["remedies"][0].split(": ", 1)[-1] if ctx["sop"] else None
    sources = (ctx["sop_all_text"] + ctx["alert_texts"]
               + [p["remediation"] + " " + p["root_cause"] for p in ctx["precedents"]])
    return _example("alert", problem_id, ctx, gold, ctx["probable_cause"]["root_cause"], key_step, sources)


def build_dataset(tools: TriageTools, holdout_frac: float = 0.2, seed: int = 42):
    rng = random.Random(seed)
    tickets = tools.tickets.to_dict("records")
    rng.shuffle(tickets)
    n_test = int(len(tickets) * holdout_frac)
    ticket_ex = [ticket_example(tools, t) for t in tickets]

    agent = TriageAgent(tools)
    alert_ex = [alert_example(agent, pid) for pid in tools.problems["problem_id"]]
    diagnoses = sorted({e["diagnosis"] for e in alert_ex})
    rng.shuffle(diagnoses)
    held_out = set(diagnoses[: max(1, round(len(diagnoses) * holdout_frac))])

    train = ticket_ex[n_test:] + [e for e in alert_ex if e["diagnosis"] not in held_out]
    test = ticket_ex[:n_test] + [e for e in alert_ex if e["diagnosis"] in held_out]
    rng.shuffle(train)
    return train, test


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    train, test = build_dataset(TriageTools.from_disk())
    write_jsonl(train, OUT / "train.jsonl")
    write_jsonl(test, OUT / "test.jsonl")
    count = lambda rows, k: sum(r["kind"] == k for r in rows)  # noqa: E731
    print(f"train: {count(train, 'ticket')} ticket + {count(train, 'alert')} alert examples")
    print(f"test:  {count(test, 'ticket')} ticket + {count(test, 'alert')} alert examples")
    print("held-out alert types:", sorted({r['diagnosis'] for r in test if r['kind'] == 'alert'}))
