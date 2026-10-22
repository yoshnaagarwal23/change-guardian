"""Model-quality metrics for fine-tuning runs (no extra dependencies).

  json_valid   the reply parses into the required JSON shape
  rouge_l      word overlap with the reference answer (ROUGE-L F1)
  grounding    share of step words found in the SOP / precedents / alerts
  consistency  share of the firing alert's words the summary mentions
               (catches reciting a real-but-unrelated precedent, which grounding can't)
  branch_hit   a step reproduces the SOP remedy branch that matches the diagnosis
"""
from __future__ import annotations

import re

from guardrails import alert_consistency, grounding_score
from llm import parse_model_output

BRANCH_MIN = 0.5


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F1 from the longest common subsequence of word tokens."""
    a, b = _tokens(pred), _tokens(ref)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(a), lcs / len(b)
    return 2 * precision * recall / (precision + recall)


def _flatten(answer: dict) -> str:
    return answer["root_cause_summary"] + " " + " ".join(answer["next_steps"])


def score_generation(text: str, gold: dict, sources: list[str], diagnosis: str | None = None,
                     key_step: str | None = None) -> dict:
    try:
        pred = parse_model_output(text)
    except ValueError:
        return {"json_valid": False, "rouge_l": 0.0, "grounding": 0.0, "n_steps": 0,
                "consistency": 0.0 if diagnosis else None, "branch_hit": False if key_step else None}
    steps = pred["next_steps"]
    branch = None
    if key_step:
        branch = any(max(grounding_score(key_step, [s]), grounding_score(s, [key_step])) >= BRANCH_MIN
                     for s in steps)
    return {"json_valid": True, "rouge_l": rouge_l(_flatten(pred), _flatten(gold)),
            "grounding": sum(grounding_score(s, sources) for s in steps) / len(steps),
            "n_steps": len(steps),
            "consistency": alert_consistency(pred["root_cause_summary"], diagnosis) if diagnosis else None,
            "branch_hit": branch}


def summarize(scores: list[dict]) -> dict:
    def mean(key: str) -> float | None:
        vals = [float(s[key]) for s in scores if s.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {"n": len(scores), "json_valid_rate": mean("json_valid"), "rouge_l": mean("rouge_l"),
            "grounding": mean("grounding"), "consistency": mean("consistency"),
            "branch_hit_rate": mean("branch_hit")}
