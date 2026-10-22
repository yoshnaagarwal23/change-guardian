"""Model-quality metrics for fine-tuning runs (no extra dependencies)."""
from __future__ import annotations

import re

from guardrails import grounding_score
from llm import parse_model_output


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


def score_generation(text: str, gold: dict, sources: list[str]) -> dict:
    try:
        pred = parse_model_output(text)
    except ValueError:
        return {"json_valid": False, "rouge_l": 0.0, "grounding": 0.0, "n_steps": 0}
    steps = pred["next_steps"]
    grounding = sum(grounding_score(s, sources) for s in steps) / len(steps)
    return {"json_valid": True, "rouge_l": rouge_l(_flatten(pred), _flatten(gold)),
            "grounding": grounding, "n_steps": len(steps)}


def summarize(scores: list[dict]) -> dict:
    n = len(scores) or 1
    return {
        "n": len(scores),
        "json_valid_rate": sum(s["json_valid"] for s in scores) / n,
        "rouge_l": sum(s["rouge_l"] for s in scores) / n,
        "grounding": sum(s["grounding"] for s in scores) / n,
    }
