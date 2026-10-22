"""Model backends behind one interface, plus the shared prompt format.

The same `build_messages` is used for inference and for building fine-tuning data,
so a LoRA adapter is always trained on exactly the prompt it will see in production.

Backends:
  hf         Hugging Face Transformers, local (default model: Qwen2.5-0.5B-Instruct),
             optionally with a LoRA adapter from train_lora.py
  anthropic  Claude via the Anthropic API (comparison runs / teacher for distillation)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from schema import ModelAnswer

DEFAULT_HF_MODEL = os.getenv("CG_HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DEFAULT_ANTHROPIC_MODEL = os.getenv("CG_ANTHROPIC_MODEL", "claude-opus-5-5")

# USD per 1M tokens (input, output), for the cost KPI. Local models cost 0 per token.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

SYSTEM_PROMPT = (
    "You are Change Guardian, an incident triage assistant for a payment network's "
    "site reliability team. Use ONLY the SOP and precedent tickets in the message. "
    "Reply with JSON only, exactly: "
    '{"root_cause_summary": "<1-2 sentences>", "next_steps": ["<step>", "..."]}. '
    "The summary must name the probable-cause alert. Give 2 to 5 next steps, each one imperative sentence taken or adapted from the SOP "
    "or a precedent. Never invent commands, numbers, hosts, or systems. Never say an "
    "action was already taken: a human engineer approves every change."
)


@dataclass
class Generation:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0


def _alert_line(a: dict) -> str:
    return f"[{a['timestamp']}] {a['event_type']} {a['severity']} on {a['entity_name']}: {a['root_cause']}"


def build_messages(ctx: dict) -> list[dict]:
    """Render a triage context into chat messages (system + user)."""
    lines = [
        f"Problem {ctx['problem_id']} | Event type: {ctx['event_type']} | "
        f"Severity: {ctx.get('severity_label', ctx['severity'])}",
        f"Probable cause: {_alert_line(ctx['probable_cause'])}",
    ]
    for a in ctx.get("symptoms", []):
        lines.append(f"Symptom: {_alert_line(a)}")
    for a in ctx.get("telemetry_gaps", []):
        lines.append(f"Telemetry gap: {_alert_line(a)}")
    changes = ctx.get("recent_changes") or []
    lines.append("Recent changes: " + ("; ".join(
        f"{c.get('change_id', '?')} {c.get('description', '')}" for c in changes) or "none on record"))

    sop = ctx.get("sop")
    if sop:
        lines.append(f"\n{sop['sop_id']}: {sop['title']}")
        lines.append("Diagnostic steps:")
        lines += [f"{i}. {s}" for i, s in enumerate(sop["diagnostic_steps"], 1)]
        lines.append("Best-matching remediation:")
        lines += [f"- {r}" for r in sop["remedies"]]
    else:
        lines.append("\nNo SOP covers this event type. Use precedent only and say a SOP is missing.")

    lines.append("\nPrecedent tickets (same event type; they may describe a different cause, "
                 "so use one only if it matches the probable cause):")
    for t in ctx.get("precedents", []):
        lines.append(f"- {t['incident_id']} (seen {t['precedent_count']}x): {t['root_cause']} "
                     f"Remediation: {t['remediation']}")
    if not ctx.get("precedents"):
        lines.append("- none")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)}]


def parse_model_output(text: str) -> dict:
    """Pull the first JSON object out of a model reply and validate its shape."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("no JSON object in model output")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    return ModelAnswer.model_validate(data).model_dump()


class HFBackend:
    """Local Hugging Face model. Greedy decoding so identical inputs give identical output."""

    def __init__(self, model_id: str = DEFAULT_HF_MODEL, adapter_path: str | None = None,
                 max_new_tokens: int = 320, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if self.device in {"cuda", "mps"} else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self.name = f"hf:{model_id.split('/')[-1]}" + ("+lora" if adapter_path else "")

    def generate(self, messages: list[dict]) -> Generation:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        new = out[0][inputs["input_ids"].shape[1]:]
        return Generation(self.tokenizer.decode(new, skip_special_tokens=True),
                          int(inputs["input_ids"].shape[1]), int(len(new)))


class AnthropicBackend:
    """Claude via the official SDK. Thinking is always on for Opus 5.5; effort keeps it cheap."""

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL, effort: str = "low"):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.name = f"anthropic:{model}"

    def generate(self, messages: list[dict]) -> Generation:
        system = messages[0]["content"]
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=4000,
            system=system,
            messages=messages[1:],
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return Generation("", response.usage.input_tokens, response.usage.output_tokens)
        text = "".join(b.text for b in response.content if b.type == "text")
        return Generation(text, response.usage.input_tokens, response.usage.output_tokens)


def get_backend(name: str, **kwargs):
    """'rules' -> None (no model). 'hf' / 'anthropic' -> a backend instance."""
    if name in {"rules", "none", ""}:
        return None
    if name == "hf":
        return HFBackend(**kwargs)
    if name == "anthropic":
        return AnthropicBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r}")


def token_cost_usd(backend_name: str, tokens_in: int, tokens_out: int) -> float:
    model = backend_name.split(":", 1)[-1]
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000
