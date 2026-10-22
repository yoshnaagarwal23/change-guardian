"""Fine-tuning data, metrics, and config plumbing (no model download needed)."""
from pathlib import Path

import pytest

from metrics import rouge_l, score_generation
from sft_data import build_dataset, label_mask, split_steps
from train_lora import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_split_steps_turns_remediation_prose_into_steps():
    text = ("Verify server status in cloud console. Pull localized diagnostic heap dumps, "
            "restart the application process with updated memory limits.")
    assert split_steps(text) == [
        "Verify server status in cloud console.",
        "Pull localized diagnostic heap dumps, restart the application process with updated memory limits."]


def test_dataset_targets_come_from_real_tickets_and_never_leak_the_same_ticket(tools):
    train, test = build_dataset(tools, holdout_frac=0.2, seed=7)
    assert len(train) + len(test) == len(tools.tickets)
    assert {e["incident_id"] for e in train}.isdisjoint({e["incident_id"] for e in test})
    ex = train[0]
    assert ex["messages"][-1]["role"] == "assistant"
    prompt = " ".join(m["content"] for m in ex["messages"][:-1])
    assert ex["incident_id"] not in prompt


def test_label_mask_only_trains_on_the_answer():
    prompt_ids, full_ids = [1, 2, 3], [1, 2, 3, 9, 9]
    assert label_mask(prompt_ids, full_ids) == [-100, -100, -100, 9, 9]


def test_rouge_l():
    assert rouge_l("restart the process", "restart the process") == pytest.approx(1.0)
    assert rouge_l("restart the process", "check the logs") < 0.5
    assert rouge_l("", "anything") == 0.0


def test_score_generation_reports_validity_and_quality():
    gold = {"root_cause_summary": "OOM killer ended the gateway.", "next_steps": ["Restart the process."]}
    good = score_generation('{"root_cause_summary": "OOM killer ended the gateway.", '
                            '"next_steps": ["Restart the process."]}', gold, sources=["Restart the process."])
    bad = score_generation("nope", gold, sources=[])
    assert good["json_valid"] and good["rouge_l"] > 0.9 and good["grounding"] > 0.9
    assert not bad["json_valid"] and bad["rouge_l"] == 0.0


def test_configs_load_with_required_keys():
    for name in ["mac.yaml", "cluster.yaml"]:
        cfg = load_config(ROOT / "configs" / name)
        for key in ["model_id", "lora", "learning_rate", "epochs", "output_dir"]:
            assert key in cfg
