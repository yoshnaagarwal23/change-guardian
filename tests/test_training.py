"""Fine-tuning data, metrics, and config plumbing (no model download needed)."""
import json
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


def test_ticket_examples_never_leak_the_same_ticket(tools):
    train, test = build_dataset(tools, holdout_frac=0.2, seed=7)
    tickets = [e for e in train + test if e["kind"] == "ticket"]
    assert len(tickets) == len(tools.tickets)
    train_ids = {e["id"] for e in train if e["kind"] == "ticket"}
    assert train_ids.isdisjoint({e["id"] for e in test if e["kind"] == "ticket"})
    ex = next(e for e in train if e["kind"] == "ticket")
    assert ex["messages"][-1]["role"] == "assistant"
    assert ex["id"] not in " ".join(m["content"] for m in ex["messages"][:-1])


def test_alert_examples_hold_out_whole_alert_types(tools):
    train, test = build_dataset(tools, holdout_frac=0.2, seed=7)
    alerts_train = [e for e in train if e["kind"] == "alert"]
    alerts_test = [e for e in test if e["kind"] == "alert"]
    assert alerts_train and alerts_test
    assert len(alerts_train) + len(alerts_test) == len(tools.problems)
    # Generalization test: no held-out diagnosis is ever seen in training.
    assert {e["diagnosis"] for e in alerts_train}.isdisjoint({e["diagnosis"] for e in alerts_test})
    ex = alerts_train[0]
    gold = json.loads(ex["messages"][-1]["content"])
    assert ex["diagnosis"].rstrip(".") in gold["root_cause_summary"]


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


def test_score_generation_checks_the_alert_and_the_remedy_branch():
    gold = {"root_cause_summary": "Disk space low on /var/log.", "next_steps": ["Archive old logs."]}
    recalled = ('{"root_cause_summary": "GC cycles took 95% of CPU.", '
                '"next_steps": ["Review GC metrics in Dynatrace."]}')
    right = ('{"root_cause_summary": "Disk space below 5% on /var/log.", '
             '"next_steps": ["Truncate or archive non-essential logs."]}')
    kw = dict(sources=["Truncate or archive non-essential logs.", "Review GC metrics in Dynatrace."],
              diagnosis="Disk space capacity remaining less than 5% on /var/log.",
              key_step="Truncate or archive non-essential logs, clear temporary files.")
    bad, good = score_generation(recalled, gold, **kw), score_generation(right, gold, **kw)
    assert good["consistency"] > bad["consistency"]
    assert good["branch_hit"] and not bad["branch_hit"]


def test_configs_load_with_required_keys():
    for name in ["mac.yaml", "cluster.yaml"]:
        cfg = load_config(ROOT / "configs" / name)
        for key in ["model_id", "lora", "learning_rate", "epochs", "output_dir"]:
            assert key in cfg
