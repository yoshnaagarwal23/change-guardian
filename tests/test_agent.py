"""The agent contract from the README, with and without an LLM."""
import json

import pytest

from agent import TriageAgent
from llm import Generation, build_messages, parse_model_output
from schema import TriageCard


class FakeBackend:
    """Stands in for a model so the pipeline is tested without downloads."""
    name = "fake"

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        return Generation(text=self.reply, tokens_in=100, tokens_out=40)


def _first_problem(tools, event_type):
    return tools.problems[tools.problems["event_type"] == event_type].iloc[0]["problem_id"]


def test_rules_only_card_meets_the_readme_contract(tools):
    pid = _first_problem(tools, "Availability")
    card = TriageAgent(tools).triage(pid)
    TriageCard.model_validate(card)  # raises if the contract is broken
    assert card["priority"] in {"Critical", "High"}
    assert card["assigned_group"] == "ITOM-Availability-SRE"
    assert card["sop_id"] == "SOP-001"
    assert card["escalate"] is False
    assert card["backend"] == "rules"
    assert 1 <= len(card["next_steps"]) <= 8


def test_no_sop_means_escalate_with_precedent_based_steps(tools):
    pid = _first_problem(tools, "Monitoring Unavailable")
    card = TriageAgent(tools).triage(pid)
    assert card["sop_id"] is None and card["escalate"] is True
    assert card["similar_tickets"], "precedents should still be offered"
    assert any("SOP" in w for w in card["warnings"])


def test_security_signals_notify_secops(tools):
    hit = tools.alerts[tools.alerts["root_cause"].str.contains("authentication")].iloc[0]
    card = TriageAgent(tools).triage(hit["problem_id"])
    assert card["notify_secops"] is True


def test_grounded_llm_output_is_used(tools):
    pid = _first_problem(tools, "Resource")
    diagnosis = tools.correlate(pid)["probable_cause"]["root_cause"]
    reply = json.dumps({
        "root_cause_summary": f"Memory pressure: {diagnosis}",
        "next_steps": ["Analyze the Process Performance breakdown to pinpoint the runaway thread.",
                       "Increase CPU/Memory allocations or scale deployment replicas."]})
    backend = FakeBackend(reply)
    card = TriageAgent(tools, backend).triage(pid)
    assert backend.calls == 1
    assert card["backend"] == "fake"
    assert card["root_cause_summary"].startswith("Memory pressure")
    assert card["tokens_in"] == 100


def test_ungrounded_steps_are_dropped_and_garbage_falls_back_to_rules(tools):
    pid = _first_problem(tools, "Error")
    invented = json.dumps({"root_cause_summary": "Aliens.",
                           "next_steps": ["Reformat every production disk immediately."]})
    card = TriageAgent(tools, FakeBackend(invented)).triage(pid)
    assert "Reformat every production disk immediately." not in card["next_steps"]
    assert any("fallback" in w for w in card["warnings"])

    card = TriageAgent(tools, FakeBackend("not json at all")).triage(pid)
    assert card["next_steps"] and any("fallback" in w for w in card["warnings"])


def test_recalled_template_for_a_different_alert_is_rejected(tools):
    # A disk alert answered with the (grounded!) GC/CPU ticket template: the failure
    # the first LoRA adapter showed. Every word is in a source, so only consistency catches it.
    disk = tools.alerts[tools.alerts["root_cause"].str.startswith("Disk space")].iloc[0]
    corr = tools.correlate(disk["problem_id"])
    if corr["probable_cause"]["root_cause"] != disk["root_cause"]:
        pytest.skip("disk alert is a symptom in this problem")
    recalled = json.dumps({
        "root_cause_summary": "Extensive garbage collection (GC) cycles taking up 95% of active CPU "
                              "cycles due to small JVM heap allocation settings.",
        "next_steps": ["Review GC metrics in Dynatrace.",
                       "Adjust container limits via deployment spec."]})
    card = TriageAgent(tools, FakeBackend(recalled)).triage(disk["problem_id"])
    assert "garbage collection" not in card["root_cause_summary"].lower()
    assert any("does not match the alert" in w for w in card["warnings"])


def test_prompt_warns_that_precedents_may_be_a_different_cause(tools):
    ctx = TriageAgent(tools).build_context(_first_problem(tools, "Resource"))
    assert "may describe a different cause" in build_messages(ctx)[-1]["content"]


def test_card_labels_precedents_and_only_cites_matching_ones(tools):
    hc = tools.alerts[tools.alerts["root_cause"].str.startswith("Health check")]
    pid = next(p for p in hc["problem_id"]
               if tools.correlate(p)["probable_cause"]["root_cause"].startswith("Health check"))
    card = TriageAgent(tools).triage(pid)
    assert card["precedents"] and all("matches_alert" in p for p in card["precedents"])
    assert not any(p["matches_alert"] for p in card["precedents"])
    assert "precedent" not in card["root_cause_summary"].lower()  # don't cite an OOM ticket


def test_parse_model_output_extracts_json_from_chatty_text():
    text = 'Sure! Here is the card:\n```json\n{"root_cause_summary": "x", "next_steps": ["a"]}\n```'
    assert parse_model_output(text) == {"root_cause_summary": "x", "next_steps": ["a"]}
    with pytest.raises(ValueError):
        parse_model_output("{broken")


def test_prompt_contains_sop_and_precedent_but_not_answers_from_nowhere(tools):
    pid = _first_problem(tools, "Slowdown")
    ctx = TriageAgent(tools).build_context(pid)
    messages = build_messages(ctx)
    user = messages[-1]["content"]
    assert messages[0]["role"] == "system"
    assert "SOP-002" in user and "Precedent" in user
    assert ctx["probable_cause"]["root_cause"] in user


def test_card_is_compatible_with_yoshnas_evaluator(tools, tmp_path, monkeypatch):
    import evaluate
    cards = [TriageAgent(tools).triage(pid) for pid in tools.problems["problem_id"][:20]]
    results = tmp_path / "agent_results.jsonl"
    results.write_text("\n".join(json.dumps(c) for c in cards) + "\n")
    monkeypatch.setattr(evaluate, "RESULTS", results)
    scores = evaluate.score(tools.problems, evaluate.load_results())
    assert scores[["priority_ok", "team_ok", "sop_ok", "escalate_ok"]].all().all()
