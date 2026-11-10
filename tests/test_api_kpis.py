"""KPIs for the business case and the HTTP API a ticketing tool would call."""
from fastapi.testclient import TestClient

from agent import TriageAgent
from kpis import compute_kpis


def test_kpis_from_a_batch_of_cards(tools):
    cards = [TriageAgent(tools).triage(p) for p in tools.problems["problem_id"][:30]]
    k = compute_kpis(cards, tools.alerts)
    assert k["problems_triaged"] == 30
    assert 0 <= k["sop_coverage_pct"] <= 100
    assert k["escalation_rate_pct"] == round(100 - k["sop_coverage_pct"], 1)
    assert k["page_reduction_pct"] == 50.0
    assert k["est_llm_cost_usd"] == 0.0  # rules-only spends no tokens


def test_api_triage_and_decision(tools, tmp_path, monkeypatch):
    import api
    monkeypatch.setattr(api, "AUDIT_PATH", tmp_path / "audit.jsonl")
    client = TestClient(api.create_app(tools))
    pid = tools.problems["problem_id"].iloc[0]

    assert client.get("/problems").json()[0]["problem_id"]
    card = client.post(f"/triage/{pid}").json()
    assert card["problem_id"] == pid

    r = client.post("/decision", json={"problem_id": pid, "decision": "approved", "user": "test"})
    assert r.status_code == 200 and r.json()["audit_chain_ok"] is True
    assert client.post("/triage/P-NOPE").status_code == 404


def test_card_cost_label_uses_the_model_price():
    from llm import cost_label
    assert cost_label("rules", 0, 0) == "$0.00"
    assert cost_label("hf:Qwen2.5-0.5B-Instruct+lora", 900, 300) == "$0.00"
    assert cost_label("anthropic:claude-sonnet-5", 861, 265) == "$0.0044"   # 861*$2/M + 265*$10/M
    assert cost_label("anthropic:claude-opus-5-5", 2000, 1000) == "$0.028"
    assert cost_label("anthropic:some-new-model", 100, 100) == "price unknown"


def test_sonnet_5_prices_are_named_constants_and_small_costs_stay_visible():
    from llm import PRICES, SONNET_5_INPUT_USD_PER_MTOK, SONNET_5_MODEL, SONNET_5_OUTPUT_USD_PER_MTOK, usd
    assert PRICES[SONNET_5_MODEL] == (SONNET_5_INPUT_USD_PER_MTOK, SONNET_5_OUTPUT_USD_PER_MTOK) == (2.00, 10.00)
    assert (usd(0), usd(0.0044), usd(0.05), usd(1.234)) == ("$0.00", "$0.0044", "$0.050", "$1.23")
