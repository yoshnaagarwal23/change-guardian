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
