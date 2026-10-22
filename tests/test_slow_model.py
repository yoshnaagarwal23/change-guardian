"""End-to-end checks with a real Hugging Face model. Downloads ~1 GB on first run.

  uv run pytest -m slow
"""
import pytest

pytestmark = pytest.mark.slow


def test_local_model_produces_a_valid_card(tools):
    from agent import TriageAgent
    from llm import HFBackend

    pid = tools.problems["problem_id"].iloc[0]
    card = TriageAgent(tools, HFBackend()).triage(pid)
    assert card["next_steps"] and card["tokens_out"] > 0


def test_two_step_training_run_saves_an_adapter(tmp_path):
    from train_lora import load_config, train

    cfg = load_config("configs/mac.yaml")
    cfg.update(output_dir=str(tmp_path / "adapter"), eval_samples=1, mlflow=False)
    result = train(cfg, max_steps=2)
    assert (tmp_path / "adapter" / "adapter_config.json").exists()
    assert result["train"]["steps"] == 2
