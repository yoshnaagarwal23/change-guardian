"""Score the base model and any number of adapters on the same held-out set.

  uv run python src/compare_models.py models/lora-mac-v1 models/lora-mac
  -> reports/model_comparison.json (read by the Streamlit KPI tab) + a Markdown table
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_lora import evaluate_adapter, load_config

OUT = Path("reports/model_comparison.json")
COLUMNS = ["alert_json_valid_rate", "alert_consistency", "alert_branch_hit_rate", "alert_grounding",
           "ticket_json_valid_rate", "ticket_rouge_l"]


def to_markdown(rows: dict[str, dict]) -> str:
    header = "| Model | " + " | ".join(COLUMNS) + " |\n|---|" + "---|" * len(COLUMNS) + "\n"
    body = "".join(f"| {name} | " + " | ".join(f"{m.get(c, float('nan')):.2f}" for c in COLUMNS) + " |\n"
                   for name, m in rows.items())
    return header + body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapters", nargs="*", help="adapter directories to compare against the base model")
    parser.add_argument("--config", type=Path, default=Path("configs/mac.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = {f"base ({cfg['model_id'].split('/')[-1]})": evaluate_adapter(cfg, None)}
    for adapter in args.adapters:
        rows[Path(adapter).name] = evaluate_adapter(cfg, adapter)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))
    table = to_markdown(rows)
    OUT.with_suffix(".md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
