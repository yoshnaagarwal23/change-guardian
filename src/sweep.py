"""Hyperparameter sweep over a grid of LoRA configs.

  uv run python src/sweep.py --list                      # show the grid
  uv run python src/sweep.py --index 7                   # run one combination (SLURM array task)
  uv run python src/sweep.py --collect                   # rank finished runs -> reports/sweep_results.csv
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path

import yaml

from train_lora import load_config, train

SWEEP_DIR = Path("models/sweep")


def expand(sweep_path: Path) -> list[dict]:
    spec = yaml.safe_load(Path(sweep_path).read_text())
    base = load_config(Path(sweep_path).parent / spec["base"])
    keys, values = zip(*spec["grid"].items())
    runs = []
    for i, combo in enumerate(itertools.product(*values)):
        cfg = copy.deepcopy(base)
        for key, value in zip(keys, combo):
            target, *rest = key.split(".")
            if rest:
                cfg[target][rest[0]] = value
                if rest[0] == "r":
                    cfg[target]["alpha"] = 2 * value  # keep alpha/r = 2
            else:
                cfg[key] = value
        cfg["output_dir"] = str(SWEEP_DIR / f"run-{i:03d}")
        cfg["objective"] = spec.get("objective", "alert_branch_hit_rate")
        runs.append(cfg)
    return runs


def collect(objective: str = "alert_branch_hit_rate") -> Path:
    import pandas as pd

    rows = []
    for metrics in sorted(SWEEP_DIR.glob("run-*/metrics.json")):
        m = json.loads(metrics.read_text())
        cfg = m["config"]
        rows.append({"run": metrics.parent.name, "model": cfg["model_id"], "lr": cfg["learning_rate"],
                     "r": cfg["lora"]["r"], "epochs": cfg["epochs"], **m.get("adapter", {}),
                     "minutes": m.get("train", {}).get("minutes")})
    out = Path("reports/sweep_results.csv")
    out.parent.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(objective, ascending=False)
    df.to_csv(out, index=False)
    print(df.head(10).to_string(index=False) if not df.empty else "no finished runs yet")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, default=Path("configs/sweep.yaml"))
    parser.add_argument("--index", type=int)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()

    runs = expand(args.sweep)
    if args.list:
        for i, cfg in enumerate(runs):
            print(i, cfg["model_id"], cfg["learning_rate"], cfg["lora"]["r"], cfg["epochs"])
    elif args.collect:
        collect(runs[0]["objective"])
    elif args.index is not None:
        train(runs[args.index])
    else:
        parser.error("pass --list, --index N, or --collect")


if __name__ == "__main__":
    main()
