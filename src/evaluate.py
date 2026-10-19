"""Score the agent's ticket drafts.

Run from the repo root:
  python src/evaluate.py --baseline   # fake "rules-only" agent output, to test this script now
  python src/evaluate.py              # score the real agent output (outputs/agent_results.jsonl)

Input:  outputs/agent_results.jsonl  - one JSON object per line, in the README format
Output: outputs/scores.csv           - pass/fail per problem for priority, team, SOP
        outputs/rating_sheet.csv     - for a human to rate summary + next steps 1-5
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from load_data import EVENT_TO_SOP, SEVERITY_TO_PRIORITY

PROBLEMS = Path("data/processed/problems.csv")
RESULTS = Path("outputs/agent_results.jsonl")
OUT = Path("outputs")


def expected(row: pd.Series) -> dict:
    """The right answer for a problem. Rules confirmed from 300 real Remedy tickets."""
    return {
        "priority": SEVERITY_TO_PRIORITY[row["severity"]],
        "assigned_group": f"ITOM-{row['event_type']}-SRE",
        "sop_id": EVENT_TO_SOP.get(row["event_type"]),
    }


def write_baseline(problems: pd.DataFrame) -> None:
    """Fake agent output using only the rules (no LLM). Lets us test scoring today."""
    OUT.mkdir(exist_ok=True)
    with RESULTS.open("w") as f:
        for _, row in problems.iterrows():
            exp = expected(row)
            f.write(json.dumps({
                "problem_id": row["problem_id"],
                "event_type": row["event_type"],
                "severity": row["severity"],
                **exp,
                "root_cause_summary": row["root_causes"],
                "next_steps": [],
                "similar_tickets": [],
                "escalate": exp["sop_id"] is None,
            }) + "\n")
    print(f"wrote baseline output for {len(problems)} problems -> {RESULTS}")


def load_results() -> pd.DataFrame:
    rows = [json.loads(line) for line in RESULTS.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def score(problems: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    merged = problems.merge(results, on="problem_id", suffixes=("", "_agent"))
    rows = []
    for _, r in merged.iterrows():
        exp = expected(r)
        rows.append({
            "problem_id": r["problem_id"],
            "event_type": r["event_type"],
            "priority_ok": r["priority"] == exp["priority"],
            "team_ok": r["assigned_group"] == exp["assigned_group"],
            # pandas turns a missing sop_id into NaN, so treat NaN and None as the same
            "sop_ok": (None if pd.isna(r["sop_id"]) else r["sop_id"]) == exp["sop_id"],
            # should escalate exactly when there is no SOP
            "escalate_ok": bool(r["escalate"]) == (exp["sop_id"] is None),
        })
    return pd.DataFrame(rows)


def rating_sheet(problems: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Sheet for a human (Gabrielle) to rate the LLM's actual work, 1-5."""
    merged = problems.merge(results, on="problem_id", suffixes=("", "_agent"))
    sheet = merged[["problem_id", "event_type", "severity", "root_causes",
                    "sop_id", "root_cause_summary", "next_steps"]].copy()
    sheet["next_steps"] = sheet["next_steps"].apply(lambda s: " | ".join(s) if isinstance(s, list) else s)
    sheet["summary_rating_1to5"] = ""
    sheet["next_steps_rating_1to5"] = ""
    sheet["notes"] = ""
    return sheet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="generate rules-only output first")
    args = parser.parse_args()

    problems = pd.read_csv(PROBLEMS)
    if args.baseline:
        write_baseline(problems)
    if not RESULTS.exists():
        raise SystemExit(f"No {RESULTS} yet. Run with --baseline, or run the agent first.")

    results = load_results()
    scores = score(problems, results)
    OUT.mkdir(exist_ok=True)
    scores.to_csv(OUT / "scores.csv", index=False)
    rating_sheet(problems, results).to_csv(OUT / "rating_sheet.csv", index=False)

    print(f"\nScored {len(scores)} of {len(problems)} problems")
    for col, label in [("priority_ok", "Priority match"), ("team_ok", "Team match"),
                       ("sop_ok", "Correct SOP"), ("escalate_ok", "Escalated correctly")]:
        print(f"  {label:<20} {scores[col].mean():.0%}")
    print("\nBy event type (correct SOP):")
    print(scores.groupby("event_type")["sop_ok"].mean().map("{:.0%}".format).to_string())
    print(f"\nWrote {OUT/'scores.csv'} and {OUT/'rating_sheet.csv'}")