"""Generate a DEMO history of alert dispositions for the alert trust score.

Invented data: ~500 past firings of the sample alert types, each marked by an engineer as
"action_needed" or "no_action_needed". About 80% need no action overall, but the noise rate
differs by alert type so the trust scores spread out. Fixed seed, so the file is reproducible.

  uv run python scripts/make_alert_dispositions.py      # writes data/alert_dispositions.csv
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

OUT = Path("data/alert_dispositions.csv")
ALERTS = Path("data/processed/alerts.csv")
ENGINEERS = ["sre-oncall-a", "sre-oncall-b", "sre-oncall-c", "sre-oncall-d", "sre-oncall-e"]

# Share of past firings that needed NO action, per alert name (keyword match). Unlisted: 0.85.
NOISE = {
    "ActiveGate heartbeat": 0.95, "telemetry reporting": 0.95, "OneAgent lost connection": 0.93,
    "OpenPipeline": 0.95, "Log pattern": 0.92, "Garbage Collection": 0.90, "CPU usage": 0.88,
    "Response time P95": 0.85, "Downstream API": 0.82, "Disk space": 0.80, "queue_depth": 0.80,
    "Memory usage": 0.75, "read locking": 0.70, "Thread pool": 0.65, "packet loss": 0.60,
    "Health check": 0.55, "OOM killer": 0.50, "out-of-memory": 0.45, "connection pool": 0.40,
    "HTTP 500": 0.35, "authentication signature": 0.30, "NullPointerException": 0.30,
}


def noise_rate(alert_name: str) -> float:
    return next((v for k, v in NOISE.items() if k.lower() in alert_name.lower()), 0.85)


def build(alerts: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    names = alerts.groupby("root_cause")["event_type"].agg(lambda s: s.mode()[0])
    # Noisy alerts fire more often, which is what makes them expensive.
    weights = [0.5 + noise_rate(a) for a in names.index]
    start = pd.Timestamp("2026-09-01")
    rows = []
    for i in range(n):
        name = rng.choices(list(names.index), weights=weights)[0]
        rows.append({
            "timestamp": start + pd.Timedelta(minutes=rng.randint(0, 40 * 24 * 60)),
            "alert_name": name,
            "alert_type": names[name],
            "disposition": "no_action_needed" if rng.random() < noise_rate(name) else "action_needed",
            "engineer": rng.choice(ENGINEERS),
        })
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df.insert(1, "alert_id", [f"HIST-{i:05d}" for i in range(1, len(df) + 1)])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    df = build(pd.read_csv(ALERTS), args.n, args.seed)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} dispositions -> {args.out} "
          f"({(df['disposition'] == 'no_action_needed').mean():.0%} no action needed)")


if __name__ == "__main__":
    main()
