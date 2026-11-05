"""Generate a DEMO deploy log so the change-risk panel has something to show.

These deployments are invented. They are timed around the sample Dynatrace problems so
the console shows every risk level: a likely culprit on the probable-cause entity, a
release on a symptom entity, and old, minor changes that should score low. Real change
records go in data/changes/deployments.csv, which always takes precedence over this file.

  uv run python scripts/make_demo_changes.py            # writes data/changes/deployments_demo.csv
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tools import TriageTools  # noqa: E402

OUT = Path("data/changes/deployments_demo.csv")

# A plausible change that would cause each diagnosis (shares its key words on purpose).
CULPRIT = {
    "queue_depth": "Lower consumer prefetch and queue_depth threshold in config",
    "OneAgent lost connection": "Firewall rule update on the Security Gateway egress",
    "out-of-memory": "Deploy release {v} with a new in-process cache (higher memory use)",
    "ActiveGate heartbeat": "Network route change for the ActiveGate subnet",
    "Thread pool": "Reduce HTTP worker thread pool size in config",
    "Memory usage": "Deploy release {v} enabling the in-memory session cache",
    "Downstream API": "Point the payment client at the new downstream API endpoint (config)",
    "connection pool": "Lower the database connection pool limit in config",
    "OpenPipeline": "OpenPipeline processing rule update (new parsing stage) in config",
    "telemetry": "OS patch for the host telemetry agent",
    "locking": "Database index migration on the settlement table",
    "Disk space": "Enable DEBUG logging config for the /var/log appenders",
    "authentication signature": "Rotate the authentication signature signing certificate",
    "Log pattern": "Deploy release {v} of the exception handling module",
    "OOM killer": "Lower the container memory limit in the Kubernetes config",
    "Garbage Collection": "Upgrade JVM version and Garbage Collection settings, release {v}",
    "Response time": "Deploy release {v} of the checkout service",
    "NullPointerException": "Hotfix deploy {v} to the transaction settlement controller",
    "HTTP 500": "Deploy release {v} of the public API",
    "packet loss": "Network interface driver update and load balancer change",
    "CPU usage": "Deploy release {v} adding the fraud scoring batch job",
    "Health check": "Change health check endpoint timeout in config",
}
RELEASE = ["Deploy release {v} of the web frontend", "Deploy release {v} of the mobile API",
           "Deploy release {v} of the catalog service"]
MINOR = ["Update Grafana dashboard tiles", "Documentation update for the on-call runbook",
         "Adjust alert rule wording in monitoring", "Rotate log level back to INFO"]


def _culprit(root_cause: str) -> str:
    return next((d for k, d in CULPRIT.items() if k.lower() in root_cause.lower()),
                "Deploy release {v} of the affected service")


def build(tools: TriageTools, share: float, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict] = []
    version = lambda: f"v{rng.randint(2, 4)}.{rng.randint(0, 9)}.{rng.randint(0, 20)}"  # noqa: E731
    for p in tools.problems.itertuples():
        if rng.random() > share:
            continue
        corr = tools.correlate(p.problem_id)
        cause = corr["probable_cause"]
        start = pd.Timestamp(p.first_seen)
        scenario = rng.choices(["culprit", "symptom", "minor"], weights=[0.55, 0.25, 0.20])[0]
        if scenario == "culprit":
            rows.append({"entity": cause["entity_name"], "description": _culprit(cause["root_cause"]),
                         "timestamp": start - pd.Timedelta(minutes=rng.randint(8, 75))})
        elif scenario == "symptom" and corr["symptoms"]:
            rows.append({"entity": corr["symptoms"][0]["entity_name"], "description": rng.choice(RELEASE),
                         "timestamp": start - pd.Timedelta(minutes=rng.randint(30, 150))})
        # Background noise: an old, minor change on one of the problem's entities.
        if scenario == "minor" or rng.random() < 0.3:
            rows.append({"entity": rng.choice(str(p.entities).split("; ")), "description": rng.choice(MINOR),
                         "timestamp": start - pd.Timedelta(hours=rng.randint(8, 22))})
    df = pd.DataFrame(rows).drop_duplicates(["entity", "timestamp"]).sort_values("timestamp")
    df["description"] = [d.format(v=version()) for d in df["description"]]
    df["timestamp"] = df["timestamp"].dt.floor("min")
    df.insert(0, "change_id", [f"DEMO-CHG-{i:04d}" for i in range(1, len(df) + 1)])
    return df[["change_id", "timestamp", "entity", "description"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--share", type=float, default=0.4, help="share of problems that get a deploy")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    df = build(TriageTools.from_disk(changes_csv=Path("/nonexistent")), args.share, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} demo deployments -> {args.out}")


if __name__ == "__main__":
    main()
