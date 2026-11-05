"""Generate DEMO inputs for the pre-deploy risk score (all invented).

  data/changes/pending_demo.csv     ~15 changes waiting to ship: safe, borderline and risky
  data/changes/service_tiers.csv    tier per entity (1 = critical) + downstream dependents
  data/changes/freeze_windows.csv   one code freeze (Thu 2026-10-15 00:00 -> Mon 2026-10-19 08:00)

  uv run python scripts/make_pending_changes.py
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

OUT_DIR = Path("data/changes")
OWNERS = ["dev-payments", "dev-checkout", "dev-platform", "dev-mobile", "dev-catalog", "dev-identity"]

TIERS = [  # entity, tier, downstream_dependents
    ("B2B Checkout App", 1, 14), ("PaymentService", 1, 18), ("host-payment-prod-01", 1, 12),
    ("AuthenticationService", 1, 16), ("host-auth-prod-02", 1, 10), ("Mobile API Gateway", 1, 11),
    ("Production Web Portal", 2, 7), ("OrderProcessingWorker", 2, 6), ("FrontendGateway", 2, 8),
    ("host-catalog-db-01", 2, 9), ("host-frontend-05", 2, 5),
    ("ProductCatalogService", 3, 3), ("host-worker-03", 3, 1),
]
FREEZE = [{"name": "Q4 payments peak freeze", "start": "2026-10-15 00:00", "end": "2026-10-19 08:00"}]

# (profile, entity, change_type, description, lines range, rollback, canary, planned-time choices)
TUESDAY = ["2026-10-20 10:00", "2026-10-20 11:30", "2026-10-21 14:00", "2026-10-22 10:30"]
FREEZE_DAYTIME = ["2026-10-15 10:00", "2026-10-15 14:30"]
# Risky changes outside the freeze, so HIGH is not just "it is a freeze weekend".
WEEKDAY_OUTSIDE_FREEZE = ["2026-10-20 10:00", "2026-10-21 15:00"]
FRI_EVENING = ["2026-10-16 18:30", "2026-10-16 21:00", "2026-10-17 02:00"]
PLAN = [
    ("safe", "ProductCatalogService", "observability", "Add latency panel to the catalog dashboard", (5, 40), True, True, TUESDAY),
    ("safe", "host-worker-03", "configuration", "Raise batch worker log retention to 14 days", (2, 15), True, True, TUESDAY),
    ("safe", "Production Web Portal", "observability", "Tune alert rule wording for portal 5xx", (3, 20), True, True, TUESDAY),
    ("safe", "ProductCatalogService", "code release", "Release catalog search ranking tweak", (20, 80), True, True, TUESDAY),
    ("safe", "host-frontend-05", "configuration", "Enable gzip for static assets", (4, 12), True, True, TUESDAY),
    ("borderline", "OrderProcessingWorker", "code release", "Release order retry backoff changes", (150, 400), True, False, TUESDAY),
    ("borderline", "Production Web Portal", "code release", "Release portal session handling refactor", (300, 700), False, True, TUESDAY),
    ("borderline", "Mobile API Gateway", "configuration", "Adjust gateway rate limit for partner API", (5, 30), True, True, TUESDAY),
    ("borderline", "host-catalog-db-01", "infrastructure", "Add index on catalog_items.updated_at", (10, 40), True, True, TUESDAY),
    ("borderline", "FrontendGateway", "code release", "Upgrade TLS library in the frontend gateway", (80, 250), True, False, FREEZE_DAYTIME),
    ("risky", "PaymentService", "infrastructure", "Database schema migration for settlement ledger", (600, 1500), False, False, WEEKDAY_OUTSIDE_FREEZE),
    ("risky", "B2B Checkout App", "code release", "Release new checkout pricing engine", (1200, 3000), True, False, FRI_EVENING),
    ("risky", "AuthenticationService", "infrastructure", "Rotate signing certificate and change token TTL", (40, 120), False, False, WEEKDAY_OUTSIDE_FREEZE),
    ("risky", "host-payment-prod-01", "configuration", "Lower database connection pool limit", (2, 10), False, False, FRI_EVENING),
    ("risky", "B2B Checkout App", "infrastructure", "Move checkout to new load balancer cluster", (300, 900), False, True, FREEZE_DAYTIME),
]


def build(seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i, (_, entity, ctype, desc, (lo, hi), rollback, canary, times) in enumerate(PLAN, 1):
        rows.append({"change_id": f"DEMO-PEND-{i:04d}", "planned_time": rng.choice(times), "entity": entity,
                     "change_type": ctype, "description": desc, "owner": rng.choice(OWNERS),
                     "lines_changed": rng.randint(lo, hi), "has_rollback_plan": rollback,
                     "tested_in_canary": canary})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    build(args.seed).to_csv(args.out_dir / "pending_demo.csv", index=False)
    pd.DataFrame(TIERS, columns=["entity", "tier", "downstream_dependents"]).to_csv(
        args.out_dir / "service_tiers.csv", index=False)
    pd.DataFrame(FREEZE).to_csv(args.out_dir / "freeze_windows.csv", index=False)
    print(f"wrote {len(PLAN)} pending changes, {len(TIERS)} service tiers, {len(FREEZE)} freeze window -> {args.out_dir}")


if __name__ == "__main__":
    main()
