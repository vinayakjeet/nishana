"""Build the cost table and break-even volume from deploy/pricing.yaml.

    uv run python scripts/cost_table.py --seconds-per-step 1.8 --rows 1000
    uv run scripts/cost_table.py --tokens-from results/baseline/frontier-v2-groq.jsonl

Kaggle training is free in currency but metered in quota, so training reports
GPU-hours against the weekly budget and treats paid training as the explicit
--training-usd scenario it actually is. Frontier token counts default to the
measured means from a baseline prediction file when one exists, falling back to
stated assumptions only when nothing has been measured.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from nishana.cost.model import (  # noqa: E402
    breakeven_volume,
    frontier_cost_per_1k,
    load_pricing,
    serving_cost_per_1k,
    training_cost,
)

PRICING = Path("deploy/pricing.yaml")
OUT_PATH = Path("results/cost.json")
BASELINE_GLOB = "results/baseline/frontier-*.jsonl"


def measured_tokens(pattern: str) -> tuple[int, int] | None:
    paths = sorted(Path().glob(pattern))
    if not paths:
        return None
    rows = [
        json.loads(line)
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tokens_in = [r["tokens_in"] for r in rows if r.get("tokens_in")]
    tokens_out = [r["tokens_out"] for r in rows if r.get("tokens_out")]
    if not tokens_in or not tokens_out:
        return None
    return round(statistics.fmean(tokens_in)), round(statistics.fmean(tokens_out))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--effective-batch", type=int, default=8)
    parser.add_argument("--seconds-per-step", type=float, required=True)
    parser.add_argument("--training-usd", type=float, default=0.0)
    parser.add_argument("--gpu", default="t4", choices=("t4", "a10g"))
    parser.add_argument("--pricing", default=str(PRICING))
    parser.add_argument(
        "--tokens-from",
        default=BASELINE_GLOB,
        help="glob of baseline prediction files to measure mean token counts from",
    )
    args = parser.parse_args()

    pricing = load_pricing(args.pricing)
    train = training_cost(args.rows, args.epochs, args.effective_batch, args.seconds_per_step)
    serving = serving_cost_per_1k(pricing, gpu=args.gpu)

    tokens = measured_tokens(args.tokens_from) or (450, 90)
    source = (
        "measured from baseline predictions"
        if measured_tokens(args.tokens_from)
        else "assumed; no baseline predictions found"
    )
    api = frontier_cost_per_1k(pricing, tokens_in=tokens[0], tokens_out=tokens[1])

    credits = pricing["modal"]["free_credits_usd"]
    covers = (
        int(credits / serving["total_per_1k_usd"])
        if serving["total_per_1k_usd"] > 0
        else None
    )
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "training": {
            **train,
            "paid_scenario_usd": args.training_usd,
            "weekly_budget_gpu_hours": 30,
        },
        "serving_per_1k_usd": serving["total_per_1k_usd"],
        "serving_detail": serving,
        "frontier_tokens": {"in": tokens[0], "out": tokens[1], "source": source},
        "frontier_per_1k_usd": api,
        "breakeven": breakeven_volume(args.training_usd, serving["total_per_1k_usd"], api),
        "modal_covers_requests": covers,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"training         : {train['steps']} steps, {train['gpu_hours']} GPU-hours,"
        f" ${train['usd']:.2f} on Kaggle quota"
    )
    print(
        f"serving (Modal {args.gpu}): ${payload['serving_per_1k_usd']:.4f} per 1k"
        " inferences at scale-to-zero"
    )
    print(
        f"frontier API     : ${api:.4f} per 1k"
        f" ({tokens[0]} in / {tokens[1]} out tokens, {source})"
    )
    be = payload["breakeven"]["breakeven_requests"]
    if be is None:
        print("break-even       : self-hosting never cheaper per 1k under these assumptions")
    elif args.training_usd == 0:
        savings = payload["breakeven"]["savings_per_1k_usd"]
        print(
            f"savings          : ${savings:.4f} per 1k; free-tier training"
            " means no volume to recover"
        )
        print(f"$30 credits cover: {covers:,} inferences")
    else:
        print(f"break-even       : {be:,} requests to recover ${args.training_usd:.2f}")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
