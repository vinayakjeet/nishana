"""Run one candidate over the eval set and write its predictions.

    uv run python scripts/run_baseline.py --mode frontier --provider groq
    uv run python scripts/run_baseline.py --mode target --url http://localhost:8000/v1
    uv run python scripts/run_baseline.py --mode replay --replay-file fixtures/replay-frontier.json

Modes cover the three things this repo ever scores: the frontier API baseline,
a served adapter speaking ShipGate's adopting contract, and a recorded replay
file for offline verification. Every row is stamped with model name and date;
a number without both does not leave this script.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import nishana  # noqa: E402,F401
from nishana.candidates import run_frontier, run_replay, run_target, write_predictions  # noqa: E402
from nishana.data.loader import load_jsonl  # noqa: E402
from nishana.eval.frontier import DEFAULT_VERSION  # noqa: E402

DATASET = Path("datasets/tickets-eval.jsonl")
BASELINE_DIR = Path("results/baseline")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("frontier", "target", "replay"), required=True)
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt-version", default=DEFAULT_VERSION)
    parser.add_argument("--url", default=None)
    parser.add_argument("--replay-file", default=None)
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument(
        "--label",
        default=None,
        help="output filename stem; defaults to mode plus today's date",
    )
    parser.add_argument("--rpm", type=float, default=25.0)
    args = parser.parse_args()

    items = load_jsonl(args.dataset)

    if args.mode == "frontier":
        rows = asyncio.run(
            run_frontier(items, args.provider, args.model or "", args.prompt_version, args.rpm)
        )
        model_slug = args.model or "default"
        stem = args.label or f"frontier-{args.prompt_version}-{args.provider}-{model_slug}"
    elif args.mode == "target":
        if not args.url:
            raise SystemExit("--mode target needs --url")
        rows = asyncio.run(run_target(items, args.url))
        stem = args.label or "adapter"
    else:
        if not args.replay_file:
            raise SystemExit("--mode replay needs --replay-file")
        rows = run_replay(items, args.replay_file)
        stem = args.label or "replay"

    out = BASELINE_DIR / f"{stem}-{datetime.now(UTC).date()}.jsonl"
    write_predictions(out, rows)
    errors = sum(1 for row in rows if row.error)
    print(f"wrote {len(rows)} predictions to {out} ({errors} errored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
