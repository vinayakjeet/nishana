"""Blind judged quality between two prediction files, both orders every time.

    uv run python scripts/judge_quality.py \
        --left results/baseline/frontier-...jsonl \
        --right results/ablation/synthetic-only-500.jsonl

The judge sees the ticket and two answers labelled A and B. It never sees which
variant or model produced an answer, so style familiarity is the only bias it
can carry in. Every pair runs twice with the candidates swapped across slots;
only consistent winners count, and the swap-disagreement rate prints beside the
score instead of being averaged away.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import nishana  # noqa: E402,F401
from llm.client import ChatClient  # noqa: E402
from llm.types import ChatMessage  # noqa: E402
from nishana.data.loader import load_jsonl  # noqa: E402
from nishana.eval.judge import (  # noqa: E402
    consistent_winner,
    order_for,
    parse_verdict,
    render_pair,
    swap_disagreement_rate,
)

DATASET = Path("datasets/tickets-eval.jsonl")
RESULTS_DIR = Path("results/judged")


def load_predictions(path: str) -> dict[str, dict]:
    return {
        row["item_id"]: row
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def _side_of_slot(slot_holding_left: str, verdict: str) -> str | None:
    """Translate a slot letter back onto which input file it came from."""
    if verdict == "TIE" or verdict == "INVALID":
        return None
    if verdict == slot_holding_left:
        return "LEFT"
    return "RIGHT"


async def judge(
    left: dict[str, dict],
    right: dict[str, dict],
    dataset: str,
    provider: str,
    model: str | None,
    rpm: float,
) -> list[dict]:
    tickets = {t.id: t for t in load_jsonl(dataset)}
    client = ChatClient()
    gap = 60.0 / rpm if rpm > 0 else 0.0
    records: list[dict] = []

    for item_id in sorted(set(left) & set(right)):
        if left[item_id].get("error") or right[item_id].get("error"):
            continue
        prompt = tickets[item_id].input["prompt"]
        ab = order_for(item_id) == "ab"

        pair_letters = []
        # Pass one puts the left file's output in slot A on 'ab' items and in
        # slot B otherwise; pass two is the reverse. Both orders, always.
        for candidates, slot_of_left in (
            ((left[item_id], right[item_id]), "A" if ab else "B"),
            ((right[item_id], left[item_id]), "B" if ab else "A"),
        ):
            messages = [
                ChatMessage(**m)
                for m in render_pair(
                    item_id, prompt, candidates[0]["prediction"], candidates[1]["prediction"]
                )
            ]
            response = await client.complete(provider, messages, model=model)
            pair_letters.append((slot_of_left, parse_verdict(response.text)))
            if gap:
                await asyncio.sleep(gap)

        sides = tuple(_side_of_slot(holding, letter) for holding, letter in pair_letters)
        records.append(
            {
                "item_id": item_id,
                "first": sides[0] or "TIE",
                "swapped": sides[1] or "TIE",
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--rpm", type=float, default=25.0)
    args = parser.parse_args()

    records = asyncio.run(
        judge(
            load_predictions(args.left),
            load_predictions(args.right),
            args.dataset,
            args.provider,
            args.model,
            args.rpm,
        )
    )

    pairs = [(r["first"], r["swapped"]) for r in records]
    rate = swap_disagreement_rate(pairs)
    winners = [consistent_winner(a, b) for a, b in pairs]
    left_wins = sum(1 for w in winners if w == "A")
    right_wins = sum(1 for w in winners if w == "B")
    decided = left_wins + right_wins

    out = RESULTS_DIR / f"{args.left_name}-vs-{args.right_name}-{datetime.now(UTC).date()}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "judge_provider": args.provider,
        "judge_model": args.model or "default",
        "left_name": args.left_name,
        "right_name": args.right_name,
        "n_pairs": len(records),
        "left_wins": left_wins,
        "right_wins": right_wins,
        "undecided": len(records) - decided,
        "swap_disagreement_rate": rate,
        "records": records,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"pairs judged      : {len(records)}")
    print(f"{args.left_name:<12} wins : {left_wins}")
    print(f"{args.right_name:<12} wins : {right_wins}")
    print(f"swap disagreement : {rate:.3f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
