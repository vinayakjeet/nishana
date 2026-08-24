"""Generate the synthetic training pool from the hand-written seed set, with
the provenance recorded next to every row.

    uv run python scripts/generate_seeds.py --provider groq --rows 1500
    uv run python scripts/generate_seeds.py --provider mock --allow-mock --rows 20

Every generated row records the generator model, the date, and a hash of the
prompt version used. Synthetic-generation provenance is where contamination
hides, so it is written into the data file itself rather than kept in a
separate log that can drift from what was actually produced.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from llm.client import ChatClient  # noqa: E402
from llm.types import ChatMessage  # noqa: E402
from nishana.data.curation import SeedRow  # noqa: E402

SEEDS_PATH = Path("datasets/seeds.jsonl")
GENERATED_PATH = Path("var/generated.jsonl")

GENERATOR_PROMPT_VERSION = "gen-v1"

GENERATOR_PROMPT = """Here are examples of code-mixed Hinglish customer support
tickets for an e-commerce company, annotated with intent (order_status, refund,
account, or other) and entities (order_id, amount, date, product, phone).

Examples:
{examples}

Write {count} NEW tickets in the same style: natural Hinglish mixing Hindi
(Latin script), English, and occasional Devanagari. Vary sentence length,
greetings, spelling of common words (plz, sir, ki, nahi), and which entities
appear. At least one third must mention no order id at all.

Reply with one JSON object per line, no numbering:
{{"text": "<ticket>", "intent": "<intent>",
  "entities": [{{"type": "<type>", "value": "<exact substring>"}}]}}"""


def prompt_fingerprint() -> str:
    return hashlib.sha256(GENERATOR_PROMPT.encode()).hexdigest()[:12]


def _example_block(seeds: list[SeedRow], per_intent: int = 3) -> str:
    by_intent: dict[str, list[SeedRow]] = {}
    for seed in seeds:
        by_intent.setdefault(seed.intent, []).append(seed)
    lines = []
    for intent, rows in sorted(by_intent.items()):
        for row in rows[:per_intent]:
            lines.append(json.dumps({"text": row.text, "intent": intent}, ensure_ascii=False))
    return "\n".join(lines)


async def generate(provider: str, model: str | None, rows: int, batch: int) -> list[dict]:
    seeds = [
        SeedRow.model_validate(json.loads(line))
        for line in SEEDS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    client = ChatClient()
    messages = [
        ChatMessage(
            role="user",
            content=GENERATOR_PROMPT.format(examples=_example_block(seeds), count=batch),
        )
    ]
    out: list[dict] = []
    while len(out) < rows:
        response = await client.complete(provider, messages)
        when = datetime.now(UTC).isoformat(timespec="seconds")
        for line in response.text.splitlines():
            line = line.strip().strip("`")
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "text" not in payload or "intent" not in payload:
                continue
            out.append(
                {
                    "id": f"gen-{prompt_fingerprint()}-{len(out):05d}",
                    "text": str(payload["text"]),
                    "intent": str(payload["intent"]),
                    "entities": payload.get("entities") or [],
                    "generator": f"{provider}:{response.model}",
                    "generated_at": when,
                    "prompt_version": GENERATOR_PROMPT_VERSION,
                    "verified": False,
                }
            )
        if len(out) >= rows:
            break
    return out[:rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", default=None)
    parser.add_argument("--rows", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="permit the mock provider; its output is shape-valid but content-free",
    )
    args = parser.parse_args()

    if args.provider == "mock" and not args.allow_mock:
        raise SystemExit(
            "the mock provider writes deterministic placeholder tickets that would "
            "poison a real training pool. Pass --allow-mock only for smoke tests."
        )

    rows_out = asyncio.run(generate(args.provider, args.model, args.rows, args.batch))
    GENERATED_PATH.parent.mkdir(parents=True, exist_ok=True)
    GENERATED_PATH.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows_out) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows_out)} generated rows to {GENERATED_PATH}")
    print(f"generator prompt version {GENERATOR_PROMPT_VERSION} ({prompt_fingerprint()})")
    print("next: scripts/build_variants.py assembles training tiers from these.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
