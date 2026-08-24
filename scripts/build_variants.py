"""Assemble the data-curation ablation variants from seeds plus the generated
pool.

    uv run python scripts/build_variants.py --sizes 250 500 1000 2000

Writes one JSONL per tier and size into var/training/, each row carrying its
source and verification flag, so a training run's dataset is a file whose
provenance can be audited row by row. The verified-id list lives in
datasets/verified-generated.txt, one id per line, appended by whoever did the
human pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from nishana.data.curation import (  # noqa: E402
    ABLATION_SIZES,
    TIERS,
    SeedRow,
    TrainingRow,
    build_variant,
)
from nishana.data.hashing import content_hash  # noqa: E402

SEEDS_PATH = Path("datasets/seeds.jsonl")
GENERATED_PATH = Path("var/generated.jsonl")
VERIFIED_PATH = Path("datasets/verified-generated.txt")
OUT_DIR = Path("var/training")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(ABLATION_SIZES),
        help=f"subset of {ABLATION_SIZES}",
    )
    args = parser.parse_args()

    seeds = [
        SeedRow.model_validate(json.loads(line))
        for line in SEEDS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    generated: list[TrainingRow] = []
    if GENERATED_PATH.is_file():
        generated = [
            TrainingRow.model_validate(json.loads(line))
            for line in GENERATED_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    verified_ids = set()
    if VERIFIED_PATH.is_file():
        verified_ids = {
            line.strip()
            for line in VERIFIED_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    missing = [vid for vid in sorted(verified_ids) if vid not in {g.id for g in generated}]
    if missing:
        print(f"warning: {len(missing)} verified ids not present in the generated pool")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tier in TIERS:
        for size in args.sizes:
            rows = build_variant(tier, seeds, generated, verified_ids, size)
            path = OUT_DIR / f"{tier}-{size}.jsonl"
            path.write_text(
                "\n".join(json.dumps(row.model_dump(), ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            digest = content_hash([row.model_dump() for row in rows])
            print(f"{path}: n={len(rows)} hash={digest[:24]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
