"""Measure and publish n-gram contamination between the synthetic seed set and
the eval set.

    uv run python scripts/decontaminate.py            # report only
    uv run python scripts/decontaminate.py --apply    # drop offenders, re-hash

The overlap numbers get published whatever they are: asserting "we ran a
decontamination check" without printing the counts is indistinguishable from
not running one, so this writes results/decontam.json either way. With --apply,
any eval item sharing a 5-gram or an 8-gram with any seed text is removed and
the dataset is re-versioned. The manifest hash moves with it, which invalidates
old baselines by design rather than letting two datasets share a score.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from nishana.data.decontam import check_overlap, decontaminate  # noqa: E402
from nishana.data.hashing import content_hash, write_manifest  # noqa: E402
from nishana.data.loader import DatasetError, load_jsonl, slice_counts  # noqa: E402

SEEDS_PATH = Path("datasets/seeds.jsonl")
EVAL_PATH = Path("datasets/tickets-eval.jsonl")
MANIFEST_PATH = Path("datasets/manifest.yaml")
RESULTS_PATH = Path("results/decontam.json")
CHECKED_NS = (5, 8)


def _load_raw(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run(seeds_path: Path, eval_path: Path, apply: bool) -> dict:
    seeds = _load_raw(seeds_path)
    eval_items = _load_raw(eval_path)
    seed_texts = [row["text"] for row in seeds]

    checks = {}
    for n in CHECKED_NS:
        report = check_overlap(seed_texts, eval_items, n)
        checks[str(n)] = {
            "shared_ngrams": report.shared_ngrams,
            "items_with_overlap": report.items_with_overlap,
            "n_eval_items": report.n_eval_items,
            "overlap_pct": report.overlap_pct,
            "offenders": report.offenders,
        }

    removed_note = "report only; nothing removed"
    if apply:
        kept, removed = decontaminate(seed_texts, eval_items, drop_on=CHECKED_NS)
        removed_note = f"{len(removed)} removed"
        if removed:
            eval_path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in kept) + "\n",
                encoding="utf-8",
            )
            try:
                items = load_jsonl(eval_path)
            except DatasetError as exc:
                raise SystemExit(f"rewritten dataset failed validation: {exc}") from exc
            digest = content_hash([item.model_dump() for item in items])
            write_manifest(
                MANIFEST_PATH,
                "tickets-eval",
                [item.model_dump() for item in items],
                slice_counts(items),
            )
            print(f"dataset rewritten: {len(removed)} items removed, new hash {digest[:24]}...")
            print("baselines measured on the old hash are invalidated by design.")
        else:
            print("no overlaps at checked lengths; dataset unchanged.")

    result = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seeds_n": len(seeds),
        "checks": checks,
        "removed_ids_note": removed_note,
        "applied": bool(apply),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=str(SEEDS_PATH))
    parser.add_argument("--eval", dest="eval_set", default=str(EVAL_PATH))
    parser.add_argument(
        "--apply",
        action="store_true",
        help="remove overlapping eval items and rewrite the dataset + manifest",
    )
    args = parser.parse_args()

    result = run(Path(args.seeds), Path(args.eval_set), args.apply)
    for n, entry in sorted(result["checks"].items()):
        print(
            f"{n}-gram overlap: {entry['overlap_pct']}% of eval items "
            f"({entry['items_with_overlap']}/{entry['n_eval_items']}), "
            f"{entry['shared_ngrams']} shared n-grams"
        )
    if result["removed_ids_note"] != "report only; nothing removed":
        print(result["removed_ids_note"])
    print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
