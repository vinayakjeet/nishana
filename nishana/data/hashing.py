from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def content_hash(items: list[dict]) -> str:
    """Order-invariant content hash.

    Sorting by id first means reordering rows does not invalidate a baseline,
    while editing any field does. Same scheme ShipGate uses, so a hash produced
    here reads the same way over there.
    """
    canonical = [
        json.dumps(item, sort_keys=True, separators=(",", ":"))
        for item in sorted(items, key=lambda i: i["id"])
    ]
    digest = hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def write_manifest(
    path: str | Path,
    dataset_id: str,
    items: list[dict],
    slices: dict[str, int],
) -> str:
    digest = content_hash(items)
    manifest = {
        "datasets": {
            dataset_id: {
                "id": dataset_id,
                "path": str(Path(path).as_posix()),
                "hash": digest,
                "n": len(items),
                "slice_counts": slices,
            }
        }
    }
    Path(path).write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return digest


def read_hash(path: str | Path) -> str | None:
    manifest = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    datasets = manifest.get("datasets", {})
    entry = datasets.get("tickets-eval") or next(iter(datasets.values()), None)
    return entry.get("hash") if entry else None
