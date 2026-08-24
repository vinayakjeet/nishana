from __future__ import annotations

from pathlib import Path

import yaml

from nishana.data.hashing import content_hash, read_hash, write_manifest


def _item(id: str, text: str = "same") -> dict:
    return {"id": id, "input": {"prompt": text}, "expected": {"intent": "other"}}


def test_reordering_does_not_change_the_hash() -> None:
    forward = content_hash([_item("a"), _item("b"), _item("c")])
    backward = content_hash([_item("c"), _item("b"), _item("a")])
    assert forward == backward


def test_editing_any_field_changes_the_hash() -> None:
    base = [_item("a"), _item("b")]
    edited = _item("b")
    edited["expected"]["intent"] = "refund"
    assert content_hash(base) != content_hash([_item("a"), edited])


def test_manifest_round_trip(tmp_path: Path) -> None:
    items = [_item("a"), _item("b", "different")]
    manifest_path = tmp_path / "manifest.yaml"
    digest = write_manifest(manifest_path, "tickets-eval", items, {"intent:other": 2})

    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    entry = payload["datasets"]["tickets-eval"]
    assert entry["hash"] == digest
    assert entry["n"] == 2
    assert read_hash(manifest_path) == digest
    assert digest.startswith("sha256:")
