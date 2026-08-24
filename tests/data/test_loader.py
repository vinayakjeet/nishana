from __future__ import annotations

import json
from pathlib import Path

import pytest

from nishana.data.loader import DatasetError, load_jsonl, slice_counts
from nishana.types import Ticket


def _ticket(id: str, slices: list[str]) -> Ticket:
    return Ticket(
        id=id,
        input={"prompt": "Order OD-12345 kahan tak pahuncha?"},
        expected={
            "intent": "order_status",
            "entities": [{"type": "order_id", "value": "OD-12345"}],
        },
        slices=slices,
    )


def test_valid_dataset_loads(tmp_path: Path) -> None:
    path = tmp_path / "mini.jsonl"
    path.write_text(
        json.dumps({"id": "tkt-001", "input": {"prompt": "hi"}, "expected": {"intent": "refund"}}),
        encoding="utf-8",
    )
    items = load_jsonl(path)
    assert len(items) == 1
    assert items[0].expected.intent == "refund"


def test_malformed_json_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "mini.jsonl"
    path.write_text(
        '{"id": "a", "input": {"prompt": "x"}, "expected": {"intent": "other"}}\n{not json',
        encoding="utf-8",
    )
    with pytest.raises(DatasetError) as exc:
        load_jsonl(path)
    assert f"{path}:2" in str(exc.value)


def test_wrong_schema_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "mini.jsonl"
    path.write_text('{"id": 42, "input": {}, "expected": {}}', encoding="utf-8")
    with pytest.raises(DatasetError) as exc:
        load_jsonl(path)
    assert f"{path}:1" in str(exc.value)


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    row = '{"id": "%s", "input": {"prompt": "x"}, "expected": {"intent": "other"}}'
    path = tmp_path / "mini.jsonl"
    path.write_text((row % "a") + "\n\n" + (row % "b"), encoding="utf-8")
    assert len(load_jsonl(path)) == 2


def test_slice_counts_sorted_and_correct() -> None:
    tickets = [
        _ticket("b", ["intent:order_status", "lang:hinglish"]),
        _ticket("a", ["intent:order_status"]),
        _ticket("c", ["intent:refund"]),
    ]
    counts = slice_counts(tickets)
    assert counts == {
        "intent:order_status": 2,
        "intent:refund": 1,
        "lang:hinglish": 1,
    }
    assert list(counts) == sorted(counts)
