from __future__ import annotations

import json
from pathlib import Path

import pytest

from nishana.eval.suite import UnparsableOutput, parse_output, score_run, variance


def test_parse_output_tolerates_wrapped_json() -> None:
    text = (
        'Here you go:\n```json\n{"intent": "refund", '
        '"entities": [{"type": "order_id", "value": "OD-1"}]}\n```'
    )
    intent, entities = parse_output(text)
    assert intent == "refund"
    assert entities[0]["type"] == "order_id"


def test_parse_output_rejects_prose_without_json() -> None:
    with pytest.raises(UnparsableOutput):
        parse_output("I am not sure what you mean")


def _write_dataset(tmp_path: Path) -> Path:
    rows = [
        {
            "id": "tkt-001",
            "input": {"prompt": "order OD-1 ka refund chahiye Rs 500"},
            "expected": {
                "intent": "refund",
                "entities": [
                    {"type": "order_id", "value": "OD-1"},
                    {"type": "amount", "value": "Rs 500"},
                ],
            },
            "slices": ["intent:refund"],
        },
        {
            "id": "tkt-002",
            "input": {"prompt": "kuch aur hi baat hai"},
            "expected": {"intent": "other", "entities": []},
            "slices": ["intent:other"],
        },
    ]
    path = tmp_path / "mini-eval.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def _write_predictions(
    tmp_path: Path, outputs: dict[str, str], errors: dict[str, str] | None = None
) -> Path:
    errors = errors or {}
    rows = [
        {
            "item_id": item_id,
            "prediction": output,
            "model": "test-model",
            "generated_at": "2026-08-24T00:00:00+00:00",
            "error": errors.get(item_id, ""),
        }
        for item_id, output in outputs.items()
    ]
    path = tmp_path / "preds.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_score_run_counts_errors_separately(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    good = (
        '{"intent": "refund", "entities": '
        '[{"type": "order_id", "value": "OD-1"}, {"type": "amount", "value": "Rs 500"}]}'
    )
    predictions = _write_predictions(
        tmp_path,
        {"tkt-001": good, "tkt-002": ""},
        errors={"tkt-002": "rate limited"},
    )
    scores = score_run(predictions, dataset)
    assert scores["errors"] == 1
    # One scored item, scored correctly.
    assert scores["n"] == 2
    assert scores["intent_accuracy"] == 1.0
    assert scores["entity_f1"] == 1.0


def test_score_run_penalizes_wrong_intent_and_inventions(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    invented = (
        '{"intent": "refund", "entities": '
        '[{"type": "order_id", "value": "OD-1"}, {"type": "amount", "value": "Rs 900"}]}'
    )
    predictions = _write_predictions(
        tmp_path,
        {
            # Right intent, invented amount.
            "tkt-001": invented,
            "tkt-002": '{"intent": "account", "entities": []}',
        },
    )
    scores = score_run(predictions, dataset)
    assert scores["intent_accuracy"] == 0.5
    assert 0.0 < scores["entity_f1"] < 1.0


def test_variance_reports_spread_over_runs() -> None:
    result = variance([0.80, 0.75, 0.85])
    assert result["mean"] == round((0.80 + 0.75 + 0.85) / 3, 4)
    assert result["spread"] == 0.10
    assert result["stdev"] > 0
    single = variance([0.9])
    assert single["spread"] == 0.0
