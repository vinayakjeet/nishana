from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nishana.candidates as candidates
from nishana.types import Ticket


def _tickets(n: int) -> list[Ticket]:
    return [
        Ticket(
            id=f"tkt-{i:03d}",
            input={"prompt": f"ticket {i} about order OD-{i}"},
            expected={
                "intent": "order_status",
                "entities": [{"type": "order_id", "value": f"OD-{i}"}],
            },
            slices=["intent:order_status"],
        )
        for i in range(1, n + 1)
    ]


def _replay_row(item_id: str) -> dict:
    return {
        "item_id": item_id,
        "prediction": '{"intent": "order_status", "entities": []}',
        "model": "replay-frontier",
    }


def test_run_replay_is_deterministic_and_labelled(tmp_path: Path) -> None:
    tickets = _tickets(2)
    replay = tmp_path / "replay.json"
    rows = [_replay_row("tkt-001"), _replay_row("tkt-002")]
    replay.write_text(json.dumps(rows), encoding="utf-8")

    first = candidates.run_replay(tickets, replay)
    second = candidates.run_replay(tickets, replay)
    assert [r.prediction for r in first] == [r.prediction for r in second]
    assert all("replay" in r.model for r in first)


def test_run_replay_fails_loudly_on_missing_rows(tmp_path: Path) -> None:
    tickets = _tickets(3)
    replay = tmp_path / "replay.json"
    missing = [{"item_id": "tkt-001", "prediction": "x", "model": "m"}]
    replay.write_text(json.dumps(missing), encoding="utf-8")
    try:
        candidates.run_replay(tickets, replay)
        raised = False
    except KeyError:
        raised = True
    assert raised


def test_write_predictions_round_trip(tmp_path: Path) -> None:
    from nishana.candidates import run_replay, write_predictions

    tickets = _tickets(2)
    replay = tmp_path / "replay.json"
    rows = [{"item_id": t.id, "prediction": f"p-{t.id}", "model": "m"} for t in tickets]
    replay.write_text(json.dumps(rows), encoding="utf-8")
    out = tmp_path / "out.jsonl"
    write_predictions(out, run_replay(tickets, replay))
    loaded = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(loaded) == 2
    assert loaded[0]["item_id"] == "tkt-001"
