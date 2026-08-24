from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import decontaminate


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return path


def _eval(id: str, text: str) -> dict:
    return {
        "id": id,
        "input": {"prompt": text},
        "expected": {"intent": "other", "entities": []},
        "slices": ["lang:en"],
        "meta": {},
    }


def test_report_only_leaves_the_dataset_alone(tmp_path: Path, monkeypatch) -> None:
    shared = "please refund my money now for order OD-777"
    seeds = _write(
        tmp_path / "seeds.jsonl",
        [{"id": "s1", "text": "customer asked us to " + shared + " immediately"}],
    )
    eval_set = _write(
        tmp_path / "eval.jsonl",
        [_eval("a", shared), _eval("b", "totally unrelated words here")],
    )
    results = tmp_path / "results"
    monkeypatch.setattr(decontaminate, "RESULTS_PATH", results / "decontam.json")

    payload = decontaminate.run(seeds, eval_set, apply=False)

    assert len(eval_set.read_text(encoding="utf-8").splitlines()) == 2
    assert payload["checks"]["5"]["items_with_overlap"] == 1
    assert payload["checks"]["5"]["offenders"] == {"a": 5}
    assert (results / "decontam.json").is_file()


def test_apply_removes_overlapping_items_and_records_it(tmp_path: Path, monkeypatch) -> None:
    shared = "the delivery person never rang the doorbell at all"
    seeds = _write(
        tmp_path / "seeds.jsonl",
        [{"id": "s1", "text": shared + " yesterday"}],
    )
    eval_set = _write(
        tmp_path / "eval.jsonl",
        [_eval("dirty", shared), _eval("clean", "where is my parcel going")],
    )
    monkeypatch.setattr(decontaminate, "RESULTS_PATH", tmp_path / "results" / "decontam.json")
    monkeypatch.setattr(decontaminate, "MANIFEST_PATH", tmp_path / "manifest.yaml")

    payload = decontaminate.run(seeds, eval_set, apply=True)

    survivors = [
        json.loads(line)["id"]
        for line in eval_set.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert survivors == ["clean"]
    assert payload["removed_ids_note"] == "1 removed"
