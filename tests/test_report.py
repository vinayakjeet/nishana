from __future__ import annotations

import json
from pathlib import Path

from nishana.report import fidelity_quality_gap, render_status, status_number


def _ablation(tmp_path: Path, variants: list[dict]) -> Path:
    path = tmp_path / "ablation.json"
    path.write_text(json.dumps({"variants": variants}), encoding="utf-8")
    return tmp_path


def test_status_number_is_none_without_artifacts(tmp_path: Path) -> None:
    assert status_number(tmp_path) is None


def test_status_number_is_the_adapter_margin_over_frontier(tmp_path: Path) -> None:
    variants = [
        {
            "variant": "frontier-v2",
            "variant_kind": "frontier",
            "model": "groq:x",
            "intent_accuracy": 0.75,
            "judged_quality": 0.60,
        },
        {
            "variant": "synthetic-only-500",
            "variant_kind": "adapter",
            "model": "adapter",
            "intent_accuracy": 0.91,
            "judged_quality": 0.55,
        },
    ]
    headline = status_number(_ablation(tmp_path, variants))
    assert headline["value"] == round(0.91 - 0.75, 4)
    assert headline["judged_quality_gap"] == round(0.55 - 0.60, 4)


def test_fidelity_quality_gap_detects_the_inversion() -> None:
    variants = [
        {"variant": "synthetic-only", "similarity": 0.92, "judged_quality": 0.41},
        {"variant": "verified-heavy", "similarity": 0.71, "judged_quality": 0.63},
    ]
    gap = fidelity_quality_gap(variants)
    assert gap["inverted"] is True
    assert gap["wins_similarity"] == "synthetic-only"
    assert gap["wins_quality"] == "verified-heavy"


def test_fidelity_quality_gap_reports_agreement_too() -> None:
    variants = [
        {"variant": "a", "similarity": 0.9, "judged_quality": 0.9},
        {"variant": "b", "similarity": 0.7, "judged_quality": 0.7},
    ]
    assert fidelity_quality_gap(variants)["inverted"] is False


def test_render_status_names_what_is_pending(tmp_path: Path) -> None:
    text = render_status(tmp_path)
    assert "pending first training run" in text
    assert "scripts/decontaminate.py" in text
    assert "scripts/cost_table.py" in text


def test_render_status_prints_the_table_when_present(tmp_path: Path) -> None:
    (tmp_path / "decontam.json").write_text(json.dumps({
        "checks": {"5": {"overlap_pct": 0.83, "items_with_overlap": 1, "n_eval_items": 120}},
        "removed_ids_note": "report only; nothing removed",
    }), encoding="utf-8")
    (tmp_path / "cost.json").write_text(json.dumps({
        "serving_per_1k_usd": 0.03,
        "frontier_per_1k_usd": 0.15,
        "breakeven": {"breakeven_requests": 250000, "savings_per_1k_usd": 0.12},
    }), encoding="utf-8")
    text = render_status(tmp_path)
    assert "0.83%" in text
    assert "250,000" in text
