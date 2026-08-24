from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path("results")


def load_json(path: Path) -> dict | list | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _variants(results_dir: Path) -> list | None:
    payload = load_json(results_dir / "ablation.json")
    if payload is None:
        return None
    return payload.get("variants") if isinstance(payload, dict) else payload


def status_number(results_dir: Path = RESULTS_DIR) -> dict | None:
    """The project's headline number, in the same spirit as ShipGate's kappa
    and noise floor: the best adapter's margin over the fair frontier baseline
    on intent accuracy, in percentage points, with the judged-quality column
    beside it. None until an ablation run exists, because a placeholder would
    be worse than printing nothing."""
    ablation = _variants(results_dir)
    if not ablation:
        return None
    adapters = [v for v in ablation if v.get("variant_kind", "adapter") == "adapter"]
    frontier = [v for v in ablation if v["variant"] == "frontier-v2"]
    if not adapters or not frontier:
        return None
    best = max(adapters, key=lambda v: v["intent_accuracy"])
    base = frontier[0]
    return {
        "metric": "best adapter intent accuracy minus frontier v2 baseline (points)",
        "value": round(best["intent_accuracy"] - base["intent_accuracy"], 4),
        "adapter": best["variant"],
        "frontier": base["model"],
        "judged_quality_gap": (
            None
            if best.get("judged_quality") is None or base.get("judged_quality") is None
            else round(best["judged_quality"] - base["judged_quality"], 4)
        ),
    }


def fidelity_quality_gap(ablation: list[dict]) -> dict | None:
    """The inversion the honesty reference predicts: the variant that wins on
    distribution similarity but loses on blind judged quality.

    Reported as a finding whatever direction it points. Finding no gap on this
    task would contradict the published result and that is worth printing too.
    """
    scored = [
        v
        for v in ablation
        if v.get("similarity") is not None and v.get("judged_quality") is not None
    ]
    if len(scored) < 2:
        return None
    best_similarity = max(scored, key=lambda v: v["similarity"])
    best_quality = max(scored, key=lambda v: v["judged_quality"])
    return {
        "wins_similarity": best_similarity["variant"],
        "wins_quality": best_quality["variant"],
        "inverted": best_similarity["variant"] != best_quality["variant"],
    }


def render_status(results_dir: Path = RESULTS_DIR) -> str:
    lines: list[str] = []
    decontam = load_json(results_dir / "decontam.json")
    ablation = _variants(results_dir)
    cost = load_json(results_dir / "cost.json")

    lines.append("Nishana status")
    lines.append("=" * 72)

    headline = status_number(results_dir)
    if headline:
        gap = headline["judged_quality_gap"]
        lines.append(
            f"headline      : {headline['metric']}: {headline['value']:+.3f}"
            f"  ({headline['adapter']} vs {headline['frontier']})"
        )
        if gap is not None:
            lines.append(f"judged quality: adapter {gap:+.3f} vs frontier (blind pairwise)")
    else:
        lines.append(
            "headline      : pending first training run. Produce it with"
            " scripts/train_eval.py on Kaggle, then rerun this script."
        )

    if decontam:
        for n in sorted(decontam.get("checks", {})):
            entry = decontam["checks"][n]
            article = "an" if n.startswith("8") else "a"
            lines.append(
                f"contamination : {entry['overlap_pct']}% of eval items share"
                f" {article} {n}-gram with seeds;"
                f" {entry['items_with_overlap']}/{entry['n_eval_items']} items"
                f" ({decontam.get('removed_ids_note', '')})"
            )
    else:
        lines.append("contamination : pending. Run scripts/decontaminate.py.")

    if ablation:
        lines.append("")
        lines.append(f"{'variant':<34} {'intent':>7} {'entF1':>7} {'sim':>7} {'judge':>7}")
        for v in sorted(ablation, key=lambda x: x["variant"]):
            lines.append(
                f"{v['variant']:<34} {_fmt(v['intent_accuracy']):>7} {_fmt(v['entity_f1']):>7}"
                f" {_fmt(v['similarity']):>7} {_fmt(v['judged_quality']):>7}"
            )

    if cost:
        be = cost["breakeven"]["breakeven_requests"]
        lines.append("")
        lines.append(
            f"serving       : ${cost['serving_per_1k_usd']:.4f}/1k self-hosted"
            f" vs ${cost['frontier_per_1k_usd']:.4f}/1k frontier API"
        )
        if be is None:
            lines.append(
                "breakeven     : never; self-hosting is not cheaper per 1k"
                " under these assumptions"
            )
        else:
            lines.append(
                f"breakeven     : {be:,} requests recovers the one-time training cost"
            )
    else:
        lines.append("cost table    : pending. Run scripts/cost_table.py.")

    lines.append("=" * 72)
    return "\n".join(lines)
