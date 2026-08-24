"""Aggregate scored runs into the ablation table the README publishes.

    uv run python scripts/train_eval.py

Reads results/runs.yaml, scores every listed prediction file on the automatic
columns (intent accuracy, entity F1, similarity), merges blind judged quality
from results/judged/, and writes results/ablation.json. Judged columns stay
null until judge_quality.py has compared the runs; the table prints the gap
rather than pretending it is zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from nishana.eval.suite import score_run  # noqa: E402
from nishana.report import fidelity_quality_gap  # noqa: E402

RUNS_FILE = Path("results/runs.yaml")
JUDGED_DIR = Path("results/judged")
ABLATION_OUT = Path("results/ablation.json")


def load_runs(path: Path) -> list[dict]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    runs = payload.get("runs")
    if not runs:
        raise SystemExit(f"{path} lists no runs; add one entry per prediction file")
    return runs


def judged_lookup() -> dict[tuple[str, str], dict]:
    """Index judged verdict files by the pair of side names they compared."""
    out: dict[tuple[str, str], dict] = {}
    if not JUDGED_DIR.is_dir():
        return out
    for path in sorted(JUDGED_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        left, right = payload["left_name"], payload["right_name"]
        total = payload["left_wins"] + payload["right_wins"]
        quality_left = round(payload["left_wins"] / total, 4) if total else None
        rate = payload.get("swap_disagreement_rate", 0.0)
        out.setdefault((left, right), {"quality": quality_left, "swap_rate": rate})
        quality_right = None if quality_left is None else round(1 - quality_left, 4)
        out.setdefault((right, left), {"quality": quality_right, "swap_rate": rate})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=str(RUNS_FILE))
    args = parser.parse_args()

    runs = load_runs(Path(args.runs))
    judged = judged_lookup()

    variants = []
    for entry in runs:
        name = entry["name"]
        scored = score_run(entry["predictions"])
        row = {
            "variant": name,
            "variant_kind": entry.get("kind", "adapter"),
            "model": entry.get("model", "unknown"),
            "model_date": entry.get("date", ""),
            "n_items": int(scored["n"]),
            "errors": int(scored["errors"]),
            "intent_accuracy": scored["intent_accuracy"],
            "entity_f1": scored["entity_f1"],
            "similarity": scored["similarity"],
            "judged_quality": None,
            "judge_agreement": None,
        }
        for other in runs:
            match = judged.get((name, other["name"]))
            if match and match["quality"] is not None:
                row["judged_quality"] = match["quality"]
                row["judge_agreement"] = match["swap_rate"]
                break
        variants.append(row)

    ABLATION_OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "variants": variants,
        "fidelity_quality_gap": fidelity_quality_gap([v for v in variants if v["errors"] == 0]),
    }
    ABLATION_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    header = f"{'variant':<34} {'intent':>7} {'entF1':>7} {'sim':>7} {'judge':>7}"
    print(header)
    for v in sorted(variants, key=lambda x: x["variant"]):
        judge_col = "-" if v["judged_quality"] is None else f"{v['judged_quality']:.3f}"
        print(
            f"{v['variant']:<34} {v['intent_accuracy']:>7.3f} {v['entity_f1']:>7.3f}"
            f" {v['similarity']:>7.3f} {judge_col:>7}"
        )
    gap = payload["fidelity_quality_gap"]
    if gap:
        direction = (
            "INVERTED: similarity winner is not the judged-quality winner"
            if gap["inverted"]
            else "no inversion on this task"
        )
        print(f"fidelity vs quality: {direction}")
    print(f"wrote {ABLATION_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
