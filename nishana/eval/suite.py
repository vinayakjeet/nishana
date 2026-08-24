from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from nishana.data.loader import load_jsonl
from nishana.eval.exact import entity_f1, intent_correct
from nishana.eval.similarity import render_reference, rouge_l

_JSON_LINE = re.compile(r"\{.*\}", re.DOTALL)


class UnparsableOutput(Exception):
    pass


def parse_output(text: str) -> tuple[str, list[dict]]:
    """Pull intent and entities out of a candidate's raw output.

    Both frontier and fine-tuned models are asked for one JSON line and both
    occasionally wrap it in prose or code fences, so extraction tolerates that
    but nothing else: an output with no parsable object is an error counted as
    such, not a free zero quietly folded into an accuracy.
    """
    match = _JSON_LINE.search(text)
    if not match:
        raise UnparsableOutput(f"no JSON object in output: {text[:80]!r}")
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise UnparsableOutput(f"unparsable JSON in output: {exc}") from exc
    intent = str(payload.get("intent", ""))
    entities = payload.get("entities") or []
    if not isinstance(entities, list):
        entities = []
    return intent, [e for e in entities if isinstance(e, dict) and "type" in e and "value" in e]


def score_run(
    predictions_path: str | Path,
    dataset_path: str | Path = "datasets/tickets-eval.jsonl",
) -> dict[str, float]:
    """Score one prediction file on all three automatic columns.

    Errors are counted separately from scores. A failed item scoring zero is
    numerically identical to a real wrong answer, and conflating them lets a
    JSON parsing bug read as a model getting worse.
    """
    tickets = {t.id: t for t in load_jsonl(dataset_path)}
    rows = [
        json.loads(line)
        for line in Path(predictions_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    intents, f1s, sims = [], [], []
    errors = 0
    for row in rows:
        ticket = tickets.get(row["item_id"])
        if ticket is None:
            continue
        if row.get("error"):
            errors += 1
            continue
        text = ticket.input["prompt"]
        gold_intent = ticket.expected.intent
        gold_entities = [e.model_dump() for e in ticket.expected.entities]
        reference = render_reference(gold_intent, gold_entities)
        try:
            pred_intent, pred_entities = parse_output(row["prediction"])
        except UnparsableOutput:
            errors += 1
            continue
        intents.append(intent_correct(pred_intent, gold_intent))
        f1s.append(entity_f1(pred_entities, gold_entities, text))
        sims.append(rouge_l(row["prediction"], reference))

    n = len(rows)
    return {
        "n": float(n),
        "errors": float(errors),
        "intent_accuracy": _mean(intents),
        "entity_f1": _mean(f1s),
        "similarity": _mean(sims),
    }


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def variance(values: list[float]) -> dict[str, float]:
    """Mean and spread over repeated runs. Every published number carries at
    least three runs behind it; this is where the spread comes from."""
    if len(values) < 2:
        return {"mean": _mean(values), "stdev": 0.0, "spread": 0.0}
    return {
        "mean": round(statistics.mean(values), 4),
        "stdev": round(statistics.stdev(values), 4),
        "spread": round(max(values) - min(values), 4),
    }
