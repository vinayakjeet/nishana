from __future__ import annotations

import re

from nishana.types import Entity

_PUNCT = re.compile(r"[\s_\-]+")

INTENTS = ("order_status", "refund", "account", "other")


def normalize_intent(raw: str) -> str:
    """Fold the surface variants a model emits (`Refund`, `refund_request`,
    `REFUND`) onto one label. Normalisation rules are fixed before scoring;
    loosening them after seeing errors is how accuracy gets edited upward."""
    return _PUNCT.sub("_", raw.strip().lower()).strip("_")


def intent_correct(prediction: str, expected: str) -> bool:
    return normalize_intent(prediction) == normalize_intent(expected)


def _norm_value(value: str) -> str:
    """Casefold and drop punctuation inside values, because `Rs 1499` and
    `rs1,499` are the same extraction on code-mixed tickets."""
    return re.sub(r"[^\w]", "", value.lower())


def _coerce(raw: Entity | dict) -> Entity:
    """Candidates arrive as parsed JSON dicts, gold as pydantic models; score
    both through one path so the metric never depends on caller plumbing."""
    return raw if isinstance(raw, Entity) else Entity.model_validate(raw)


def entity_f1(
    predicted: list, gold: list, text: str, strict: bool = True
) -> float:
    """Span F1 over typed entities.

    Strict counts an entity correct when its type matches and its normalised
    value equals a gold value that appears verbatim in the ticket, so
    plausible-looking inventions score zero. Relaxed accepts case-insensitive
    substring containment in either direction, which absorbs spacing and symbol
    drift around amounts and dates.
    """
    predicted = [_coerce(p) for p in predicted]
    gold = [_coerce(g) for g in gold]
    if not gold and not predicted:
        return 1.0
    if not gold or not predicted:
        return 0.0

    def hit(pred: Entity, ref: Entity) -> bool:
        if pred.type != ref.type:
            return False
        pv, rv = _norm_value(pred.value), _norm_value(ref.value)
        if strict:
            return pv == rv and ref.value.lower() in text.lower()
        return pv in rv or rv in pv

    correct = sum(1 for p in predicted if any(hit(p, g) for g in gold))
    precision = correct / len(predicted)
    recall = correct / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
