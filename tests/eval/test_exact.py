from __future__ import annotations

from nishana.eval.exact import entity_f1, intent_correct, normalize_intent
from nishana.types import Entity


def test_normalize_folds_surface_variants() -> None:
    assert normalize_intent("Refund") == "refund"
    assert normalize_intent("refund_request") == "refund_request"
    assert normalize_intent("  ORDER-STATUS ") == "order_status"


def test_intent_correct_ignores_case_and_punctuation() -> None:
    assert intent_correct("Order Status", "order_status")
    assert not intent_correct("refund", "account")


def test_entity_f1_perfect_and_empty() -> None:
    gold = [Entity(type="order_id", value="OD-12345")]
    pred = [Entity(type="order_id", value="OD-12345")]
    text = "order OD-12345 kahan hai"
    assert entity_f1(pred, gold, text) == 1.0
    assert entity_f1([], [], text) == 1.0
    assert entity_f1([], gold, text) == 0.0


def test_strict_mode_rejects_values_absent_from_the_ticket() -> None:
    # The model emits the right-shaped id that was never in the ticket: an
    # invention, and the most dangerous kind because it looks plausible.
    gold = [Entity(type="order_id", value="OD-12345")]
    pred = [Entity(type="order_id", value="OD-12345")]
    assert entity_f1(pred, gold, "mera order kahan hai") == 0.0


def test_relaxed_mode_tolerates_symbol_drift() -> None:
    gold = [Entity(type="amount", value="Rs 1499")]
    pred = [Entity(type="amount", value="rs 1499.")]
    assert entity_f1([pred[0]], gold, "paid rs 1499 for it", strict=False) == 1.0


def test_type_mismatch_never_scores() -> None:
    gold = [Entity(type="date", value="12 Aug")]
    pred = [Entity(type="product", value="12 Aug")]
    assert entity_f1(pred, gold, "delivered on 12 Aug") == 0.0


def test_partial_predictions_average_through_f1() -> None:
    gold = [
        Entity(type="order_id", value="OD-1"),
        Entity(type="amount", value="Rs 500"),
    ]
    pred = [Entity(type="order_id", value="OD-1"), Entity(type="phone", value="98765")]
    score = entity_f1(pred, gold, "order OD-1 for Rs 500")
    # Precision 1/2, recall 1/2.
    assert abs(score - 0.5) < 1e-9
