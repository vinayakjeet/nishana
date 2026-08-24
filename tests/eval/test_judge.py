from __future__ import annotations

from nishana.eval.judge import (
    JUDGE_SYSTEM,
    consistent_winner,
    judged_quality,
    order_for,
    parse_verdict,
    render_pair,
    swap_disagreement_rate,
)


def test_order_is_deterministic_and_candidate_blind() -> None:
    assert order_for("tkt-001") == order_for("tkt-001")
    assert order_for("tkt-001") in ("ab", "ba")


def test_both_orders_occur_across_the_corpus() -> None:
    ids = [f"tkt-{i:03d}" for i in range(1, 121)]
    orders = {order_for(i) for i in ids}
    assert orders == {"ab", "ba"}


def test_rendered_prompt_carries_no_variant_identity() -> None:
    messages = render_pair(
        "tkt-007",
        "ticket text",
        '{"intent": "order_status", "entities": []}',
        '{"intent":"order_status","entities":[{"type":"order_id","value":"OD-1"}]}',
    )
    blob = "\n".join(m["content"] for m in messages)
    for forbidden in (
        "synthetic-only",
        "verified-heavy",
        "mixed",
        "frontier",
        "adapter",
        "qwen",
        "groq",
        "llama",
        "fine-tune",
    ):
        assert forbidden not in blob.lower(), forbidden


def test_the_system_rubric_is_part_of_every_prompt() -> None:
    messages = render_pair("tkt-008", "t", "a", "b")
    assert messages[0]["content"] == JUDGE_SYSTEM
    assert "Ticket:" in messages[1]["content"]


def test_parse_verdict_tolerates_preamble() -> None:
    assert parse_verdict("**A**") == "A"
    assert parse_verdict("The better answer is B.") == "B"
    assert parse_verdict("TIE") == "TIE"
    assert parse_verdict("cannot say") == "INVALID"


def test_consistent_winner_requires_both_orders() -> None:
    assert consistent_winner("A", "A") == "A"
    assert consistent_winner("B", "B") == "B"
    assert consistent_winner("A", "B") is None
    assert consistent_winner("TIE", "A") is None
    assert consistent_winner("INVALID", "A") is None


def test_swap_disagreement_rate_counts_split_pairs() -> None:
    verdicts = [("A", "A"), ("A", "B"), ("TIE", "TIE")]
    # A double tie is consistent, a split pair is not: 1 of 3.
    assert swap_disagreement_rate(verdicts) == round(1 / 3, 4)
    assert swap_disagreement_rate([("INVALID", "A")]) == 0.0


def test_judged_quality_uses_only_consistent_wins() -> None:
    verdicts = [("A", "A"), ("A", "B"), ("B", "B")]
    assert judged_quality(verdicts, candidate_slot_of_variant="A") == round(1 / 2, 4)
    assert judged_quality(verdicts, candidate_slot_of_variant="B") == round(1 / 2, 4)
