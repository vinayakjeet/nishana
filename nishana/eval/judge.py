from __future__ import annotations

import hashlib

JUDGE_SYSTEM = """You are grading two assistant answers to the same customer
support ticket. Judge only which answer is better on:
1. correct intent (order status, refund, account, or other)
2. every order id, amount, date or product actually present in the ticket is
   extracted correctly, and nothing is invented
3. concise output a downstream system could parse
Reply with exactly one letter: A, B, or TIE."""

JUDGE_TEMPLATE = """Ticket:
{ticket}

Answer A:
{a}

Answer B:
{b}

One letter: A, B, or TIE."""


def order_for(item_id: str) -> str:
    """Stable per-item ordering derived from the id.

    Which candidate sits in slot A must not depend on which variant produced it,
    and must not change between runs. A hash of the item id gives both: it is
    deterministic, and it knows nothing about the candidates.
    """
    return "ab" if int(hashlib.sha256(item_id.encode()).hexdigest(), 16) % 2 == 0 else "ba"


def render_pair(
    item_id: str, ticket: str, first_candidate: str, second_candidate: str
) -> list[dict[str, str]]:
    """Build the judge messages for one ordered pass.

    On items whose stable order is 'ab', first_candidate sits in slot A;
    otherwise the two are swapped. Callers run every pair twice, passing the
    candidates in both argument orders, so neither output colonises one slot.
    """
    if order_for(item_id) == "ab":
        slot_a, slot_b = first_candidate, second_candidate
    else:
        slot_a, slot_b = second_candidate, first_candidate
    prompt = JUDGE_TEMPLATE.format(ticket=ticket, a=slot_a, b=slot_b)
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": prompt},
    ]


def parse_verdict(text: str) -> str:
    """Pull one of {A, B, TIE} out of a judge reply. Judges add punctuation and
    preamble no matter what the instructions say, so accept the first token that
    parses rather than failing the whole run over a full stop."""
    for token in text.upper().split():
        stripped = token.strip(".,:;*`'\"")
        if stripped in ("A", "B", "TIE"):
            return stripped
    return "INVALID"


def consistent_winner(first_verdict: str, swapped_verdict: str) -> str | None:
    """Collapse both orders into a position-stable result.

    Every comparison runs twice with the candidates swapped. An answer that wins
    in both orders is a real preference; split verdicts are the judge's position
    bias showing through, and they count for nobody. The swap-disagreement rate
    is reported separately rather than being averaged away.
    """
    first_canon = "A" if first_verdict == "A" else "B" if first_verdict == "B" else None
    swapped_canon = (
        "A" if swapped_verdict == "A" else "B" if swapped_verdict == "B" else None
    )
    if first_canon is None or swapped_canon is None:
        return None
    if first_canon == swapped_canon:
        return first_canon
    return None


def swap_disagreement_rate(verdicts: list[tuple[str, str]]) -> float:
    """Fraction of pairs where the two orders disagreed.

    Any pair where the two orders returned different letters counts, including
    a win in one order and a tie in the other: the judge contradicted itself
    either way. The rate is published beside every judged score rather than
    being averaged away.
    """
    scored = [(a, b) for a, b in verdicts if a != "INVALID" and b != "INVALID"]
    if not scored:
        return 0.0
    disagreements = sum(1 for a, b in scored if a != b)
    return round(disagreements / len(scored), 4)


def judged_quality(
    verdicts: list[tuple[str, str]], candidate_slot_of_variant: str
) -> float | None:
    """Share of consistently-decided pairs won by one variant's outputs."""
    decided = [
        consistent_winner(a, b)
        for a, b in verdicts
        if a != "INVALID" and b != "INVALID"
    ]
    decided = [d for d in decided if d is not None]
    if not decided:
        return None
    wins = sum(1 for d in decided if d == candidate_slot_of_variant)
    return round(wins / len(decided), 4)
