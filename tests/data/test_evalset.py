from __future__ import annotations

from pathlib import Path

from nishana.data.hashing import content_hash, read_hash
from nishana.data.loader import load_jsonl, slice_counts

EVAL = Path("datasets/tickets-eval.jsonl")
MANIFEST = Path("datasets/manifest.yaml")


def test_eval_set_is_the_published_size() -> None:
    items = load_jsonl(EVAL)
    assert len(items) == 120


def test_every_slice_has_at_least_ten_items() -> None:
    counts = slice_counts(load_jsonl(EVAL))
    small = {tag: n for tag, n in counts.items() if n < 10}
    assert not small, small


def test_intents_are_balanced_at_thirty_each() -> None:
    counts = slice_counts(load_jsonl(EVAL))
    intents = ("order_status", "refund", "account", "other")
    assert [counts[f"intent:{i}"] for i in intents] == [30, 30, 30, 30]


def test_language_slices_cover_all_three_registers() -> None:
    counts = slice_counts(load_jsonl(EVAL))
    assert counts["lang:hinglish"] >= 50
    assert counts["lang:en"] >= 25
    assert counts["lang:hi"] >= 15


def test_entity_values_appear_verbatim_in_their_tickets() -> None:
    for ticket in load_jsonl(EVAL):
        text = ticket.input["prompt"]
        for entity in ticket.expected.entities:
            assert entity.value in text, f"{ticket.id}: {entity.value!r} not in text"



def test_manifest_hash_matches_current_content() -> None:
    """If this fails, someone edited the eval set without re-versioning it.
    Regenerate with scripts/build_manifest logic or rerun the manifest step."""
    items = load_jsonl(EVAL)
    current = content_hash([t.model_dump() for t in items])
    assert read_hash(MANIFEST) == current
