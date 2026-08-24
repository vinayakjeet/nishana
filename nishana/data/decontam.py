from __future__ import annotations

import re
from dataclasses import dataclass, field

# Word characters plus the Devanagari combining-vowel-sign range, so Hindi
# words stay whole instead of shattering at every matra. Splitting there would
# still be fair (both corpora split identically), but whole words make the
# published n-grams readable.
WORD = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Unicode-aware word tokens, lowercased.

    The token pattern keeps Devanagari and other Indic scripts intact, which
    matters when half the corpus is Hindi.
    """
    return WORD.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


@dataclass
class OverlapReport:
    """Measured contamination between the synthetic-generation seed set and the
    eval set. The numbers get published whatever they turn out to be: claiming
    "we ran decontamination" without printing them is how inflated scores pass
    for clean ones."""

    n: int
    shared_ngrams: int
    items_with_overlap: int
    n_eval_items: int
    offenders: dict[str, int] = field(default_factory=dict)

    @property
    def overlap_pct(self) -> float:
        if self.n_eval_items == 0:
            return 0.0
        return round(100.0 * self.items_with_overlap / self.n_eval_items, 2)


def check_overlap(
    seed_texts: list[str], eval_items: list[dict], n: int
) -> OverlapReport:
    """Every n-gram shared between any seed text and an eval item is counted,
    per item, so the worst offenders can be dropped rather than waved through."""
    seed_grams: set[tuple[str, ...]] = set()
    for text in seed_texts:
        seed_grams |= ngrams(tokenize(text), n)

    offenders: dict[str, int] = {}
    hits = 0
    for item in eval_items:
        text = item["input"]["prompt"] if "input" in item else item.get("text", "")
        shared = ngrams(tokenize(text), n) & seed_grams
        if shared:
            hits += 1
            offenders[item["id"]] = len(shared)

    return OverlapReport(
        n=n,
        shared_ngrams=sum(offenders.values()),
        items_with_overlap=hits,
        n_eval_items=len(eval_items),
        offenders=dict(sorted(offenders.items(), key=lambda kv: -kv[1])),
    )


def decontaminate(
    seed_texts: list[str], eval_items: list[dict], drop_on: tuple[int, ...] = (5, 8)
) -> tuple[list[dict], list[str]]:
    """Drop every eval item that shares an n-gram with a seed at any checked
    length. Returning the removed ids matters: the manifest records what was
    taken out, so the published number covers the survivors by construction.

    A long shared 5-gram between two support tickets can be boilerplate ("kindly
    refund my amount as soon as possible") rather than leakage. The honest move
    is still removal plus a note, not a judgement call per pair made after
    looking at the scores.
    """
    banned: set[tuple[str, ...]] = set()
    for n in drop_on:
        for text in seed_texts:
            banned |= ngrams(tokenize(text), n)

    kept, removed = [], []
    for item in eval_items:
        text = item["input"]["prompt"] if "input" in item else item.get("text", "")
        tokens = tokenize(text)
        if any(ngrams(tokens, n) & banned for n in drop_on):
            removed.append(item["id"])
        else:
            kept.append(item)
    return kept, removed
