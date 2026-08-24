from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field


class TrainingRow(BaseModel):
    """One supervised example as the trainer consumes it."""

    id: str
    text: str
    intent: str
    entities: list[dict] = Field(default_factory=list)
    source: str  # "seed" or "generated"
    verified: bool = False
    meta: dict[str, object] = Field(default_factory=dict)


class SeedRow(BaseModel):
    id: str
    text: str
    intent: str
    entities: list[dict] = Field(default_factory=list)
    curated: bool = False


TIERS = ("synthetic-only", "mixed", "verified-heavy")

# The published sweet spot for narrow tasks sits between these; the size ablation
# exists to show whether that holds on this task rather than to assume it.
ABLATION_SIZES = (250, 500, 1000, 2000)


def _stable_order(rows: list[TrainingRow]) -> list[TrainingRow]:
    return sorted(rows, key=lambda r: r.id)


def _take(rows: list[TrainingRow], size: int) -> list[TrainingRow]:
    """Deterministic prefix of a canonically ordered pool. Rebuilding a variant
    must produce a byte-identical file, because a content hash that moves between
    runs makes every downstream comparison meaningless."""
    return _stable_order(rows)[:size]


def build_variant(
    tier: str,
    seeds: list[SeedRow],
    generated: list[TrainingRow],
    verified_ids: set[str],
    size: int,
) -> list[TrainingRow]:
    """Assemble one data-curation variant.

    The three tiers carry the ablation's hypothesis: if fine-tuning on
    unverified synthetic data mostly teaches distribution match, the
    synthetic-only variant should score well on similarity while blind judges
    prefer the variants a human passed over. Naming the mechanism before
    running is what turns the result into a test rather than a story.
    """
    if tier not in TIERS:
        raise ValueError(f"unknown curation tier '{tier}', expected one of {TIERS}")
    if size not in ABLATION_SIZES:
        raise ValueError(f"size must be one of {ABLATION_SIZES}, got {size}")

    marked = [
        row.model_copy(update={"verified": True})
        for row in generated
        if row.id in verified_ids
    ]
    unmarked = [row for row in generated if row.id not in verified_ids]

    if tier == "synthetic-only":
        pool = generated
    elif tier == "mixed":
        # All verified rows plus an equal number of unverified ones, so neither
        # signal can dominate the mix.
        pool = marked + _stable_order(unmarked)[: len(marked)]
    else:
        curated_seeds = [
            TrainingRow(
                id=s.id,
                text=s.text,
                intent=s.intent,
                entities=s.entities,
                source="seed",
                verified=True,
            )
            for s in seeds
            if s.curated
        ]
        pool = curated_seeds + marked

    return _take(pool, size)


def variant_id(tier: str, size: int, base_model: str, rank: int) -> str:
    """Stable identifier for a run directory and results filename."""
    slug = base_model.split("/")[-1]
    raw = f"{tier}|{size}|{slug}|r{rank}"
    short = hashlib.sha256(raw.encode()).hexdigest()[:8]
    return f"{tier}-{size}-{slug}-r{rank}-{short}"
