from __future__ import annotations

import pytest

from nishana.data.curation import (
    ABLATION_SIZES,
    SeedRow,
    TrainingRow,
    build_variant,
    variant_id,
)


def _seed(id: str, curated: bool = True) -> SeedRow:
    return SeedRow(id=id, text=f"seed text {id}", intent="refund", curated=curated)


def _gen(id: str) -> TrainingRow:
    return TrainingRow(
        id=id, text=f"generated text {id}", intent="refund", source="generated"
    )


def test_synthetic_only_takes_the_whole_pool() -> None:
    pool = [_gen(f"g{i:03d}") for i in range(10)]
    rows = build_variant("synthetic-only", [], pool, set(), 250)
    assert len(rows) == 10
    assert {r.id for r in rows} == {f"g{i:03d}" for i in range(10)}


def test_mixed_is_verified_plus_an_equal_number_unverified() -> None:
    pool = [_gen(f"g{i:03d}") for i in range(20)]
    verified = {f"g{i:03d}" for i in range(5)}
    rows = build_variant("mixed", [], pool, verified, 500)
    marked = [r for r in rows if r.verified]
    unmarked = [r for r in rows if not r.verified]
    assert len(marked) == 5
    assert len(unmarked) == 5


def test_verified_heavy_is_curated_seeds_plus_marked_generated() -> None:
    seeds = [_seed(f"s{i:03d}") for i in range(6)] + [_seed("s-un", curated=False)]
    pool = [_gen(f"g{i:03d}") for i in range(10)]
    verified = {f"g{i:03d}" for i in range(4)}
    rows = build_variant("verified-heavy", seeds, pool, verified, 250)
    assert all(r.verified for r in rows)
    seed_rows = [r for r in rows if r.source == "seed"]
    assert len(seed_rows) == 6
    assert len(rows) == 10


def test_output_is_deterministic_regardless_of_input_order() -> None:
    pool = [_gen(f"g{i:03d}") for i in range(30)]
    forward = build_variant("synthetic-only", [], list(reversed(pool)), set(), 500)
    assert [r.id for r in forward] == [f"g{i:03d}" for i in range(30)]


def test_size_caps_the_pool_and_validates() -> None:
    pool = [_gen(f"g{i:03d}") for i in range(300)]
    rows = build_variant("synthetic-only", [], pool, set(), 250)
    assert len(rows) == 250
    with pytest.raises(ValueError):
        build_variant("synthetic-only", [], pool, set(), 123)


def test_unknown_tier_refuses() -> None:
    with pytest.raises(ValueError, match="unknown curation tier"):
        build_variant("vibes", [], [], set(), 250)


def test_variant_id_stable_and_informative() -> None:
    a = variant_id("synthetic-only", 500, "unsloth/Qwen2.5-7B-Instruct", 16)
    b = variant_id("synthetic-only", 500, "unsloth/Qwen2.5-7B-Instruct", 16)
    c = variant_id("synthetic-only", 500, "unsloth/Qwen2.5-7B-Instruct", 64)
    assert a == b
    assert a != c
    assert "synthetic-only-500" in a and "-r16-" in a


def test_ablation_sizes_are_the_published_grid() -> None:
    assert ABLATION_SIZES == (250, 500, 1000, 2000)
