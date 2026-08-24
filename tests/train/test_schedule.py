from __future__ import annotations

import pytest

from nishana.train.config import TrainConfig, rank_grid
from nishana.train.schedule import plan_sessions, session_plan_is_complete, total_steps


def test_total_steps_rounds_up() -> None:
    assert total_steps(500, 3, 8) == 188
    with pytest.raises(ValueError):
        total_steps(0, 3, 8)
    with pytest.raises(ValueError):
        total_steps(100, 1, 0)


def test_plan_splits_by_measured_throughput() -> None:
    # 188 steps at 2s/step is 376s; a 330 minute cap fits it in one session.
    plan = plan_sessions(188, minutes_per_session=330, seconds_per_step=2.0)
    assert plan.n_sessions == 1
    assert plan.sessions[0] == (0, 188)


def test_plan_cuts_before_the_wall_does() -> None:
    # A 9-hour run against a 60 minute cap: the 20 minute safety margin keeps
    # each session under the limit that actually kills Kaggle jobs.
    plan = plan_sessions(3600, minutes_per_session=60, seconds_per_step=1.0)
    for start, end in plan.sessions[:-1]:
        assert end - start == (60 - 20) * 60
    assert session_plan_is_complete(plan)


def test_session_of_locates_a_step() -> None:
    plan = plan_sessions(300, minutes_per_session=10, seconds_per_step=2.0)
    first_end = plan.sessions[0][1]
    assert plan.session_of(first_end) == 1
    assert plan.session_of(first_end + 1) == 2
    assert plan.session_of(300) == plan.n_sessions


def test_rank_grid_brackets_the_default() -> None:
    base = TrainConfig(variant_id="v", curation_tier="synthetic-only", dataset_size=500)
    grid = rank_grid(base)
    assert [c.lora_r for c in grid] == [8, 16, 64]
    assert all(c.dataset_size == 500 for c in grid)
