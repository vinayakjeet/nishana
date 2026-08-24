from __future__ import annotations

import math

from pydantic import BaseModel


def total_steps(rows: int, epochs: int, effective_batch: int) -> int:
    if rows <= 0:
        raise ValueError(f"rows must be positive, got {rows}")
    if effective_batch <= 0:
        raise ValueError(f"effective batch must be positive, got {effective_batch}")
    return math.ceil(rows * epochs / effective_batch)


class SessionPlan(BaseModel):
    """How one training run splits across Kaggle's wall-clock-limited GPU
    sessions. Sessions end wherever the clock cuts, not at an epoch boundary,
    which is precisely why checkpointing is adapter-only: the thing being saved
    has to be small, portable, and loadable onto a different GPU next session."""

    steps_total: int
    sessions: list[tuple[int, int]]

    @property
    def n_sessions(self) -> int:
        return len(self.sessions)

    def session_of(self, step: int) -> int:
        for number, (start, end) in enumerate(self.sessions, start=1):
            if start < step <= end:
                return number
        return self.n_sessions


def session_plan_is_complete(plan: SessionPlan) -> bool:
    """True when the sessions tile every step exactly once, which the resume
    logic relies on: a gap would mean silently skipped training."""
    covered = 0
    for start, end in plan.sessions:
        if start != covered:
            return False
        covered = end
    return covered == plan.steps_total


def plan_sessions(
    steps_total: int, minutes_per_session: float, seconds_per_step: float
) -> SessionPlan:
    """Split a run into sessions sized by measured throughput, not by hope.

    The seconds-per-step figure comes from the first session's own log; planning
    the second session on the first's measurement is the whole point. A 20
    minute safety margin absorbs dataset download and model warmup, which on
    Kaggle are neither free nor predictable.
    """
    if minutes_per_session <= 0 or seconds_per_step <= 0:
        raise ValueError("minutes_per_session and seconds_per_step must be positive")

    usable_seconds = max(minutes_per_session - 20, 0) * 60
    budget = (
        1
        if usable_seconds == 0
        else max(int(usable_seconds / seconds_per_step), 1)
    )

    sessions: list[tuple[int, int]] = []
    done = 0
    while done < steps_total:
        end = min(done + budget, steps_total)
        sessions.append((done, end))
        done = end
    return SessionPlan(steps_total=steps_total, sessions=sessions)
