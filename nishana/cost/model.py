from __future__ import annotations

import math
from pathlib import Path

import yaml

DEFAULT_PRICING = Path("deploy/pricing.yaml")


def load_pricing(path: str | Path = DEFAULT_PRICING) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def training_cost(
    rows: int, epochs: int, effective_batch: int, seconds_per_step: float
) -> dict[str, float]:
    """GPU-hours and rupees for one training run.

    Kaggle is free but quota-metered, so the honest unit is GPU-hours against
    the weekly 30-hour budget: a run can cost nothing in currency and still be
    expensive in wall-clock and quota. Seconds-per-step arrives measured from
    the first session log, never assumed.
    """
    steps = math.ceil(rows * epochs / effective_batch)
    gpu_hours = round(steps * seconds_per_step / 3600, 2)
    return {"steps": steps, "gpu_hours": gpu_hours, "usd": 0.0}


def serving_cost_per_1k(
    pricing: dict,
    per_request_ms: float | None = None,
    idle_gap_s: float | None = None,
    gpu: str = "t4",
) -> dict[str, float]:
    """Cost of one thousand inferences on Modal at scale-to-zero.

    Two components: active compute while serving, and cold starts paid when the
    container has idled past its keep-alive. The cold-start rate is an explicit
    assumption rather than a hidden constant, because it is the number most
    likely to differ between someone's demo traffic and production.
    """
    serving = pricing["serving"]
    tier = pricing["modal"][gpu]
    cold_start_s = pricing["modal"]["cold_start_s"]
    ms = per_request_ms if per_request_ms is not None else serving["per_request_ms"]
    gap = idle_gap_s if idle_gap_s is not None else serving["assumed_idle_gap_s"]

    active_usd = 1000 * (ms / 1000) * tier["per_second_usd"]
    # Every request is assumed to arrive after a full idle gap (sparse demo
    # traffic, the worst case for scale-to-zero), and each accumulated idle
    # hour produces one cold start.
    idle_hours = 1000 * max(gap - ms / 1000, 0) / 3600
    cold_usd = idle_hours * cold_start_s * tier["per_second_usd"]
    return {
        "active_usd": round(active_usd, 6),
        "cold_start_usd": round(cold_usd, 6),
        "total_per_1k_usd": round(active_usd + cold_usd, 6),
    }


def frontier_cost_per_1k(
    pricing: dict, tokens_in: int = 450, tokens_out: int = 90, model: str = "gemini-flash-latest"
) -> float:
    """Frontier API list price for the same thousand inferences."""
    entry = pricing["frontier_api"][model]
    cost = (tokens_in / 1e6 * entry["input_per_1m_usd"]) + (
        tokens_out / 1e6 * entry["output_per_1m_usd"]
    )
    return round(cost * 1000, 6)


def breakeven_volume(training_usd: float, selfhost_per_1k: float, frontier_per_1k: float) -> dict:
    """Request volume where owning the adapter beats renting the frontier.

    With scale-to-zero there is no standing monthly bill to amortize; the only
    fixed cost is training itself. If serving were somehow dearer than the API
    the volume would come out negative, which is reported as never rather than
    being swept under a positive-looking row.
    """
    savings = frontier_per_1k - selfhost_per_1k
    savings = round(savings, 6)
    if savings <= 0:
        return {"savings_per_1k_usd": round(savings, 6), "breakeven_requests": None}
    # The epsilon keeps float drift from turning an exact break-even into 25,001.
    # Savings are per thousand requests; the answer is a request count.
    requests = (training_usd - 1e-9) / savings * 1000
    return {
        "savings_per_1k_usd": round(savings, 6),
        "breakeven_requests": math.ceil(requests),
    }
