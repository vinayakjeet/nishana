from __future__ import annotations

import pytest

from nishana.cost.model import (
    breakeven_volume,
    frontier_cost_per_1k,
    load_pricing,
    serving_cost_per_1k,
    training_cost,
)


@pytest.fixture
def pricing() -> dict:
    return load_pricing("deploy/pricing.yaml")


def test_training_cost_is_quota_not_currency(pricing) -> None:
    cost = training_cost(rows=1000, epochs=3, effective_batch=8, seconds_per_step=2.0)
    assert cost["steps"] == 375
    assert abs(cost["gpu_hours"] - 0.21) < 0.01
    assert cost["usd"] == 0.0


def test_serving_cost_has_active_and_cold_components(pricing) -> None:
    result = serving_cost_per_1k(pricing, per_request_ms=100, idle_gap_s=3600)
    tier = pricing["modal"]["t4"]
    cold_start_s = pricing["modal"]["cold_start_s"]
    expected_active = 1000 * 0.1 * tier["per_second_usd"]
    assert result["active_usd"] == round(expected_active, 6)
    # Each of the 1,000 requests idles just under an hour; accumulated idle
    # hours each produce one cold start.
    expected_cold = 1000 * (3600 - 0.1) / 3600 * cold_start_s * tier["per_second_usd"]
    assert result["cold_start_usd"] == round(expected_cold, 6)
    assert result["total_per_1k_usd"] == round(result["active_usd"] + result["cold_start_usd"], 6)


def test_frontier_cost_uses_both_token_directions(pricing) -> None:
    entry = pricing["frontier_api"]["gemini-flash-latest"]
    cost = frontier_cost_per_1k(pricing, tokens_in=1_000_000, tokens_out=1_000_000)
    assert cost == round(1000 * (entry["input_per_1m_usd"] + entry["output_per_1m_usd"]), 6)


def test_breakeven_recovers_training_from_savings() -> None:
    result = breakeven_volume(training_usd=10.0, selfhost_per_1k=0.02, frontier_per_1k=0.42)
    assert result["savings_per_1k_usd"] == 0.40
    assert result["breakeven_requests"] == 25_000


def test_breakeven_reports_never_when_selfhosting_is_dearer() -> None:
    result = breakeven_volume(training_usd=10.0, selfhost_per_1k=0.50, frontier_per_1k=0.42)
    assert result["breakeven_requests"] is None
