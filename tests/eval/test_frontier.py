from __future__ import annotations

import pytest

from nishana.eval.frontier import (
    DEFAULT_VERSION,
    FRONTIER_PROMPTS,
    PROMPT_LOG,
    build_messages,
    stamp,
)


def test_both_prompt_versions_exist_and_are_published() -> None:
    assert set(FRONTIER_PROMPTS) == {"v1", "v2"}
    # The iteration log documents why v2 exists; deleting it would hide the
    # one prompt-engineering pass the fairness rule demands.
    assert "v1 (2026-08-24)" in PROMPT_LOG
    assert "v2 (2026-08-24)" in PROMPT_LOG


def test_v2_carries_the_copy_exactly_rule() -> None:
    prompt = FRONTIER_PROMPTS["v2"]
    assert "copied exactly" in prompt or "Copy entity values character for character" in prompt
    assert "order_status" in prompt and "refund" in prompt


def test_build_messages_embeds_the_ticket() -> None:
    messages = build_messages(DEFAULT_VERSION, "mera order OD-1 kahan hai")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "mera order OD-1 kahan hai" in messages[0]["content"]


def test_unknown_version_refuses() -> None:
    with pytest.raises(KeyError):
        build_messages("v9", "ticket")


def test_stamp_records_model_and_date() -> None:
    stamped = stamp("llama-3.3-70b-versatile", "groq")
    assert stamped["model"] == "groq:llama-3.3-70b-versatile"
    assert stamped["generated_at"].startswith("20")
