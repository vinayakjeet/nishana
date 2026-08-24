from __future__ import annotations

from nishana.eval.similarity import bertscore_similarity, render_reference, rouge_l


def test_rouge_l_is_one_for_identical_text() -> None:
    assert rouge_l("intent=refund order_id=OD-1", "intent=refund order_id=OD-1") == 1.0


def test_rouge_l_is_zero_for_disjoint_text() -> None:
    assert rouge_l("alpha beta gamma", "delta epsilon zeta") == 0.0


def test_rouge_l_partial_overlap_lands_between() -> None:
    score = rouge_l("the refund was processed today", "the refund processed today quickly")
    assert 0.0 < score < 1.0


def test_render_reference_is_canonical() -> None:
    reference = render_reference(
        "order_status",
        [
            {"type": "amount", "value": "Rs 500"},
            {"type": "order_id", "value": "OD-9"},
        ],
    )
    # Entities sorted by type, so the same facts always render identically.
    assert reference == "intent=order_status amount=Rs 500 order_id=OD-9"


def test_bertscore_returns_none_when_package_missing(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fail_bert(name, *args, **kwargs):
        if name == "bert_score":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_bert)
    assert bertscore_similarity("a", "b") is None
