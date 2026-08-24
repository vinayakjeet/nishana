from __future__ import annotations

from pathlib import Path

from nishana.data.decontam import check_overlap, decontaminate, ngrams, tokenize


def _eval(id: str, text: str) -> dict:
    return {"id": id, "input": {"prompt": text}}


def test_tokenize_keeps_devanagari_whole() -> None:
    tokens = tokenize("कृपया ऑर्डर OD-12345 बताइए")
    assert tokens == ["कृपया", "ऑर्डर", "od", "12345", "बताइए"]
    # Latin-script Hinglish shares the same tokenizer, so overlap comparisons
    # never depend on which script a phrase arrived in.
    assert tokenize("Order OD-1 please") == ["order", "od", "1", "please"]


def test_ngrams_count_is_exact() -> None:
    assert len(ngrams(["a", "b", "c", "d", "e"], 4)) == 2
    assert ngrams(["a"], 5) == set()


def test_overlap_finds_copied_text() -> None:
    seeds = ["please refund my money for order OD-99999 immediately"]
    items = [
        _eval("clean", "where is my order today"),
        _eval("dirty", "I want you to please refund my money for order OD-12345"),
    ]
    report = check_overlap(seeds, items, 5)
    assert report.items_with_overlap == 1
    assert list(report.offenders) == ["dirty"]


def test_overlap_percentage_published_even_when_zero() -> None:
    report = check_overlap(["seed text here"], [_eval("a", "totally different words")], 8)
    assert report.overlap_pct == 0.0
    assert report.shared_ngrams == 0


def test_decontaminate_drops_and_reports(tmp_path: Path) -> None:
    seeds = ["the quick brown fox jumps over the lazy dog today"]
    kept_items = [
        _eval("keep-me", "unrelated ticket about a missing parcel"),
        _eval("drop-me", "the quick brown fox jumps over my fence nightly"),
    ]
    kept, removed = decontaminate(seeds, kept_items)
    assert removed == ["drop-me"]
    assert [item["id"] for item in kept] == ["keep-me"]


def test_decontaminate_checks_every_requested_length() -> None:
    # Shares an 8-gram but no 5-gram overlap beyond it; either way it goes.
    shared = "one two three four five six seven eight"
    seeds = [shared + " nine ten"]
    items = [_eval("x", "prefix words " + shared + " suffix")]
    _, removed = decontaminate(seeds, items, drop_on=(5, 8))
    assert removed == ["x"]
