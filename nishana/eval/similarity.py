from __future__ import annotations

from nishana.data.decontam import tokenize


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for i in a:
        current = [0]
        for j, bj in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if i == bj else max(current[-1], previous[j]))
        previous = current
    return previous[-1]


def rouge_l(candidate: str, reference: str) -> float:
    """LCS F-measure between a prediction and its reference rendering.

    This is the distribution-similarity column. It rewards output that matches
    the phrasing the training data teaches, and it is exactly the kind of metric
    the honesty reference warns can rise while blind quality falls, which is why
    no number here means anything without the judged-quality column beside it.
    """
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(cand_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def render_reference(intent: str, entities: list[dict]) -> str:
    """Canonical gold rendering every candidate text is scored against.

    A fixed renderer keeps similarity comparable across variants: two outputs
    are never favoured for sharing formatting accidents with one prompt and not
    another.
    """
    parts = [f"intent={intent}"]
    for entity in sorted(entities, key=lambda e: (e.get("type", ""), e.get("value", ""))):
        parts.append(f"{entity['type']}={entity['value']}")
    return " ".join(parts)


def bertscore_similarity(candidate: str, reference: str) -> float | None:
    """Optional model-based similarity, used only where the package exists.

    Returns None rather than raising when bert-score is absent, so callers print
    an honest gap in the table instead of silently substituting the lexical
    score under the same column name.
    """
    try:
        from bert_score import score  # noqa: PLC0415
    except ImportError:
        return None
    padded_candidate = [candidate] if isinstance(candidate, str) else candidate
    result = score(
        padded_candidate,
        [reference] * len(padded_candidate),
        lang="en",
        verbose=False,
    )
    return float(result[1].mean())
