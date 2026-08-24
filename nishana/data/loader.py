from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from nishana.types import Ticket


class DatasetError(Exception):
    """A dataset file is malformed. Carries the file and line number, because a
    bare pydantic traceback on line 84 of a JSONL helps nobody."""


def load_jsonl(path: str | Path) -> list[Ticket]:
    out: list[Ticket] = []
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
            out.append(Ticket.model_validate(record))
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}:{number}: invalid JSON: {exc}") from exc
        except ValidationError as exc:
            raise DatasetError(f"{path}:{number}: {exc.errors()[0]['msg']}") from exc
    return out


def slice_counts(items: list[Ticket]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        for tag in item.slices:
            counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items()))
