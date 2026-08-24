from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from llm.client import ChatClient
from llm.types import ChatMessage
from nishana.data.loader import load_jsonl
from nishana.eval import frontier
from nishana.types import Entity, Prediction

ADOPTING_TIMEOUT_S = 60.0


def _stamp(model: str, provider: str) -> tuple[str, str]:
    stamped = frontier.stamp(model, provider)
    return stamped["model"], stamped["generated_at"]


async def run_frontier(
    items,
    provider: str,
    model: str,
    prompt_version: str = frontier.DEFAULT_VERSION,
    rpm: float = 25.0,
) -> list[Prediction]:
    """Score the eval set against a frontier API through the chassis client.

    Throttled well under every free tier's documented limit, because the
    baseline has to finish for its numbers to exist at all.
    """
    client = ChatClient()
    out: list[Prediction] = []
    gap = 60.0 / rpm if rpm > 0 else 0.0
    for item in items:
        prompt = item.input["prompt"]
        messages = [ChatMessage(**m) for m in frontier.build_messages(prompt_version, prompt)]
        response = await client.complete(provider, messages)
        model_name, when = _stamp(response.model, provider)
        out.append(
            Prediction(
                item_id=item.id,
                prediction=response.text,
                entities=[],
                model=model_name,
                generated_at=when,
                tokens_in=response.tokens_in,
                tokens_out=response.tokens_out,
                latency_ms=response.latency_ms,
            )
        )
        if gap:
            await asyncio.sleep(gap)
    return out


async def run_target(items, url: str) -> list[Prediction]:
    """Score a served adapter over HTTP using ShipGate's adopting contract.

    One POST per item, output plus meta back. Speaking the same contract means
    the same harness scores the frontier baseline, a local LoRA server, and the
    Modal deployment without branching anywhere.
    """
    stamp_model, when = "", ""
    out: list[Prediction] = []
    async with httpx.AsyncClient(timeout=ADOPTING_TIMEOUT_S) as http:
        for item in items:
            try:
                response = await http.post(
                    url, json={"input": {"prompt": item.input["prompt"]}, "item_id": item.id}
                )
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                out.append(
                    Prediction(
                        item_id=item.id,
                        prediction="",
                        model="unknown",
                        generated_at=when,
                        error=str(exc),
                    )
                )
                continue
            meta = payload.get("meta") or {}
            model_name = str(meta.get("model", "unknown"))
            if not stamp_model:
                stamp_model = model_name
                when = frontier.stamp(model_name, "target")["generated_at"]
            raw_entities = payload.get("entities", [])
            out.append(
                Prediction(
                    item_id=item.id,
                    prediction=str(payload.get("output", "")),
                    entities=[
                        Entity.model_validate(e)
                        for e in raw_entities
                        if isinstance(e, dict)
                    ],
                    model=model_name,
                    generated_at=when,
                    tokens_in=meta.get("tokens_in"),
                    tokens_out=meta.get("tokens_out"),
                    latency_ms=meta.get("latency_ms"),
                )
            )
    return out


def run_replay(items, replay_path: str | Path, label: str = "replay") -> list[Prediction]:
    """Deterministic outputs from a recorded file, keyed by item id.

    This exists so the scoring pipeline can be exercised end to end in tests
    and CI with no network, no GPU and no keys. Replay rows carry their own
    label so they can never be mistaken for measured results.
    """
    recorded = {
        row["item_id"]: row
        for row in json.loads(Path(replay_path).read_text(encoding="utf-8"))
    }
    out: list[Prediction] = []
    for item in items:
        row = recorded.get(item.id)
        if row is None:
            raise KeyError(f"replay file has no row for {item.id}")
        out.append(
            Prediction(
                item_id=item.id,
                prediction=row["prediction"],
                entities=[Entity.model_validate(e) for e in row.get("entities", [])],
                model=row.get("model", label),
                generated_at=frontier.stamp(row.get("model", label), "replay")["generated_at"],
            )
        )
    return out


def write_predictions(path: str | Path, rows: list[Prediction]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(r.model_dump_json() for r in rows) + "\n", encoding="utf-8"
    )


def load_items(dataset_path: str | Path):
    return load_jsonl(dataset_path)
