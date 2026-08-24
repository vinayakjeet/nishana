from __future__ import annotations

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """One annotated span inside a ticket. `value` must appear verbatim in the
    text, because the extraction F1 checks substring membership before type."""

    type: str
    value: str


class Ticket(BaseModel):
    """One eval item. The shape follows ShipGate's adopting contract: `input`
    carries the prompt a target answers, `expected` is structured so the exact
    runner can score intent and entities without a model in the loop."""

    id: str
    input: dict[str, str]
    expected: IntentLabel
    slices: list[str] = Field(default_factory=list)
    meta: dict[str, object] = Field(default_factory=dict)


class IntentLabel(BaseModel):
    intent: str
    entities: list[Entity] = Field(default_factory=list)


class Prediction(BaseModel):
    """What any candidate (frontier API, fine-tuned adapter, mock) produced for
    one ticket. `model` and `generated_at` travel with every row so no number
    exists anywhere without its model name and date."""

    item_id: str
    prediction: str
    entities: list[Entity] = Field(default_factory=list)
    model: str
    generated_at: str
    error: str = ""
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: float | None = None


class VariantScore(BaseModel):
    """The dual score the brief demands for every ablation variant: a similarity
    metric that rewards matching the training distribution, and a blind judged
    quality score that does not have to agree with it. The gap between the two
    columns is the finding the whole project is built to expose."""

    variant: str
    model: str
    model_date: str
    n_items: int
    intent_accuracy: float
    entity_f1: float
    similarity: float
    judged_quality: float | None = None
    judge_agreement: float | None = None
