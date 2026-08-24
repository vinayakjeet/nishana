from __future__ import annotations

from pydantic import BaseModel

DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct"


class TrainConfig(BaseModel):
    """One training configuration. Every ablation cell is exactly one of these,
    so a results row can name its cell without ambiguity.

    Defaults follow the Unsloth recipe for narrow classification/extraction
    tasks: rank 16 across all linear layers rather than attention only, learning
    rate 2e-4, two to three epochs. Rank appears in the ablation grid at 8 and
    64 to bracket the default from both sides.
    """

    variant_id: str
    base_model: str = DEFAULT_BASE_MODEL
    base_revision: str = "main"
    curation_tier: str
    dataset_size: int
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = "all-linear"
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 2
    grad_accum: int = 4
    max_seq_len: int = 512
    warmup_ratio: float = 0.03
    seed: int = 42
    scheduler: str = "cosine"

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum


def rank_grid(base: TrainConfig) -> list[TrainConfig]:
    """Rank ablation at fixed data: r in {8, 16, 64}, everything else held."""
    out = []
    for r in (8, 16, 64):
        out.append(base.model_copy(update={"lora_r": r, "lora_alpha": 32}))
    return out
