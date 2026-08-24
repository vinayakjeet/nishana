from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

MANIFEST_NAME = "adapter-manifest.json"


class ResumeError(Exception):
    """A checkpoint cannot be resumed onto the requested configuration. Resuming
    a LoRA adapter onto different base weights silently produces garbage rather
    than an error, so every mismatch refuses loudly instead."""


class AdapterManifest(BaseModel):
    """Everything needed to prove a checkpoint resumes onto the right run.

    Optimizer and scheduler state are deliberately absent. The free-tier
    precedent this project follows resumed a three-epoch QLoRA across two
    different GPUs by carrying the adapter alone; full state did not transfer.
    The cost is a small loss discontinuity at each session boundary, which is
    measured and reported rather than hidden.
    """

    kind: str = "nishana-adapter"
    version: int = 1
    variant_id: str
    base_model: str
    base_revision: str
    dataset_hash: str
    curation_tier: str
    dataset_size: int
    lora_r: int
    step_start: int
    step_end: int
    epoch_start: float
    epoch_end: float
    session: int
    adapter_sha256: str = ""
    created_at: str = ""

    def model_post_init(self, __context) -> None:
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat(timespec="seconds")


def hash_adapter_dir(path: str | Path) -> str:
    """Order-stable digest over every file in an adapter directory.

    Safetensors shards sort by name, so hashing them sorted makes the digest a
    property of content rather than of write order.
    """
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"adapter directory not found: {root}")
    digest = hashlib.sha256()
    for file in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(file.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(file.read_bytes()).digest())
    return digest.hexdigest()


def save_manifest(checkpoint_dir: str | Path, manifest: AdapterManifest) -> Path:
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / MANIFEST_NAME
    payload = manifest.model_dump()
    if not payload["adapter_sha256"]:
        payload["adapter_sha256"] = hash_adapter_dir(root)
        manifest.adapter_sha256 = payload["adapter_sha256"]
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def load_manifest(checkpoint_dir: str | Path) -> AdapterManifest:
    path = Path(checkpoint_dir) / MANIFEST_NAME
    try:
        return AdapterManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResumeError(f"{path} does not exist; there is nothing to resume") from exc
    except json.JSONDecodeError as exc:
        raise ResumeError(f"{path} is corrupt: {exc}") from exc
    except ValidationError as exc:
        raise ResumeError(f"{path} is not a Nishana adapter manifest: {exc}") from exc


def check_resume(manifest: AdapterManifest, config, dataset_hash: str) -> None:
    """Refuse any resume whose base weights, data, or adapter geometry differ
    from the original run."""
    problems = []
    if manifest.base_model != config.base_model:
        problems.append(
            f"base model {manifest.base_model!r} != configured {config.base_model!r}"
        )
    if manifest.base_revision != config.base_revision:
        problems.append(
            f"base revision {manifest.base_revision!r} != configured {config.base_revision!r}"
        )
    if manifest.dataset_hash != dataset_hash:
        problems.append(
            f"dataset hash {manifest.dataset_hash[:24]}... != current {dataset_hash[:24]}..."
        )
    if manifest.lora_r != config.lora_r:
        problems.append(f"rank {manifest.lora_r} != configured {config.lora_r}")
    if manifest.variant_id != config.variant_id:
        problems.append(
            f"variant {manifest.variant_id!r} != configured {config.variant_id!r}"
        )
    if problems:
        raise ResumeError("cannot resume: " + "; ".join(problems))
