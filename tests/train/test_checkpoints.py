from __future__ import annotations

import json
from pathlib import Path

import pytest

from nishana.train.checkpoints import (
    AdapterManifest,
    ResumeError,
    check_resume,
    hash_adapter_dir,
    load_manifest,
    save_manifest,
)
from nishana.train.config import TrainConfig


def _manifest(**overrides) -> AdapterManifest:
    base = dict(
        variant_id="synthetic-only-500-Qwen2.5-7B-Instruct-r16-ab12cd34",
        base_model="unsloth/Qwen2.5-7B-Instruct",
        base_revision="main",
        dataset_hash="sha256:" + "a" * 64,
        curation_tier="synthetic-only",
        dataset_size=500,
        lora_r=16,
        step_start=1,
        step_end=94,
        epoch_start=0.0,
        epoch_end=1.5,
        session=1,
    )
    return AdapterManifest(**(base | overrides))


def _config() -> TrainConfig:
    return TrainConfig(
        variant_id="synthetic-only-500-Qwen2.5-7B-Instruct-r16-ab12cd34",
        curation_tier="synthetic-only",
        dataset_size=500,
    )


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = save_manifest(tmp_path, _manifest())
    assert manifest.name == "adapter-manifest.json"
    loaded = load_manifest(tmp_path)
    assert loaded.variant_id == _manifest().variant_id
    assert loaded.created_at != ""


def test_save_hashes_the_adapter_directory(tmp_path: Path) -> None:
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    (tmp_path / "adapter_config.json").write_bytes(b"{}")
    expected = hash_adapter_dir(tmp_path)
    manifest = _manifest()
    save_manifest(tmp_path, manifest)
    assert manifest.adapter_sha256 == expected
    stored = json.loads((tmp_path / "adapter-manifest.json").read_text(encoding="utf-8"))
    assert stored["adapter_sha256"] == expected


def test_hash_is_order_stable_and_content_sensitive(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"one")
    (tmp_path / "b.bin").write_bytes(b"two")
    first = hash_adapter_dir(tmp_path)

    (tmp_path / "b.bin").unlink()
    (tmp_path / "b.bin").write_bytes(b"two")
    assert hash_adapter_dir(tmp_path) == first

    (tmp_path / "b.bin").write_bytes(b"changed")
    assert hash_adapter_dir(tmp_path) != first


def test_load_refuses_missing_and_corrupt(tmp_path: Path) -> None:
    with pytest.raises(ResumeError, match="nothing to resume"):
        load_manifest(tmp_path)
    (tmp_path / "adapter-manifest.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ResumeError):
        load_manifest(tmp_path)


def test_resume_accepts_a_matching_config() -> None:
    check_resume(_manifest(), _config(), "sha256:" + "a" * 64)


@pytest.mark.parametrize(
    "override",
    [
        {"base_model": "unsloth/Llama-3.2-3B-Instruct"},
        {"base_revision": "v0.2"},
        {"lora_r": 64},
        {"dataset_hash": "sha256:" + "b" * 64},
        {"variant_id": "mixed-500-Qwen2.5-7B-Instruct-r16-ffffffff"},
    ],
)
def test_resume_refuses_any_mismatch(override: dict) -> None:
    manifest = _manifest(**override)
    with pytest.raises(ResumeError):
        check_resume(manifest, _config(), "sha256:" + "a" * 64)


def test_manifest_json_carries_every_provenance_field(tmp_path: Path) -> None:
    save_manifest(tmp_path, _manifest())
    payload = json.loads((tmp_path / "adapter-manifest.json").read_text(encoding="utf-8"))
    provenance = (
        "base_model",
        "base_revision",
        "dataset_hash",
        "curation_tier",
        "lora_r",
        "session",
    )
    for field in provenance:
        assert field in payload
