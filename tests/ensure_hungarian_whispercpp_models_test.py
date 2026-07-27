"""Isolated transaction tests for the Hungarian Whisper.cpp provisioner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/ensure_hungarian_whispercpp_models.py"
SPEC = importlib.util.spec_from_file_location("hungarian_provisioner", SCRIPT)
assert SPEC and SPEC.loader
provisioner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provisioner)


def _ggml(path: Path, fill: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(provisioner.GGML_MAGIC.to_bytes(4, "little"))
        output.write(fill)
        output.truncate(provisioner.MIN_OUTPUT_SIZE)


def _entry(model: dict, target: Path, converter: dict | None = None) -> dict:
    return {
        "repo": model["repo"],
        "revision": model["revision"],
        "target": model["target"],
        "filesize": target.stat().st_size,
        "sha256": provisioner._sha256(target),
        "converter": converter or {"version": "test", "sha256": "test"},
    }


@pytest.fixture()
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(provisioner, "MODEL_ROOT", tmp_path / "models")
    provisioner._secure_directory(provisioner.MODEL_ROOT)
    monkeypatch.setattr(
        provisioner.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=provisioner.REQUIRED_FREE_BYTES * 2),
    )
    return tmp_path / "models"


def test_each_model_is_committed_and_retry_only_converts_fourth(isolated, monkeypatch):
    calls: list[str] = []
    failed = True

    def fake_convert(model, staging_root, converter):
        nonlocal failed
        target = model["target"]
        calls.append(target)
        staging = staging_root / target
        conversion = staging_root / f"conversion-{target}"
        staging.mkdir(parents=True, exist_ok=True)
        conversion.mkdir(parents=True, exist_ok=True)
        if target == provisioner.MODELS[3]["target"] and failed:
            (staging / "download-sentinel").write_text("keep me")
            failed = False
            raise RuntimeError("simulated fourth conversion failure")
        output = conversion / "ggml-model.bin"
        _ggml(output, target.encode())
        return {
            "repo": model["repo"],
            "revision": model["revision"],
            "target": target,
            "filesize": output.stat().st_size,
            "sha256": provisioner._sha256(output),
            "output": str(output),
        }

    monkeypatch.setattr(provisioner, "_download_and_convert", fake_convert)
    monkeypatch.setattr(
        provisioner,
        "_converter_info",
        lambda _: {"version": "test", "sha256": "test"},
    )

    with pytest.raises(RuntimeError, match="simulated"):
        provisioner.provision()

    assert calls == [model["target"] for model in provisioner.MODELS]
    manifest = json.loads((isolated / provisioner.MANIFEST_NAME).read_text())
    for model in provisioner.MODELS[:3]:
        assert provisioner._entry_is_valid(model, manifest)
    fourth_staging = isolated / provisioner.STAGING_NAME / provisioner.MODELS[3]["target"]
    assert (fourth_staging / "download-sentinel").read_text() == "keep me"

    provisioner.provision()
    assert calls == [model["target"] for model in provisioner.MODELS] + [
        provisioner.MODELS[3]["target"]
    ]
    assert not (isolated / provisioner.STAGING_NAME).exists()


def test_recovery_after_target_replace_publishes_manifest(isolated, monkeypatch):
    model = provisioner.MODELS[0]
    staging_root = isolated / provisioner.STAGING_NAME
    output = staging_root / f"conversion-{model['target']}" / "ggml-model.bin"
    _ggml(output, b"new")
    old_target = isolated / model["target"]
    _ggml(old_target, b"old")
    old_manifest = {"format": 1, "models": {model["target"]: _entry(model, old_target)}}
    provisioner._write_json_atomic(provisioner._manifest_path(), old_manifest)
    result = {
        "repo": model["repo"],
        "revision": model["revision"],
        "target": model["target"],
        "filesize": output.stat().st_size,
        "sha256": provisioner._sha256(output),
        "output": str(output),
    }

    original_write = provisioner._write_json_atomic
    manifest_path = provisioner._manifest_path()

    def crash_on_manifest(path, value):
        if path == manifest_path:
            raise KeyboardInterrupt()
        return original_write(path, value)

    monkeypatch.setattr(provisioner, "_write_json_atomic", crash_on_manifest)
    with pytest.raises(KeyboardInterrupt):
        provisioner._commit_model(
            model,
            result,
            old_manifest,
            staging_root,
            {"version": "test", "sha256": "test"},
        )

    marker = provisioner._transaction_path(staging_root, model["target"])
    assert marker.exists()
    assert provisioner._valid_ggml(old_target)

    monkeypatch.setattr(provisioner, "_write_json_atomic", original_write)
    recovered = provisioner._recover_transactions(old_manifest, staging_root)
    assert recovered and provisioner._entry_is_valid(model, recovered)
    assert provisioner._entry_is_valid(model, provisioner._read_manifest())
    assert not marker.exists()
    assert not provisioner._backup_path(staging_root, model["target"]).exists()
