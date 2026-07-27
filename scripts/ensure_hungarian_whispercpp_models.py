#!/usr/bin/env python3
"""Provision the pinned Hungarian Whisper.cpp models used by Buzz.

Each model has its own durable transaction.  This is intentionally separate
from Buzz's normal model downloader: an interrupted conversion must not make
already-installed models get downloaded and converted again.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

MODEL_ROOT = Path(os.environ.get("BUZZ_MODEL_ROOT", str(Path.home() / ".cache/Buzz/models")))
MANIFEST_NAME = ".hungarian-whispercpp-manifest.json"
LOCK_NAME = ".hungarian-whispercpp.lock"
STAGING_NAME = ".hungarian-whispercpp-staging"
MIN_OUTPUT_SIZE = 1024 * 1024
REQUIRED_FREE_BYTES = 12 * 1024**3
# The old all-at-once operation reserved 12 GiB for four models.  Keep the
# same safety estimate, but only reserve it for checkpoints still pending.
FREE_BYTES_PER_MODEL = REQUIRED_FREE_BYTES // 4
GGML_MAGIC = 0x67676D6C

MODELS = (
    {
        "repo": "sarpba/whisper-base-hungarian_v1",
        "revision": "2d5825d0d97c65a5ac92f69eb3ea23914ba2ed5c",
        "target": "ggml-hungarian-base.bin",
    },
    {
        "repo": "sarpba/whisper-hu-tiny-finetuned-V2",
        "revision": "7aff1823ddceb0e4412ae286b6391eebd74a2651",
        "target": "ggml-hungarian-tiny-v2.bin",
    },
    {
        "repo": "sarpba/whisper-hu-small-finetuned",
        "revision": "695bec9fd9ac32998ade9cfe59e7e486695f7339",
        "target": "ggml-hungarian-small.bin",
    },
    {
        "repo": "sarpba/whisper-hu-large-v3-turbo-finetuned",
        "revision": "9d63092bd80b66729b86f2c6d044a964afb39f7f",
        "target": "ggml-hungarian-large-v3-turbo.bin",
    },
)

ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "pytorch_model.bin",
    "pytorch_model-*.bin",
    "pytorch_model.bin.index.json",
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
]


def _error(message: str) -> None:
    print(f"Hungarian Whisper.cpp provisioning error: {message}", file=sys.stderr)


def _secure_directory(path: Path) -> None:
    """Create/check a provisioner-owned directory without following its link."""
    if path.is_symlink():
        raise RuntimeError(f"refusing symlink directory: {path}")
    if path.exists():
        if not path.is_dir():
            raise RuntimeError(f"not a directory: {path}")
        return
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"directory was not created safely: {path}")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_ggml(path: Path) -> bool:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size < MIN_OUTPUT_SIZE:
            return False
        with path.open("rb") as source:
            return int.from_bytes(source.read(4), "little") == GGML_MAGIC
    except OSError:
        return False


def _manifest_path() -> Path:
    return MODEL_ROOT / MANIFEST_NAME


def _read_manifest() -> dict | None:
    path = _manifest_path()
    if path.is_symlink():
        raise RuntimeError(f"refusing symlink manifest: {path}")
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
        return value if isinstance(value, dict) else None
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return None


def _model_for_target(target: str) -> dict | None:
    return next((model for model in MODELS if model["target"] == target), None)


def _entry_is_valid(model: dict, manifest: dict | None) -> bool:
    target = MODEL_ROOT / model["target"]
    entries = manifest.get("models", {}) if manifest else {}
    entry = entries.get(model["target"]) if isinstance(entries, dict) else None
    if not isinstance(entry, dict) or "output" in entry:
        return False
    if entry.get("repo") != model["repo"] or entry.get("revision") != model["revision"]:
        return False
    return _entry_matches_file(model, entry, target)


def _entry_matches_file(model: dict, entry: dict, target: Path) -> bool:
    if entry.get("target") != model["target"] or not _valid_ggml(target):
        return False
    converter = entry.get("converter")
    if (
        not isinstance(converter, dict)
        or not isinstance(converter.get("version"), str)
        or not converter["version"]
        or not isinstance(converter.get("sha256"), str)
        or not converter["sha256"]
    ):
        return False
    try:
        return (
            isinstance(entry.get("filesize"), int)
            and entry["filesize"] == target.stat().st_size
            and entry.get("sha256") == _sha256(target)
        )
    except OSError:
        return False


def _complete(manifest: dict | None) -> bool:
    return all(_entry_is_valid(model, manifest) for model in MODELS)


@contextmanager
def _model_lock():
    _secure_directory(MODEL_ROOT)
    lock_path = MODEL_ROOT / LOCK_NAME
    if lock_path.is_symlink():
        raise RuntimeError(f"refusing symlink lock: {lock_path}")
    with lock_path.open("a+") as lock:
        os.chmod(lock_path, stat.S_IRUSR | stat.S_IWUSR)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def _converter_info(converter: Path) -> dict:
    return {
        "path": str(converter),
        "version": "convert-h5-to-ggml.py",
        "sha256": _sha256(converter),
    }


def _write_json_atomic(path: Path, value: dict) -> None:
    """Write JSON and make both the file and its containing directory durable."""
    parent = path.parent
    _secure_directory(parent)
    if path.is_symlink():
        raise RuntimeError(f"refusing symlink JSON path: {path}")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(parent)
    finally:
        temporary.unlink(missing_ok=True)


def _cleanup_staging(staging_root: Path) -> None:
    if staging_root.is_symlink():
        raise RuntimeError(f"refusing symlink staging directory: {staging_root}")
    if staging_root.exists():
        if not staging_root.is_dir():
            raise RuntimeError(f"not a staging directory: {staging_root}")
        shutil.rmtree(staging_root)


def _model_staging_paths(model: dict, staging_root: Path) -> tuple[Path, Path]:
    return (
        staging_root / model["target"],
        staging_root / f"conversion-{model['target']}",
    )


def _cleanup_model_staging(model: dict, staging_root: Path) -> None:
    for path in _model_staging_paths(model, staging_root):
        if path.is_symlink():
            raise RuntimeError(f"refusing symlink staging directory: {path}")
        if path.exists():
            if not path.is_dir():
                raise RuntimeError(f"not a staging directory: {path}")
            shutil.rmtree(path)


def _clean_entry(entry: dict) -> dict:
    clean = dict(entry)
    clean.pop("output", None)
    return clean


def _manifest_with_entry(manifest: dict | None, model: dict, entry: dict) -> dict:
    old_models = manifest.get("models", {}) if isinstance(manifest, dict) else {}
    models = dict(old_models) if isinstance(old_models, dict) else {}
    # Keep existing managed entries (and any unknown metadata), while ensuring
    # no path to a transient staging output can ever be committed.
    for existing_model in MODELS:
        old_entry = models.get(existing_model["target"])
        if isinstance(old_entry, dict):
            models[existing_model["target"]] = _clean_entry(old_entry)
    models[model["target"]] = _clean_entry(entry)
    return {"format": 1, "models": models}


def _transaction_path(staging_root: Path, target: str) -> Path:
    return staging_root / f".transaction-{target}.json"


def _backup_path(staging_root: Path, target: str) -> Path:
    return staging_root / f".previous-{target}"


def _remove_transaction_artifacts(marker: Path, backup: Path) -> None:
    if backup.is_symlink():
        backup.unlink()
    else:
        backup.unlink(missing_ok=True)
    marker.unlink(missing_ok=True)
    _fsync_directory(marker.parent)


def _read_transaction(marker: Path) -> tuple[dict, dict, Path, bool]:
    if marker.is_symlink():
        raise RuntimeError(f"refusing symlink transaction marker: {marker}")
    try:
        with marker.open(encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid transaction marker: {marker}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid transaction marker: {marker}")
    target = value.get("target")
    if not isinstance(target, str):
        raise RuntimeError(f"invalid transaction marker: {marker}")
    model = _model_for_target(target)
    entry = value.get("entry")
    backup_name = value.get("backup")
    had_target = value.get("had_target")
    if (
        model is None
        or not isinstance(entry, dict)
        or "output" in entry
        or entry.get("target") != target
        or backup_name != _backup_path(marker.parent, target).name
        or not isinstance(had_target, bool)
    ):
        raise RuntimeError(f"invalid transaction marker: {marker}")
    return model, entry, _backup_path(marker.parent, target), had_target


def _recover_transactions(manifest: dict | None, staging_root: Path) -> dict | None:
    """Finish or undo transactions left by a crash, before computing pending work."""
    if not staging_root.exists():
        return manifest
    _secure_directory(staging_root)
    markers = sorted(
        path for path in staging_root.iterdir() if path.name.startswith(".transaction-")
    )
    for marker in markers:
        model, entry, backup, had_target = _read_transaction(marker)
        target = MODEL_ROOT / model["target"]
        marker_manifest = {"models": {model["target"]: entry}}
        if _entry_is_valid(model, marker_manifest):
            manifest = _manifest_with_entry(manifest, model, entry)
            _write_json_atomic(_manifest_path(), manifest)
            _remove_transaction_artifacts(marker, backup)
            _cleanup_model_staging(model, staging_root)
            continue

        # The target replacement did not produce the promised file.  A
        # regular backup is the old target and is always preferred.  A symlink
        # backup is removed rather than restored, so recovery never installs a
        # path whose contents are outside the provisioner root.
        if backup.exists() or backup.is_symlink():
            if target.exists() or target.is_symlink():
                target.unlink()
            if backup.is_symlink():
                backup.unlink()
            else:
                os.replace(backup, target)
                _fsync_directory(MODEL_ROOT)
        elif not had_target and (target.exists() or target.is_symlink()):
            target.unlink()
            _fsync_directory(MODEL_ROOT)
        marker.unlink(missing_ok=True)
        _fsync_directory(staging_root)
    return manifest


def _download_and_convert(model: dict, staging_root: Path, converter: Path) -> dict:
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    import whisper  # type: ignore[import-not-found]

    staging, conversion = _model_staging_paths(model, staging_root)
    _secure_directory(staging)
    _secure_directory(conversion)

    print(f"Downloading {model['target']} from {model['repo']}@{model['revision']}", file=sys.stderr)
    snapshot_download(
        repo_id=model["repo"],
        revision=model["revision"],
        local_dir=str(staging),
        allow_patterns=ALLOW_PATTERNS,
    )

    required = [staging / "config.json", staging / "vocab.json", staging / "added_tokens.json"]
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"downloaded checkpoint for {model['target']} is missing converter metadata")
    if not any(staging.glob(pattern) for pattern in ("pytorch_model*.bin", "model*.safetensors")):
        raise RuntimeError(f"downloaded checkpoint for {model['target']} has no model weights")

    if whisper.__file__ is None:
        raise RuntimeError("installed whisper package has no module path")
    whisper_package = Path(whisper.__file__).resolve().parent
    mel_filters = whisper_package / "assets" / "mel_filters.npz"
    if not mel_filters.is_file():
        raise RuntimeError(f"installed whisper package has no mel_filters.npz: {mel_filters}")
    whisper_root = whisper_package.parent
    output = conversion / "ggml-model.bin"
    output.unlink(missing_ok=True)
    print(f"Converting {model['target']} with whisper.cpp converter", file=sys.stderr)
    subprocess.run(
        [sys.executable, str(converter), str(staging), str(whisper_root), str(conversion)],
        check=True,
    )
    if not _valid_ggml(output):
        raise RuntimeError(f"converter produced an invalid output for {model['target']}")

    return {
        "repo": model["repo"],
        "revision": model["revision"],
        "target": model["target"],
        "filesize": output.stat().st_size,
        "sha256": _sha256(output),
        "output": str(output),
    }


def _commit_model(
    model: dict,
    result: dict,
    manifest: dict | None,
    staging_root: Path,
    converter_data: dict,
) -> dict:
    """Commit one converted output, leaving a marker until both files commit."""
    target = MODEL_ROOT / model["target"]
    output = Path(result.get("output", ""))
    expected_output = _model_staging_paths(model, staging_root)[1] / "ggml-model.bin"
    if output != expected_output or not _valid_ggml(output):
        raise RuntimeError(f"staged model output failed validation for {model['target']}")
    _fsync_file(output)
    entry = dict(result)
    entry.pop("output", None)
    entry["converter"] = dict(converter_data)
    if not _entry_matches_file(model, entry, output):
        # This also verifies the recorded size/hash against the staged file.
        raise RuntimeError(f"staged model manifest entry failed validation for {model['target']}")

    marker = _transaction_path(staging_root, model["target"])
    backup = _backup_path(staging_root, model["target"])
    if backup.exists() or backup.is_symlink() or marker.exists() or marker.is_symlink():
        raise RuntimeError(f"stale transaction artifacts for {model['target']}")
    had_target = target.exists() or target.is_symlink()
    _write_json_atomic(
        marker,
        {
            "format": 1,
            "target": model["target"],
            "entry": entry,
            "backup": backup.name,
            "had_target": had_target,
        },
    )
    try:
        if had_target:
            os.replace(target, backup)
            _fsync_directory(MODEL_ROOT)
        os.replace(output, target)
        _fsync_directory(MODEL_ROOT)
        if not _entry_is_valid(model, {"models": {model["target"]: entry}}):
            raise RuntimeError(f"installed model output failed validation for {model['target']}")
        committed_manifest = _manifest_with_entry(manifest, model, entry)
        _write_json_atomic(_manifest_path(), committed_manifest)
    except BaseException:
        # Do not roll back here: the durable marker records whether the old
        # target is in backup and lets the next locked startup finish or undo
        # this exact two-file transaction, including Ctrl-C.
        raise

    _remove_transaction_artifacts(marker, backup)
    _cleanup_model_staging(model, staging_root)
    return committed_manifest


def check() -> int:
    manifest = _read_manifest()
    missing = [model["target"] for model in MODELS if not _entry_is_valid(model, manifest)]
    if missing:
        _error("--check incomplete; missing or unverified: " + ", ".join(missing))
        return 1
    print("Hungarian Whisper.cpp model manifest and paths are complete", file=sys.stderr)
    return 0


def provision() -> int:
    with _model_lock():
        staging_root = MODEL_ROOT / STAGING_NAME
        if staging_root.exists() or staging_root.is_symlink():
            _secure_directory(staging_root)
        manifest = _read_manifest()
        manifest = _recover_transactions(manifest, staging_root)
        pending = [model for model in MODELS if not _entry_is_valid(model, manifest)]
        for model in MODELS:
            if model not in pending and staging_root.exists():
                _cleanup_model_staging(model, staging_root)
        if not pending:
            _cleanup_staging(staging_root)
            print("Hungarian Whisper.cpp models are already provisioned", file=sys.stderr)
            return 0

        free = shutil.disk_usage(MODEL_ROOT).free
        required_free = len(pending) * FREE_BYTES_PER_MODEL
        if free < required_free:
            raise RuntimeError(
                f"at least {required_free / 1024**3:.0f} GiB free disk space is required for "
                f"{len(pending)} pending model(s); only {free / 1024**3:.1f} GiB is available"
            )

        converter = Path(__file__).resolve().parents[1] / "whisper.cpp/models/convert-h5-to-ggml.py"
        if not converter.is_file():
            raise RuntimeError(f"converter not found: {converter}")
        _secure_directory(staging_root)
        if any(model["target"] == "ggml-hungarian-base.bin" for model in pending):
            print(
                "!!! NOTICE: Hungarian Base has an upstream non-commercial restriction. "
                "It is not being represented as commercial-ready. !!!",
                file=sys.stderr,
            )

        converter_data = _converter_info(converter)
        for model in pending:
            result = _download_and_convert(model, staging_root, converter)
            manifest = _commit_model(model, result, manifest, staging_root, converter_data)

        _cleanup_staging(staging_root)
        print("Hungarian Whisper.cpp provisioning complete", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify local files and manifest without downloading")
    args = parser.parse_args()
    try:
        return check() if args.check else provision()
    except Exception as exc:
        _error(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
