#!/usr/bin/env python3
"""Copy a generated Articraft record build into the AgentPre cache.

The Articraft harness intentionally generates URDF and mesh files outside the
small source checkout.  This helper makes one immutable, hash-audited copy
under ``/cache/liluchen/agentpre/assets`` and refuses to overwrite a different
existing materialization.  It inventories the generated files; it does not run
the compiler or establish task-level acceptance.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

if __package__:
    from .apply_articraft_inertials import (
        InertialSpecificationError,
        load_specification,
        validate_completed_inertials,
    )
else:  # Direct execution as ``python scripts/materialize_articraft_asset.py``.
    from apply_articraft_inertials import (  # type: ignore[no-redef]
        InertialSpecificationError,
        load_specification,
        validate_completed_inertials,
    )


RECORD_ID = "rec_microwave_oven_5e86f3429e954dcd9ab6c9d3a94db707"
RECORD_REVISION = "rev_000001"
ARTICRAFT_COMMIT = "59eb5e0ed72a734111012b43f881423b15d4931d"
DATA_COMMIT = "0cdcaa49f5571e9b4df04476c7f09587ee3ab7bd"
MODEL_URL = (
    "https://github.com/mattzh72/articraft-data/blob/"
    f"{DATA_COMMIT}/records/{RECORD_ID}/revisions/{RECORD_REVISION}/model.py"
)
MANIFEST_NAME = f"{RECORD_ID}.manifest.json"
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
INERTIAL_SIDECAR_NAME = "agentpre_inertial_completion.json"


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _open_regular_no_follow(path: Path) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    descriptor = os.open(path, flags)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise RuntimeError(f"Articraft input is not a regular file: {path}")
    return descriptor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with os.fdopen(_open_regular_no_follow(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_bytes_no_follow(path: Path) -> bytes:
    with os.fdopen(_open_regular_no_follow(path), "rb") as stream:
        return stream.read()


def _asset_inventory(root: Path) -> list[dict[str, Any]]:
    """Hash the runtime URDF/assets, excluding volatile compile diagnostics."""

    if root.is_symlink():
        raise RuntimeError(f"Articraft inventory root must not be a symlink: {root}")
    candidates = [root / "model.urdf"]
    assets = root / "assets"
    if assets.is_symlink():
        raise RuntimeError(f"Articraft inventory must not contain symlinks: {assets}")
    if assets.is_dir():
        candidates.extend(assets.rglob("*"))
    files: list[dict[str, Any]] = []
    for path in sorted(candidates):
        if path.is_symlink():
            raise RuntimeError(f"Articraft inventory must not contain symlinks: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(f"Articraft inventory contains a non-regular file: {path}")
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return files


def _commit_sha(value: str, name: str) -> str:
    if not isinstance(value, str) or _COMMIT_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 40-character Git commit SHA")
    return value


def _json_object_no_follow(
    path: Path, description: str
) -> tuple[dict[str, Any], bytes]:
    try:
        raw = _regular_bytes_no_follow(path)
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{description} is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{description} must be a JSON object: {path}")
    return payload, raw


def _exact_object_keys(
    value: Any, expected: set[str], description: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise RuntimeError(
            f"{description} has unexpected schema: expected={sorted(expected)}, "
            f"actual={actual}"
        )
    return value


def _inertial_provenance(
    source_root: Path,
    inertial_spec: Path,
    inertial_sidecar: Path,
    *,
    data_commit: str,
) -> dict[str, Any]:
    if inertial_spec.is_symlink():
        raise RuntimeError(
            f"Articraft inertial specification must not be a symlink: {inertial_spec}"
        )
    if inertial_sidecar.is_symlink():
        raise RuntimeError(
            f"Articraft inertial sidecar must not be a symlink: {inertial_sidecar}"
        )
    spec_bytes = _regular_bytes_no_follow(inertial_spec)
    spec_sha256 = hashlib.sha256(spec_bytes).hexdigest()
    try:
        strict_specification = load_specification(inertial_spec)
    except (FileNotFoundError, InertialSpecificationError) as exc:
        raise RuntimeError(
            f"Articraft inertial specification failed strict validation: {inertial_spec}"
        ) from exc
    expected_record = {
        "id": RECORD_ID,
        "revision": RECORD_REVISION,
        "data_commit": data_commit,
        "model_url": MODEL_URL,
    }
    actual_record = {
        key: strict_specification["record"][key] for key in expected_record
    }
    if actual_record != expected_record:
        raise RuntimeError(
            "Articraft inertial specification record does not match the requested "
            f"materialization: existing={actual_record}, expected={expected_record}"
        )
    sidecar, sidecar_bytes = _json_object_no_follow(
        inertial_sidecar, "Articraft inertial completion sidecar"
    )
    root = _exact_object_keys(
        sidecar,
        {"schema_version", "specification", "urdf", "injected_links"},
        "Articraft inertial completion sidecar",
    )
    specification = _exact_object_keys(
        root["specification"],
        {"path", "sha256"},
        "Articraft inertial completion specification",
    )
    urdf = _exact_object_keys(
        root["urdf"],
        {"path", "pre_sha256", "post_sha256"},
        "Articraft inertial completion URDF",
    )
    compiled_urdf = source_root / "model.urdf"
    compiled_urdf_bytes = _regular_bytes_no_follow(compiled_urdf)
    compiled_urdf_sha256 = hashlib.sha256(compiled_urdf_bytes).hexdigest()
    expected_specification = {
        "path": str(inertial_spec),
        "sha256": spec_sha256,
    }
    if root["schema_version"] != 1:
        raise RuntimeError("Articraft inertial completion schema_version must equal 1")
    if specification != expected_specification:
        raise RuntimeError(
            "Articraft inertial completion sidecar does not match its specification: "
            f"existing={specification}, expected={expected_specification}"
        )
    if urdf["path"] != str(compiled_urdf) or urdf["post_sha256"] != compiled_urdf_sha256:
        raise RuntimeError(
            "Articraft inertial completion sidecar does not match compiled model.urdf"
        )
    for name in ("pre_sha256", "post_sha256"):
        value = urdf[name]
        if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
            raise RuntimeError(
                f"Articraft inertial completion {name} is not a lowercase SHA-256"
            )
    if urdf["pre_sha256"] == urdf["post_sha256"]:
        raise RuntimeError(
            "Articraft inertial completion pre/post URDF hashes must differ"
        )
    injected_links = root["injected_links"]
    if (
        not isinstance(injected_links, list)
        or not injected_links
        or any(not isinstance(name, str) or not name for name in injected_links)
        or injected_links != sorted(set(injected_links))
    ):
        raise RuntimeError(
            "Articraft inertial completion injected_links must be a sorted, unique, "
            "non-empty string list"
        )
    try:
        compiled_root = ET.fromstring(compiled_urdf_bytes)
    except ET.ParseError as exc:
        raise RuntimeError(f"compiled Articraft URDF is invalid: {compiled_urdf}") from exc
    urdf_links: list[str] = []
    for link in compiled_root.findall("link"):
        name = link.get("name")
        if not name or name in urdf_links:
            raise RuntimeError(
                "compiled Articraft URDF has an unnamed or duplicate link"
            )
        if len(link.findall("inertial")) != 1:
            raise RuntimeError(
                f"compiled Articraft URDF link {name!r} does not have exactly one inertial"
            )
        urdf_links.append(name)
    if sorted(urdf_links) != injected_links:
        raise RuntimeError(
            "Articraft inertial completion injected_links do not match model.urdf: "
            f"sidecar={injected_links}, urdf={sorted(urdf_links)}"
        )
    try:
        strict_verification = validate_completed_inertials(
            compiled_urdf, inertial_spec, inertial_sidecar
        )
    except (FileNotFoundError, InertialSpecificationError, RuntimeError) as exc:
        raise RuntimeError(
            "compiled Articraft inertials do not strictly match the checked-in "
            "specification and completion sidecar"
        ) from exc
    if strict_verification["modified"]:
        raise AssertionError("read-only inertial validation reported a modification")
    if strict_verification["specification_sha256"] != spec_sha256:
        raise RuntimeError(
            "Articraft inertial specification changed during strict verification"
        )
    return {
        "specification": {
            "path": str(inertial_spec),
            "size_bytes": len(spec_bytes),
            "sha256": spec_sha256,
        },
        "completion_sidecar": {
            "path": str(inertial_sidecar),
            "size_bytes": len(sidecar_bytes),
            "sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
            "content": sidecar,
        },
    }


def _copy_regular_file_no_follow(source: Path, destination: Path) -> None:
    """Copy one regular file without following a source symlink."""

    descriptor = _open_regular_no_follow(source)
    try:
        with os.fdopen(descriptor, "rb") as input_stream, destination.open(
            "xb"
        ) as output_stream:
            shutil.copyfileobj(input_stream, output_stream, 1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


@contextmanager
def _exclusive_lock(path: Path):
    if path.is_symlink():
        raise RuntimeError(f"Articraft lock must not be a symlink: {path}")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "r+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    except BaseException:
        # fdopen owns and closes the descriptor once it succeeds.  If it fails,
        # the raw descriptor still belongs to this helper.
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _read_existing_manifest(path: Path) -> dict[str, Any] | None:
    if path.is_symlink():
        raise RuntimeError(f"Articraft manifest must not be a symlink: {path}")
    if not path.exists():
        return None
    if not path.is_file():
        raise RuntimeError(f"Articraft manifest is not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"existing Articraft manifest is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"existing Articraft manifest is not an object: {path}")
    return payload


def _write_manifest_atomic(path: Path, manifest: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_compiled_source(source_root: Path) -> None:
    if not source_root.is_dir():
        raise FileNotFoundError(f"Articraft materialization is missing: {source_root}")
    urdf = source_root / "model.urdf"
    if not urdf.is_file():
        raise FileNotFoundError(f"compiled Articraft URDF is missing: {urdf}")


def materialize(
    source_root: Path,
    cache_root: Path,
    *,
    inertial_spec: Path,
    inertial_sidecar: Path | None = None,
    articraft_commit: str = ARTICRAFT_COMMIT,
    data_commit: str = DATA_COMMIT,
) -> dict[str, Any]:
    """Create or verify the immutable cached record and return its manifest."""

    source_input = _absolute_without_symlink_resolution(source_root)
    cache_input = _absolute_without_symlink_resolution(cache_root)
    if source_input.is_symlink():
        raise RuntimeError(f"Articraft source root must not be a symlink: {source_input}")
    if cache_input.is_symlink():
        raise RuntimeError(f"AgentPre cache root must not be a symlink: {cache_input}")
    source_root = source_input.resolve()
    cache_root = cache_input.resolve()
    inertial_spec_input = _absolute_without_symlink_resolution(inertial_spec)
    if inertial_spec_input.is_symlink():
        raise RuntimeError(
            f"Articraft inertial specification must not be a symlink: "
            f"{inertial_spec_input}"
        )
    inertial_spec = inertial_spec_input.resolve()
    if inertial_sidecar is None:
        inertial_sidecar = source_root / INERTIAL_SIDECAR_NAME
    inertial_sidecar_input = _absolute_without_symlink_resolution(inertial_sidecar)
    if inertial_sidecar_input.is_symlink():
        raise RuntimeError(
            f"Articraft inertial sidecar must not be a symlink: "
            f"{inertial_sidecar_input}"
        )
    inertial_sidecar = inertial_sidecar_input.resolve()
    articraft_commit = _commit_sha(articraft_commit, "articraft_commit")
    data_commit = _commit_sha(data_commit, "data_commit")
    _require_compiled_source(source_root)
    inertial_provenance = _inertial_provenance(
        source_root,
        inertial_spec,
        inertial_sidecar,
        data_commit=data_commit,
    )
    source_files = _asset_inventory(source_root)

    assets_root = cache_root / "assets"
    asset_parent = assets_root / "articraft"
    destination = asset_parent / RECORD_ID
    if (
        destination == source_root
        or destination in source_root.parents
        or source_root in destination.parents
    ):
        raise RuntimeError("source and destination Articraft roots must be distinct")
    for directory in (assets_root, asset_parent):
        if directory.is_symlink():
            raise RuntimeError(
                f"Articraft cache directory must not be a symlink: {directory}"
            )
        directory.mkdir(parents=True, exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise RuntimeError(
                f"Articraft cache directory is not a regular directory: {directory}"
            )

    manifest_path = asset_parent / MANIFEST_NAME
    lock_path = asset_parent / f".{RECORD_ID}.lock"
    compile_report_path = source_root / "compile_report.json"
    if compile_report_path.is_symlink():
        raise RuntimeError(
            f"Articraft compile report must not be a symlink: {compile_report_path}"
        )
    source_compile_report = (
        {
            "path": str(compile_report_path),
            "size_bytes": compile_report_path.stat().st_size,
            "sha256": _sha256(compile_report_path),
        }
        if compile_report_path.is_file()
        else None
    )
    expected_identity = {
        "schema_version": 1,
        "record_id": RECORD_ID,
        "record_revision": RECORD_REVISION,
        "articraft_commit": articraft_commit,
        "data_commit": data_commit,
    }
    stable_manifest = {
        **expected_identity,
        "articraft_repository": "https://github.com/mattzh72/articraft",
        "articraft_license": "Apache-2.0",
        "data_repository": "https://github.com/mattzh72/articraft-data",
        "data_license": "CC-BY-4.0",
        "source_root": str(source_root),
        "destination_root": str(destination),
        "urdf": str(destination / "model.urdf"),
        "file_count": len(source_files),
        "total_size_bytes": sum(item["size_bytes"] for item in source_files),
        "files": source_files,
        "manifest_path": str(manifest_path),
        "inertial_postprocessing": inertial_provenance,
    }
    with _exclusive_lock(lock_path):
        existing_manifest = _read_existing_manifest(manifest_path)
        if existing_manifest is not None:
            allowed_fields = set(stable_manifest) | {"source_compile_report"}
            unexpected = sorted(set(existing_manifest) - allowed_fields)
            missing = sorted(allowed_fields - set(existing_manifest))
            mismatches = {
                field: {
                    "existing": existing_manifest.get(field),
                    "requested": expected,
                }
                for field, expected in stable_manifest.items()
                if existing_manifest.get(field) != expected
            }
            if unexpected or missing or mismatches:
                raise RuntimeError(
                    "refusing to rewrite Articraft provenance for an existing cache: "
                    f"unexpected_fields={unexpected}, missing_fields={missing}, "
                    f"mismatches={mismatches}"
                )

        if (
            _inertial_provenance(
                source_root,
                inertial_spec,
                inertial_sidecar,
                data_commit=data_commit,
            )
            != inertial_provenance
        ):
            raise RuntimeError(
                "Articraft inertial provenance changed before materialization"
            )

        copied = False
        if destination.is_symlink():
            raise RuntimeError(
                f"Articraft destination must not be a symlink: {destination}"
            )
        if destination.exists():
            if not destination.is_dir():
                raise RuntimeError(
                    f"Articraft destination is not a directory: {destination}"
                )
            destination_files = _asset_inventory(destination)
            if destination_files != source_files:
                raise RuntimeError(
                    "refusing to overwrite a different cached Articraft materialization: "
                    f"{destination}"
                )
        else:
            temporary = Path(
                tempfile.mkdtemp(prefix=f".{RECORD_ID}.", dir=str(asset_parent))
            )
            try:
                _copy_regular_file_no_follow(
                    source_root / "model.urdf", temporary / "model.urdf"
                )
                if (source_root / "assets").is_dir():
                    shutil.copytree(
                        source_root / "assets",
                        temporary / "assets",
                        symlinks=True,
                    )
                if _asset_inventory(temporary) != source_files:
                    raise RuntimeError(
                        "copied Articraft inventory does not match its source"
                    )
                if _asset_inventory(source_root) != source_files:
                    raise RuntimeError(
                        "Articraft source changed while it was being materialized"
                    )
                if (
                    _inertial_provenance(
                        source_root,
                        inertial_spec,
                        inertial_sidecar,
                        data_commit=data_commit,
                    )
                    != inertial_provenance
                ):
                    raise RuntimeError(
                        "Articraft inertial provenance changed while staging; "
                        "canonical cache was not published"
                    )
                os.replace(temporary, destination)
                copied = True
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)

        if existing_manifest is None:
            persisted_manifest = {
                **stable_manifest,
                "source_compile_report": source_compile_report,
            }
            _write_manifest_atomic(manifest_path, persisted_manifest)
        else:
            persisted_manifest = existing_manifest
        return {**persisted_manifest, "copied": copied}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/cache/liluchen/articraft-data/cache/record_materialization")
        / RECORD_ID,
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(
            os.environ.get("AGENTPRE_CACHE_ROOT", "/cache/liluchen/agentpre")
        ),
    )
    parser.add_argument("--articraft-commit", default=ARTICRAFT_COMMIT)
    parser.add_argument("--data-commit", default=DATA_COMMIT)
    parser.add_argument("--inertial-spec", type=Path, required=True)
    parser.add_argument(
        "--inertial-sidecar",
        type=Path,
        help=(
            "completion sidecar (default: source root/"
            f"{INERTIAL_SIDECAR_NAME})"
        ),
    )
    args = parser.parse_args()

    manifest = materialize(
        args.source_root,
        args.cache_root,
        inertial_spec=args.inertial_spec,
        inertial_sidecar=args.inertial_sidecar,
        articraft_commit=args.articraft_commit,
        data_commit=args.data_commit,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
