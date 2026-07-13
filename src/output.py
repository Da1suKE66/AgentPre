"""Deterministic, inspectable run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .errors import FailureCode, PipelineError


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                value,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                default=_json_default,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except (OSError, TypeError, ValueError) as exc:
        raise PipelineError(
            FailureCode.OUTPUT_FAILURE,
            f"failed to write {path}",
            stage="output",
            details={"error": repr(exc)},
        ) from exc


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=_json_default,
                        allow_nan=False,
                    )
                    + "\n"
                )
        temporary.replace(path)
    except (OSError, TypeError, ValueError) as exc:
        raise PipelineError(
            FailureCode.OUTPUT_FAILURE,
            f"failed to write {path}",
            stage="output",
            details={"error": repr(exc)},
        ) from exc


def write_trajectory(path: Path, arrays: dict[str, np.ndarray]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    except OSError as exc:
        raise PipelineError(
            FailureCode.OUTPUT_FAILURE,
            f"failed to write {path}",
            stage="output",
            details={"error": repr(exc)},
        ) from exc
