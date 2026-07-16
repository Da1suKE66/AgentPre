"""Upper-level AgentPre orchestration for a new articulated appliance.

The low-level :mod:`src.run` command deliberately requires a complete audited
configuration.  This module is the small decision layer above it: resolve a
URDF or materialized Articraft record, infer task semantics, emit a reviewable
configuration/manifest, and execute that frozen configuration on request.

``prepare`` never starts IK or physics.  ``execute`` never re-infers semantics;
it runs the exact configuration recorded by ``prepare``.  This split keeps
expensive runs reproducible and makes every automatic choice inspectable.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np

from .affordances import (
    CandidateGenerationConfig,
    extract_handle_geometry,
    generate_handle_candidates_from_geometry,
    load_affordances,
)
from .asset_inspector import inspect_asset
from .animation import write_animation_html
from .config import validate_config
from .door_kinematics import forward_kinematics
from .output import write_json
from .urdf_model import URDFModel, URDFModelError, load_urdf


MANIFEST_NAME = "agent_manifest.json"
CONFIG_NAME = "agent_config.json"
GENERATED_AFFORDANCES_NAME = "affordances.generated.json"
MANIFEST_SCHEMA_VERSION = 1
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TEMPLATE = _PROJECT_ROOT / "configs" / "articraft_microwave_franka.json"
_DEFAULT_POLICY = _PROJECT_ROOT / "configs" / "upper_agent_policy.json"


class AgentError(RuntimeError):
    """Expected upper-agent failure with a stable machine-readable contract."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str,
        details: Mapping[str, Any] | None = None,
        recoverable: bool = True,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.details = dict(details or {})
        self.recoverable = bool(recoverable)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "stage": self.stage,
            "message": str(self),
            "details": self.details,
            "recoverable": self.recoverable,
        }


@dataclass(frozen=True, slots=True)
class Decision:
    """One selected value and enough provenance to review or override it."""

    value: Any
    source: str
    confidence: float
    rationale: str
    previous_value: Any | None = None
    override_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "source": self.source,
            "confidence": float(self.confidence),
            "rationale": self.rationale,
            "override": {
                "applied": self.override_applied,
                "previous_value": self.previous_value,
            },
        }


@dataclass(frozen=True, slots=True)
class AgentOutcome:
    workspace: Path
    manifest_path: Path
    status: str
    success: bool
    exit_code: int
    failure: Mapping[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "success": self.success,
            "status": self.status,
            "workspace": str(self.workspace),
            "manifest": str(self.manifest_path),
            "exit_code": self.exit_code,
        }
        if self.failure is not None:
            payload["failure"] = dict(self.failure)
        return payload


SemanticProvider = Callable[[Path], Mapping[str, Any]]
Runner = Callable[[Sequence[str]], int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AgentError(
            "input_unreadable",
            f"cannot hash input file: {path}",
            stage="source_resolution",
            details={"path": str(path), "error": repr(exc)},
        ) from exc
    return digest.hexdigest()


def _read_json_object(path: Path, *, stage: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentError(
            "json_invalid",
            f"cannot read JSON object: {path}",
            stage=stage,
            details={"path": str(path), "error": repr(exc)},
        ) from exc
    if not isinstance(value, dict):
        raise AgentError(
            "json_invalid",
            f"JSON root must be an object: {path}",
            stage=stage,
            details={"path": str(path)},
        )
    return value


def _expanded_path(value: str, *, relative_to: Path) -> Path:
    cache_root = os.environ.get("AGENTPRE_CACHE_ROOT", "/cache/liluchen/agentpre")
    expanded = value.replace("${AGENTPRE_CACHE_ROOT}", cache_root)
    path = Path(os.path.expanduser(os.path.expandvars(expanded)))
    return (path if path.is_absolute() else relative_to / path).resolve()


def _record_metadata_urdf(path: Path) -> tuple[Path, dict[str, Any]]:
    metadata = _read_json_object(path, stage="source_resolution")
    raw = metadata.get("urdf") or metadata.get("materialized_urdf")
    if not isinstance(raw, str) or not raw.strip():
        raise AgentError(
            "record_not_materialized",
            "Articraft metadata does not identify a materialized URDF",
            stage="source_resolution",
            details={"metadata": str(path), "accepted_fields": ["urdf", "materialized_urdf"]},
        )
    return _expanded_path(raw, relative_to=path.parent), metadata


def resolve_source(
    *,
    urdf: Path | None = None,
    articraft_record: str | None = None,
) -> dict[str, Any]:
    """Resolve exactly one local source without downloading or compiling it."""

    if (urdf is None) == (articraft_record is None):
        raise AgentError(
            "source_ambiguous",
            "provide exactly one of --urdf or --articraft-record",
            stage="source_resolution",
        )

    metadata: dict[str, Any] = {}
    metadata_path: Path | None = None
    record_id: str | None = None
    if urdf is not None:
        resolved = urdf.expanduser().resolve()
        kind = "urdf"
        supplied = str(urdf)
    else:
        assert articraft_record is not None
        supplied = articraft_record
        candidate = Path(articraft_record).expanduser()
        kind = "articraft_record"
        if candidate.exists():
            candidate = candidate.resolve()
            if candidate.is_dir():
                direct = candidate / "model.urdf"
                metadata_candidate = candidate / "source.json"
                manifest_candidates = sorted(candidate.glob("*.manifest.json"))
                if direct.is_file():
                    resolved = direct
                    metadata_path = metadata_candidate if metadata_candidate.is_file() else None
                    if metadata_path is not None:
                        metadata = _read_json_object(metadata_path, stage="source_resolution")
                elif metadata_candidate.is_file():
                    metadata_path = metadata_candidate
                    resolved, metadata = _record_metadata_urdf(metadata_candidate)
                elif len(manifest_candidates) == 1:
                    metadata_path = manifest_candidates[0]
                    resolved, metadata = _record_metadata_urdf(metadata_path)
                else:
                    raise AgentError(
                        "record_not_materialized",
                        "Articraft record directory has no unambiguous model.urdf or metadata",
                        stage="source_resolution",
                        details={"record": str(candidate)},
                    )
            elif candidate.suffix.lower() == ".urdf":
                resolved = candidate
            elif candidate.suffix.lower() == ".json":
                metadata_path = candidate
                resolved, metadata = _record_metadata_urdf(candidate)
            else:
                raise AgentError(
                    "record_input_invalid",
                    "Articraft record path must be a directory, URDF, or JSON manifest",
                    stage="source_resolution",
                    details={"record": str(candidate)},
                )
        else:
            record_id = articraft_record
            cache = Path(os.environ.get("AGENTPRE_CACHE_ROOT", "/cache/liluchen/agentpre"))
            cached = cache / "assets" / "articraft" / record_id / "model.urdf"
            checked_in = _PROJECT_ROOT / "assets" / "articraft" / record_id / "source.json"
            if cached.is_file():
                resolved = cached.resolve()
                if checked_in.is_file():
                    metadata_path = checked_in
                    metadata = _read_json_object(checked_in, stage="source_resolution")
            elif checked_in.is_file():
                metadata_path = checked_in
                resolved, metadata = _record_metadata_urdf(checked_in)
            else:
                raise AgentError(
                    "record_not_materialized",
                    "Articraft record is not available in the AgentPre cache",
                    stage="source_resolution",
                    details={
                        "record_id": record_id,
                        "expected_urdf": str(cached),
                        "expected_metadata": str(checked_in),
                    },
                )
        raw_record_id = metadata.get("record_id")
        if isinstance(raw_record_id, str) and raw_record_id:
            record_id = raw_record_id

    if not resolved.is_file():
        raise AgentError(
            "urdf_missing",
            f"resolved URDF does not exist: {resolved}",
            stage="source_resolution",
            details={"path": str(resolved), "source": supplied},
        )
    return {
        "kind": kind,
        "supplied": supplied,
        "urdf": resolved,
        "urdf_sha256": _sha256(resolved),
        "record_id": record_id,
        "metadata_path": metadata_path,
        "metadata": metadata,
    }


def _descendants(model: URDFModel, root: str) -> set[str]:
    children: dict[str, list[str]] = {}
    for joint in model.joints.values():
        children.setdefault(joint.parent, []).append(joint.child)
    result = {root}
    pending = [root]
    while pending:
        parent = pending.pop()
        for child in children.get(parent, []):
            if child not in result:
                result.add(child)
                pending.append(child)
    return result


def _local_semantic_inference(urdf_path: Path) -> dict[str, Any]:
    """Conservative name/topology fallback used when no provider is installed."""

    try:
        model = load_urdf(urdf_path)
    except URDFModelError as exc:
        raise AgentError(
            "urdf_invalid",
            str(exc),
            stage="semantic_inference",
            details=exc.to_dict(),
        ) from exc

    candidates = []
    for joint in model.joints.values():
        if joint.joint_type != "revolute" or joint.limit is None:
            continue
        if joint.limit.lower is None or joint.limit.upper is None:
            continue
        text = f"{joint.name} {joint.child}".lower()
        score = 0
        score += 8 if "door" in text else 0
        score += 5 if "hinge" in text else 0
        score += 2 if "lid" in text else 0
        span = float(joint.limit.upper - joint.limit.lower)
        score += 1 if span >= math.radians(30.0) else 0
        candidates.append((score, joint.name, joint))
    if not candidates:
        raise AgentError(
            "door_joint_not_found",
            "no finite revolute joint can be used as an appliance door",
            stage="semantic_inference",
            details={"joint_names": list(model.joint_names)},
        )
    candidates.sort(key=lambda item: (-item[0], item[1]))
    best_score, _, door_joint = candidates[0]
    if len(candidates) > 1 and candidates[1][0] == best_score:
        raise AgentError(
            "door_joint_ambiguous",
            "multiple door-joint candidates have the same semantic score",
            stage="semantic_inference",
            details={"candidates": [item[1] for item in candidates if item[0] == best_score]},
        )

    door_link = door_joint.child
    descendants = _descendants(model, door_link)
    handle_names = sorted(
        name
        for name in descendants
        if any(token in name.lower() for token in ("handle", "grip", "latch", "pull"))
    )
    handle_link = handle_names[0] if handle_names else door_link
    confidence = 0.92 if best_score >= 13 else (0.78 if best_score >= 8 else 0.45)
    return {
        "door_joint": {
            "value": door_joint.name,
            "confidence": confidence,
            "source": "local_urdf_topology",
            "rationale": "highest-scoring finite revolute joint",
        },
        "door_link": {
            "value": door_link,
            "confidence": confidence,
            "source": "door_joint_child",
            "rationale": "child link of the selected door joint",
        },
        "handle_link": {
            "value": handle_link,
            "confidence": 0.9 if handle_names else 0.55,
            "source": "local_urdf_topology",
            "rationale": (
                "named handle descendant of the door"
                if handle_names
                else "door link fallback; geometry candidate generation remains required"
            ),
        },
        "handle_frame": {
            "value": "auto_handle",
            "confidence": 0.4,
            "source": "geometry_fallback_request",
            "rationale": "missing frame intentionally triggers URDF geometry candidate generation",
        },
    }


def _load_semantic_provider(spec: str) -> SemanticProvider | None:
    if spec == "local":
        return None
    if spec == "auto":
        for module_name in ("src.asset_semantics", "src.semantic_inference"):
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as exc:
                if exc.name == module_name:
                    continue
                raise
            for name in ("infer_task_semantics", "infer_semantics"):
                provider = getattr(module, name, None)
                if callable(provider):
                    return provider
        return None
    module_name, separator, function_name = spec.partition(":")
    if not separator or not module_name or not function_name:
        raise AgentError(
            "semantic_provider_invalid",
            "semantic provider must be 'auto', 'local', or MODULE:FUNCTION",
            stage="semantic_inference",
            details={"provider": spec},
        )
    try:
        provider = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as exc:
        raise AgentError(
            "semantic_provider_unavailable",
            "cannot import semantic provider",
            stage="semantic_inference",
            details={"provider": spec, "error": repr(exc)},
        ) from exc
    if not callable(provider):
        raise AgentError(
            "semantic_provider_invalid",
            "semantic provider is not callable",
            stage="semantic_inference",
            details={"provider": spec},
        )
    return provider


def _normalize_decision(name: str, raw: Any, *, default_source: str) -> Decision:
    if isinstance(raw, Mapping) and "value" in raw:
        value = raw["value"]
        source = str(raw.get("source", default_source))
        rationale = str(raw.get("rationale", "semantic provider result"))
        confidence_raw = raw.get("confidence", 0.5)
    else:
        value = raw
        source = default_source
        rationale = "semantic provider result"
        confidence_raw = 0.5
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError) as exc:
        raise AgentError(
            "semantic_result_invalid",
            f"confidence for {name} is not numeric",
            stage="semantic_inference",
            details={"field": name, "confidence": confidence_raw},
        ) from exc
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise AgentError(
            "semantic_result_invalid",
            f"confidence for {name} must be in [0, 1]",
            stage="semantic_inference",
            details={"field": name, "confidence": confidence_raw},
        )
    return Decision(value, source, confidence, rationale)


def infer_semantic_decisions(
    urdf_path: Path,
    *,
    provider: SemanticProvider | None = None,
    provider_spec: str = "auto",
) -> tuple[dict[str, Decision], Mapping[str, Any]]:
    selected = provider if provider is not None else _load_semantic_provider(provider_spec)
    provider_name = (
        f"{selected.__module__}:{getattr(selected, '__name__', type(selected).__name__)}"
        if selected is not None
        else "local_urdf_topology"
    )
    try:
        raw = selected(urdf_path) if selected is not None else _local_semantic_inference(urdf_path)
    except AgentError:
        raise
    except Exception as exc:
        provider_details = (
            exc.to_dict()
            if callable(getattr(exc, "to_dict", None))
            else {"exception_type": type(exc).__name__, "error": str(exc)}
        )
        raise AgentError(
            "semantic_provider_failed",
            "semantic provider raised an exception",
            stage="semantic_inference",
            details={"provider": provider_name, "provider_failure": provider_details},
        ) from exc
    if not isinstance(raw, Mapping):
        raise AgentError(
            "semantic_result_invalid",
            "semantic provider must return a mapping",
            stage="semantic_inference",
            details={"provider": provider_name},
        )
    values = raw.get("decisions", raw)
    if not isinstance(values, Mapping):
        raise AgentError(
            "semantic_result_invalid",
            "semantic provider decisions must be a mapping",
            stage="semantic_inference",
            details={"provider": provider_name},
        )
    reserved = {
        "config_patch",
        "affordance_payload",
        "affordances_payload",
        "metadata",
        "decisions",
    }
    decisions = {
        str(name): _normalize_decision(str(name), value, default_source=provider_name)
        for name, value in values.items()
        if name not in reserved
    }
    missing = sorted({"door_joint", "door_link", "handle_link"} - set(decisions))
    if missing:
        raise AgentError(
            "semantic_result_incomplete",
            "semantic provider omitted required task semantics",
            stage="semantic_inference",
            details={"provider": provider_name, "missing": missing},
        )
    metadata = {
        "provider": provider_name,
        "config_patch": raw.get("config_patch") or values.get("config_patch"),
        "affordance_payload": (
            raw.get("affordance_payload")
            or raw.get("affordances_payload")
            or values.get("affordance_payload")
            or values.get("affordances_payload")
        ),
        "evidence": copy.deepcopy(raw.get("semantic_evidence", [])),
        "warnings": copy.deepcopy(raw.get("warnings", [])),
    }
    return decisions, metadata


def _override(decision: Decision, value: Any) -> Decision:
    if value is None:
        return decision
    return Decision(
        value=value,
        source="user_override",
        confidence=1.0,
        rationale="explicit CLI override",
        previous_value=decision.value,
        override_applied=True,
    )


def _deep_merge(destination: MutableMapping[str, Any], patch: Mapping[str, Any]) -> None:
    for key, value in patch.items():
        current = destination.get(key)
        if isinstance(current, MutableMapping) and isinstance(value, Mapping):
            _deep_merge(current, value)
        else:
            destination[key] = copy.deepcopy(value)


def _discover_affordances(
    source: Mapping[str, Any],
    *,
    explicit: Path | None,
    semantic_value: Any = None,
) -> tuple[Path | None, Decision]:
    candidates: list[tuple[Path, str, float]] = []
    if explicit is not None:
        candidates.append((explicit.expanduser().resolve(), "user_override", 1.0))
    if isinstance(semantic_value, (str, Path)):
        candidates.append((Path(semantic_value).expanduser().resolve(), "semantic_provider", 0.8))
    metadata_path = source.get("metadata_path")
    if isinstance(metadata_path, Path):
        candidates.append((metadata_path.parent / "affordances.json", "record_sidecar", 0.98))
    urdf = source["urdf"]
    assert isinstance(urdf, Path)
    candidates.append((urdf.parent / "affordances.json", "urdf_sidecar", 0.9))
    record_id = source.get("record_id")
    if isinstance(record_id, str) and record_id:
        candidates.append(
            (
                _PROJECT_ROOT / "assets" / "articraft" / record_id / "affordances.json",
                "checked_in_record_sidecar",
                0.98,
            )
        )
    for path, label, confidence in candidates:
        if path.is_file():
            return path.resolve(), Decision(
                str(path.resolve()),
                label,
                confidence,
                "existing affordance annotation selected",
                override_applied=label == "user_override",
            )
    if explicit is not None:
        raise AgentError(
            "affordances_missing",
            f"explicit affordances file does not exist: {explicit}",
            stage="semantic_inference",
            details={"path": str(explicit)},
        )
    return None, Decision(
        GENERATED_AFFORDANCES_NAME,
        "generated_empty_sidecar",
        0.45,
        "empty frames request deterministic URDF handle-geometry fallback",
    )


def _single_affordance_frame(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    frames = payload.get("frames") if isinstance(payload, dict) else None
    if isinstance(frames, dict) and len(frames) == 1:
        name = next(iter(frames))
        return name if isinstance(name, str) and name else None
    return None


def _semantic_affordances_sidecar(
    raw: Any,
    *,
    frame_name: str,
    asset_name: str,
    asset_source: str,
) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise AgentError(
            "semantic_result_invalid",
            "affordance_payload must be a mapping",
            stage="semantic_inference",
        )
    if isinstance(raw.get("frames"), Mapping):
        payload = copy.deepcopy(dict(raw))
        payload.setdefault("schema_version", 1)
        payload.setdefault("quaternion_order", "wxyz")
        payload.setdefault("asset_name", asset_name)
        payload.setdefault("asset_source", asset_source)
        return payload
    required = {
        "link",
        "position",
        "quaternion_wxyz",
        "gripper_closing_axis",
        "approach_axis",
        "recommended_gripper_width_m",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise AgentError(
            "semantic_result_invalid",
            "single-frame affordance_payload is incomplete",
            stage="semantic_inference",
            details={"missing": missing},
        )
    return {
        "schema_version": 1,
        "asset_name": asset_name,
        "asset_source": asset_source,
        "quaternion_order": "wxyz",
        "frames": {frame_name: copy.deepcopy(dict(raw))},
    }


def _decision_from_template(value: Any, field: str, confidence: float) -> Decision:
    return Decision(value, "template_default", confidence, f"copied from validated template field {field}")


def _load_template(path: Path) -> dict[str, Any]:
    return _read_json_object(path.expanduser().resolve(), stage="config_generation")


def _load_policy(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    policy = _read_json_object(resolved, stage="policy")
    if policy.get("schema_version") != 1:
        raise AgentError(
            "policy_invalid",
            "upper-agent policy schema_version must be exactly 1",
            stage="policy",
            details={"path": str(resolved), "schema_version": policy.get("schema_version")},
        )
    required_objects = ("workspace", "semantics", "task", "search")
    missing = [name for name in required_objects if not isinstance(policy.get(name), Mapping)]
    if missing:
        raise AgentError(
            "policy_invalid",
            "upper-agent policy is missing required objects",
            stage="policy",
            details={"path": str(resolved), "missing": missing},
        )
    base_config = policy.get("base_config")
    if not isinstance(base_config, str) or not base_config.strip():
        raise AgentError(
            "policy_invalid",
            "upper-agent policy base_config must be a path string",
            stage="policy",
            details={"path": str(resolved)},
        )
    anchor = policy["workspace"].get("closed_handle_world_position_m")
    offsets = policy["workspace"].get("kinematic_object_offset_candidates_m")
    if (
        not isinstance(anchor, list)
        or len(anchor) != 3
        or not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in anchor)
        or not isinstance(offsets, list)
        or not offsets
        or any(
            not isinstance(offset, list)
            or len(offset) != 3
            or not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in offset)
            for offset in offsets
        )
    ):
        raise AgentError(
            "policy_invalid",
            "workspace anchor and offset candidates must contain finite XYZ vectors",
            stage="policy",
            details={"path": str(resolved)},
        )
    max_attempts = policy["search"].get("max_kinematic_attempts")
    if (
        not isinstance(max_attempts, int)
        or isinstance(max_attempts, bool)
        or max_attempts < 1
    ):
        raise AgentError(
            "policy_invalid",
            "search.max_kinematic_attempts must be a positive integer",
            stage="policy",
            details={"path": str(resolved), "value": max_attempts},
        )
    return resolved, policy


def _policy_base_config(policy_path: Path, policy: Mapping[str, Any]) -> Path:
    raw = Path(str(policy["base_config"])).expanduser()
    return (raw if raw.is_absolute() else policy_path.parent / raw).resolve()


def _handle_local_position(
    *,
    urdf: Path,
    handle_link: str,
    affordances: Path | None,
    handle_frame: str,
    primitive_radial_samples: int,
) -> tuple[np.ndarray, str, float]:
    if affordances is not None:
        try:
            frame = load_affordances(affordances).frames.get(handle_frame)
        except Exception:
            frame = None
        if frame is not None:
            return np.asarray(frame.position, dtype=float), "affordance_frame", 0.98
    geometry = extract_handle_geometry(
        urdf,
        handle_link,
        primitive_radial_samples=primitive_radial_samples,
    )
    return np.asarray(geometry.aabb_center, dtype=float), "handle_geometry_aabb", 0.65


def _aligned_object_pose(
    *,
    model: URDFModel,
    door_joint: str,
    handle_link: str,
    closed_angle_rad: float,
    handle_local_position: np.ndarray,
    anchor: Sequence[float],
    orientation_wxyz: Sequence[float],
    offset: Sequence[float],
) -> dict[str, list[float]]:
    # The current policy intentionally uses identity orientation; retaining the
    # field in the contract lets a future provider replace this calculation.
    orientation = np.asarray(orientation_wxyz, dtype=float)
    if orientation.shape != (4,) or not np.allclose(orientation, [1.0, 0.0, 0.0, 0.0], atol=1.0e-8):
        raise AgentError(
            "policy_orientation_unsupported",
            "automatic workspace alignment currently requires identity object orientation",
            stage="config_generation",
            details={"orientation_wxyz": orientation.tolist()},
        )
    transforms = forward_kinematics(
        model,
        np.eye(4, dtype=float),
        {door_joint: float(closed_angle_rad)},
    )
    local_homogeneous = np.concatenate((np.asarray(handle_local_position, dtype=float), [1.0]))
    root_handle = transforms[handle_link] @ local_homogeneous
    position = np.asarray(anchor, dtype=float) - root_handle[:3] + np.asarray(offset, dtype=float)
    return {
        "position": [float(value) for value in position],
        "orientation_wxyz": [float(value) for value in orientation],
    }


def _write_failure_manifest(workspace: Path, failure: Mapping[str, Any], *, command: str) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    manifest_path = workspace / MANIFEST_NAME
    write_json(
        manifest_path,
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "agent": "agentpre.upper_agent",
            "command": command,
            "status": "failed",
            "success": False,
            "workspace": str(workspace),
            "failure": dict(failure),
        },
    )
    return manifest_path


def prepare_workspace(
    *,
    workspace: Path,
    urdf: Path | None = None,
    articraft_record: str | None = None,
    policy: Path = _DEFAULT_POLICY,
    template: Path | None = None,
    semantic_provider: SemanticProvider | None = None,
    semantic_provider_spec: str = "auto",
    door_joint: str | None = None,
    door_link: str | None = None,
    handle_link: str | None = None,
    handle_frame: str | None = None,
    affordances: Path | None = None,
    robot_urdf: Path | None = None,
    goal_angle_deg: float | None = None,
    force: bool = False,
) -> AgentOutcome:
    """Prepare a frozen low-level config and decision manifest; run nothing."""

    workspace = workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    manifest_path = workspace / MANIFEST_NAME
    config_path = workspace / CONFIG_NAME
    generated_affordances = workspace / GENERATED_AFFORDANCES_NAME
    occupied = [path for path in (manifest_path, config_path) if path.exists()]
    if occupied and not force:
        failure = AgentError(
            "workspace_not_empty",
            "prepared artifacts already exist; use --force to replace them",
            stage="workspace",
            details={"paths": [str(path) for path in occupied]},
        ).to_dict()
        return AgentOutcome(workspace, manifest_path, "failed", False, 2, failure)

    try:
        policy_path, policy_data = _load_policy(policy)
        template_path = template.expanduser().resolve() if template is not None else _policy_base_config(policy_path, policy_data)
        source = resolve_source(urdf=urdf, articraft_record=articraft_record)
        decisions, semantic_metadata = infer_semantic_decisions(
            source["urdf"],
            provider=semantic_provider,
            provider_spec=semantic_provider_spec,
        )
        decisions["door_joint"] = _override(decisions["door_joint"], door_joint)
        decisions["door_link"] = _override(decisions["door_link"], door_link)
        decisions["handle_link"] = _override(decisions["handle_link"], handle_link)

        affordance_path, affordance_decision = _discover_affordances(
            source,
            explicit=affordances,
            semantic_value=(decisions.get("affordances").value if "affordances" in decisions else None),
        )
        source_label = (
            f"articraft:{source.get('record_id') or source['supplied']}"
            if source["kind"] == "articraft_record"
            else f"urdf:{source['urdf']}"
        )
        policy_frame_name = str(policy_data["task"]["grasp_frame_name"])
        if affordance_path is None:
            semantic_sidecar = _semantic_affordances_sidecar(
                semantic_metadata.get("affordance_payload"),
                frame_name=policy_frame_name,
                asset_name="agentpre_auto_asset",
                asset_source=source_label,
            )
            if semantic_sidecar is not None:
                write_json(generated_affordances, semantic_sidecar)
                affordance_path = generated_affordances
                affordance_decision = Decision(
                    str(generated_affordances),
                    str(semantic_metadata["provider"]),
                    0.9,
                    "semantic provider generated an explicit handle frame",
                )
        decisions["affordances"] = affordance_decision
        inferred_handle_frame = decisions.get(
            "handle_frame",
            Decision("auto_handle", "geometry_fallback_request", 0.4, "request geometry fallback"),
        )
        if handle_frame is None and affordance_path is not None:
            only_frame = _single_affordance_frame(affordance_path)
            if only_frame is not None:
                inferred_handle_frame = Decision(
                    only_frame,
                    "single_affordance_frame",
                    0.98,
                    "the selected affordance sidecar declares exactly one frame",
                    previous_value=inferred_handle_frame.value,
                )
        if handle_frame is None and affordance_path == generated_affordances:
            semantic_frame = _single_affordance_frame(generated_affordances)
            if semantic_frame is not None:
                inferred_handle_frame = Decision(
                    semantic_frame,
                    str(semantic_metadata["provider"]),
                    0.9,
                    "semantic provider generated the selected handle frame",
                    previous_value=inferred_handle_frame.value,
                )
        decisions["handle_frame"] = _override(inferred_handle_frame, handle_frame)

        try:
            model = load_urdf(source["urdf"])
        except URDFModelError as exc:
            raise AgentError("urdf_invalid", str(exc), stage="asset_inspection", details=exc.to_dict()) from exc
        if affordance_path is None:
            selected_handle = str(decisions["handle_link"].value)
            selected_door = str(decisions["door_link"].value)
            if selected_handle == selected_door:
                raise AgentError(
                    "handle_frame_not_inferred",
                    "same-link door/handle assets require an explicit semantic affordance frame",
                    stage="semantic_inference",
                    details={
                        "door_link": selected_door,
                        "handle_link": selected_handle,
                        "hint": "install the default semantic provider or pass --affordances",
                    },
                )
            try:
                geometry = extract_handle_geometry(
                    source["urdf"],
                    selected_handle,
                    primitive_radial_samples=16,
                )
                candidate = generate_handle_candidates_from_geometry(
                    geometry,
                    CandidateGenerationConfig(
                        width_margin_m=0.01,
                        max_gripper_width_m=0.08,
                        max_candidates=1,
                        primitive_radial_samples=16,
                    ),
                )[0]
            except Exception as exc:
                details = exc.to_dict() if callable(getattr(exc, "to_dict", None)) else {"error": str(exc)}
                raise AgentError(
                    "handle_frame_not_inferred",
                    "independent handle geometry could not produce a grasp frame",
                    stage="semantic_inference",
                    details=details,
                ) from exc
            generated_frame_name = policy_frame_name
            write_json(
                generated_affordances,
                {
                    "schema_version": 1,
                    "asset_name": model.name,
                    "asset_source": source_label,
                    "quaternion_order": "wxyz",
                    "frames": {
                        generated_frame_name: {
                            "link": candidate.link_name,
                            "position": list(candidate.position),
                            "quaternion_wxyz": list(candidate.quaternion_wxyz),
                            "gripper_closing_axis": list(candidate.gripper_closing_axis),
                            "approach_axis": list(candidate.approach_axis),
                            "recommended_gripper_width_m": candidate.gripper_width_m,
                        }
                    },
                },
            )
            affordance_path = generated_affordances
            decisions["affordances"] = Decision(
                str(generated_affordances),
                "independent_handle_geometry",
                0.7,
                "generated only from a distinct handle link; never from aggregate door geometry",
            )
            if handle_frame is None:
                decisions["handle_frame"] = Decision(
                    generated_frame_name,
                    "independent_handle_geometry",
                    0.7,
                    "single generated PCA grasp frame",
                    previous_value=decisions["handle_frame"].value,
                )
        selected_joint = model.joints.get(str(decisions["door_joint"].value))
        if selected_joint is None or selected_joint.limit is None:
            raise AgentError(
                "door_joint_invalid",
                "selected door joint is absent or has no finite limits",
                stage="asset_inspection",
                details={"door_joint": decisions["door_joint"].value},
            )
        lower = selected_joint.limit.lower
        upper = selected_joint.limit.upper
        if lower is None or upper is None or not lower < upper:
            raise AgentError(
                "door_joint_invalid",
                "selected door joint must have ordered finite limits",
                stage="asset_inspection",
                details={"door_joint": selected_joint.name, "lower": lower, "upper": upper},
            )
        closed_decision = decisions.get(
            "closed_angle_deg",
            Decision(
                math.degrees(float(lower)),
                "door_joint_lower_limit",
                1.0,
                "closed state defaults to the URDF lower joint limit",
            ),
        )
        closed_deg = float(closed_decision.value)
        preferred_goal = float(policy_data["task"]["preferred_goal_angle_deg"])
        joint_margin = float(policy_data["task"]["joint_limit_margin_deg"])
        upper_with_margin = max(math.radians(closed_deg), float(upper) - math.radians(joint_margin))
        automatic_goal = math.degrees(
            min(upper_with_margin, math.radians(closed_deg + preferred_goal))
        )
        goal_decision = decisions.get(
            "goal_angle_deg",
            Decision(
                automatic_goal,
                "door_joint_limit_policy",
                0.95,
                "use the policy preferred angle without crossing the joint-limit margin",
            ),
        )
        goal_decision = _override(goal_decision, goal_angle_deg)
        selected_goal = float(goal_decision.value)
        lower_deg = math.degrees(float(lower))
        upper_deg = math.degrees(float(upper))
        if not lower_deg <= closed_deg <= upper_deg or not closed_deg <= selected_goal <= upper_deg:
            raise AgentError(
                "goal_outside_joint_limits",
                "closed or requested door angle lies outside the selected joint limits",
                stage="config_generation",
                details={
                    "closed_angle_deg": closed_deg,
                    "goal_angle_deg": selected_goal,
                    "lower_deg": lower_deg,
                    "upper_deg": upper_deg,
                },
            )
        decisions["closed_angle_deg"] = closed_decision
        decisions["goal_angle_deg"] = goal_decision

        report = inspect_asset(
            source["urdf"],
            door_joint_name=str(decisions["door_joint"].value),
            door_link_name=str(decisions["door_link"].value),
            handle_link_name=str(decisions["handle_link"].value),
        )
        if not report.ok:
            raise AgentError(
                "asset_inspection_failed",
                "URDF failed structural task inspection",
                stage="asset_inspection",
                details={"errors": [issue.to_dict() for issue in report.errors]},
            )

        config = _load_template(template_path)
        patch = semantic_metadata.get("config_patch")
        if patch is not None:
            if not isinstance(patch, Mapping):
                raise AgentError(
                    "semantic_result_invalid",
                    "semantic config_patch must be a mapping",
                    stage="config_generation",
                )
            _deep_merge(config, patch)
            decisions["config_patch"] = Decision(
                copy.deepcopy(patch),
                str(semantic_metadata["provider"]),
                0.7,
                "semantic provider patch applied before explicit CLI overrides",
            )

        config["project_root"] = str(_PROJECT_ROOT)
        object_config = config["assets"]["object"]
        object_config["name"] = model.name
        object_config["source"] = source_label
        object_config["urdf"] = str(source["urdf"])
        object_config["expected_urdf_sha256"] = source["urdf_sha256"]
        object_config["affordances"] = str(affordance_path or generated_affordances)
        object_config["door_joint"] = str(decisions["door_joint"].value)
        object_config["door_link"] = str(decisions["door_link"].value)
        object_config["handle_link"] = str(decisions["handle_link"].value)
        object_config["handle_frame"] = str(decisions["handle_frame"].value)
        old_door_link = str(config["assets"]["object"].get("door_link", "door"))
        allowed_contacts = config["collision"].get("allowed_contact_links", [])
        config["collision"]["allowed_contact_links"] = [
            str(decisions["door_link"].value)
            if str(name) in {old_door_link, "door"}
            else str(name)
            for name in allowed_contacts
        ]
        if robot_urdf is not None:
            resolved_robot = robot_urdf.expanduser().resolve()
            if not resolved_robot.is_file():
                raise AgentError(
                    "robot_urdf_missing",
                    f"robot URDF does not exist: {resolved_robot}",
                    stage="config_generation",
                    details={"path": str(resolved_robot)},
                )
            try:
                robot_model = load_urdf(resolved_robot)
            except URDFModelError as exc:
                raise AgentError(
                    "robot_urdf_invalid",
                    str(exc),
                    stage="config_generation",
                    details=exc.to_dict(),
                ) from exc
            previous_robot_urdf = config["assets"]["robot"].get("urdf")
            config["assets"]["robot"]["name"] = robot_model.name
            config["assets"]["robot"]["urdf"] = str(resolved_robot)
            config["assets"]["robot"]["expected_urdf_sha256"] = _sha256(resolved_robot)
            decisions["robot_urdf"] = Decision(
                str(resolved_robot),
                "user_override",
                1.0,
                "explicit robot URDF with recomputed identity hash",
                previous_value=previous_robot_urdf,
                override_applied=True,
            )
        if "object_world_pose" not in decisions:
            workspace_policy = policy_data["workspace"]
            offsets = workspace_policy["kinematic_object_offset_candidates_m"]
            handle_local, alignment_source, alignment_confidence = _handle_local_position(
                urdf=source["urdf"],
                handle_link=str(decisions["handle_link"].value),
                affordances=affordance_path,
                handle_frame=str(decisions["handle_frame"].value),
                primitive_radial_samples=int(config["affordance_generation"]["primitive_radial_samples"]),
            )
            aligned_pose = _aligned_object_pose(
                model=model,
                door_joint=str(decisions["door_joint"].value),
                handle_link=str(decisions["handle_link"].value),
                closed_angle_rad=math.radians(closed_deg),
                handle_local_position=handle_local,
                anchor=workspace_policy["closed_handle_world_position_m"],
                orientation_wxyz=workspace_policy["object_orientation_wxyz"],
                offset=offsets[0],
            )
            decisions["object_world_pose"] = Decision(
                aligned_pose,
                f"workspace_anchor+{alignment_source}",
                alignment_confidence,
                "align the closed handle with the policy workspace anchor using the first search offset",
            )
        config["task"]["closed_angle_deg"] = closed_deg
        config["task"]["goal_angle_deg"] = selected_goal
        config["output"]["root"] = str(workspace / "runs")

        decisions.setdefault(
            "object_world_pose",
            _decision_from_template(config["assets"]["object"]["world_pose"], "assets.object.world_pose", 0.45),
        )
        decisions.setdefault(
            "robot_world_pose",
            _decision_from_template(config["assets"]["robot"]["world_pose"], "assets.robot.world_pose", 0.9),
        )
        decisions.setdefault(
            "grasp_offset",
            _decision_from_template(config["task"]["grasp_offset"], "task.grasp_offset", 0.9),
        )
        decisions["phase_profile"] = _decision_from_template(
            config["task"]["phases"], "task.phases", 0.9
        )
        decisions["controller"] = _decision_from_template(
            config["simulation"], "simulation", 0.9
        )

        # A provider may express these high-level values either as decisions or
        # as a config patch.  Decisions win so the manifest and config agree.
        for name, dotted in (
            ("object_world_pose", ("assets", "object", "world_pose")),
            ("robot_world_pose", ("assets", "robot", "world_pose")),
            ("grasp_offset", ("task", "grasp_offset")),
        ):
            if name in decisions and decisions[name].source != "template_default":
                target: MutableMapping[str, Any] = config
                for key in dotted[:-1]:
                    target = target[key]
                target[dotted[-1]] = copy.deepcopy(decisions[name].value)

        try:
            validate_config(config)
        except Exception as exc:
            details = exc.to_dict() if callable(getattr(exc, "to_dict", None)) else {"error": str(exc)}
            raise AgentError(
                "generated_config_invalid",
                "automatic decisions did not produce a valid low-level config",
                stage="config_generation",
                details=details,
            ) from exc

        assert affordance_path is not None
        write_json(config_path, config)

        low_confidence = sorted(
            name for name, decision in decisions.items() if decision.confidence < 0.6
        )
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "agent": "agentpre.upper_agent",
            "command": "prepare",
            "status": "prepared",
            "success": True,
            "workspace": str(workspace),
            "policy": {
                "path": str(policy_path),
                "sha256": _sha256(policy_path),
                "base_config": str(template_path),
                "base_config_sha256": _sha256(template_path),
                "workspace_anchor_m": copy.deepcopy(
                    policy_data["workspace"]["closed_handle_world_position_m"]
                ),
                "kinematic_object_offset_candidates_m": copy.deepcopy(
                    policy_data["workspace"]["kinematic_object_offset_candidates_m"]
                ),
                "search": copy.deepcopy(policy_data["search"]),
            },
            "input": {
                "kind": source["kind"],
                "supplied": source["supplied"],
                "record_id": source.get("record_id"),
                "metadata_path": str(source["metadata_path"]) if source.get("metadata_path") else None,
                "resolved_urdf": str(source["urdf"]),
                "urdf_sha256": source["urdf_sha256"],
            },
            "semantic_inference": {
                "provider": semantic_metadata["provider"],
                "decisions": {name: decision.to_dict() for name, decision in sorted(decisions.items())},
                "low_confidence_decisions": low_confidence,
                "review_required": bool(low_confidence),
                "evidence": semantic_metadata.get("evidence", []),
                "warnings": semantic_metadata.get("warnings", []),
            },
            "checks": {"asset_inspection": report.to_dict(), "config_valid": True},
            "readiness": {"ready_for_execution": True, "blockers": []},
            "artifacts": {
                "config": {"path": str(config_path), "sha256": _sha256(config_path)},
                "affordances": {"path": str(affordance_path), "sha256": _sha256(affordance_path)},
                "manifest": {"path": str(manifest_path)},
                "policy": {"path": str(policy_path), "sha256": _sha256(policy_path)},
                "run_root": str(workspace / "runs"),
            },
            "executions": [],
            "failure": None,
        }
        write_json(manifest_path, manifest)
        return AgentOutcome(workspace, manifest_path, "prepared", True, 0)
    except AgentError as exc:
        failure = exc.to_dict()
    except Exception as exc:
        failure = AgentError(
            "unexpected_error",
            str(exc),
            stage="prepare",
            details={"exception_type": type(exc).__name__},
            recoverable=False,
        ).to_dict()
    try:
        _write_failure_manifest(workspace, failure, command="prepare")
    except Exception:
        pass
    return AgentOutcome(workspace, manifest_path, "failed", False, 2, failure)


def _default_runner(argv: Sequence[str]) -> int:
    from .run import main as run_main

    return int(run_main(list(argv)))


def _export_animation_artifact(output_dir: Path) -> dict[str, Any] | None:
    """Export a successful rollout when its trajectory artifact is present."""

    trajectory_path = output_dir / "trajectory.npz"
    if not trajectory_path.is_file():
        return None
    animation_path = output_dir / "animation.html"
    try:
        write_animation_html(trajectory_path, animation_path)
    except Exception as exc:
        return {
            "status": "failed",
            "trajectory_path": str(trajectory_path),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    return {
        "status": "succeeded",
        "path": str(animation_path),
        "sha256": _sha256(animation_path),
        "trajectory_path": str(trajectory_path),
        "trajectory_sha256": _sha256(trajectory_path),
    }


def execute_workspace(
    *,
    workspace: Path,
    mode: str,
    output_dir: Path | None = None,
    runner: Runner | None = None,
) -> AgentOutcome:
    """Run one prepared config without changing or re-inferring its decisions."""

    workspace = workspace.expanduser().resolve()
    manifest_path = workspace / MANIFEST_NAME
    try:
        manifest = _read_json_object(manifest_path, stage="execute_preflight")
        readiness = manifest.get("readiness")
        if not isinstance(readiness, Mapping) or readiness.get("ready_for_execution") is not True:
            raise AgentError(
                "workspace_not_prepared",
                "manifest is not marked ready for execution",
                stage="execute_preflight",
                details={"manifest": str(manifest_path)},
            )
        config_artifact = manifest.get("artifacts", {}).get("config", {})
        config_path = Path(str(config_artifact.get("path", ""))).expanduser().resolve()
        if not config_path.is_file():
            raise AgentError(
                "prepared_config_missing",
                "prepared config file is missing",
                stage="execute_preflight",
                details={"path": str(config_path)},
            )
        expected_hash = config_artifact.get("sha256")
        actual_hash = _sha256(config_path)
        if expected_hash != actual_hash:
            raise AgentError(
                "prepared_config_changed",
                "prepared config hash no longer matches the manifest",
                stage="execute_preflight",
                details={"expected_sha256": expected_hash, "observed_sha256": actual_hash},
            )
        if mode not in {"kinematic", "physics_assisted"}:
            raise AgentError(
                "mode_invalid",
                "mode must be kinematic or physics_assisted",
                stage="execute_preflight",
                details={"mode": mode},
            )
        destination = (output_dir or workspace / "runs" / mode).expanduser().resolve()
        attempt = {
            "mode": mode,
            "config_sha256": actual_hash,
            "output_dir": str(destination),
            "status": "running",
            "exit_code": None,
            "success": False,
            "failure": None,
            "animation": None,
        }
        executions = manifest.setdefault("executions", [])
        if not isinstance(executions, list):
            raise AgentError(
                "manifest_invalid",
                "manifest executions field must be a list",
                stage="execute_preflight",
            )
        executions.append(attempt)
        manifest["command"] = "execute"
        manifest["status"] = "running"
        manifest["success"] = False
        write_json(manifest_path, manifest)

        selected_runner = runner or _default_runner
        try:
            exit_code = int(
                selected_runner(
                    ["--config", str(config_path), "--mode", mode, "--output-dir", str(destination)]
                )
            )
        except Exception as exc:
            raise AgentError(
                "runner_exception",
                "low-level runner raised an exception",
                stage="execution",
                details={"exception_type": type(exc).__name__, "error": str(exc)},
                recoverable=False,
            ) from exc

        metrics_path = destination / "metrics.json"
        metrics: dict[str, Any] | None = None
        if metrics_path.is_file():
            try:
                metrics = _read_json_object(metrics_path, stage="execution_result")
            except AgentError:
                metrics = None
        run_success = exit_code == 0 and metrics is not None and metrics.get("success") is True
        failure: Mapping[str, Any] | None = None
        if not run_success:
            if metrics is not None and isinstance(metrics.get("failure"), Mapping):
                failure = dict(metrics["failure"])
            else:
                failure = AgentError(
                    "execution_failed",
                    "low-level run did not produce successful metrics",
                    stage="execution",
                    details={
                        "runner_exit_code": exit_code,
                        "metrics_path": str(metrics_path),
                        "metrics_present": metrics is not None,
                    },
                ).to_dict()
        attempt.update(
            {
                "status": "succeeded" if run_success else "failed",
                "exit_code": exit_code,
                "success": run_success,
                "metrics": str(metrics_path) if metrics is not None else None,
                "failure": failure,
                "animation": _export_animation_artifact(destination) if run_success else None,
            }
        )
        manifest["status"] = attempt["status"]
        manifest["success"] = run_success
        manifest["failure"] = failure
        write_json(manifest_path, manifest)
        return AgentOutcome(
            workspace,
            manifest_path,
            str(attempt["status"]),
            run_success,
            0 if run_success else (exit_code if exit_code else 2),
            failure,
        )
    except AgentError as exc:
        failure = exc.to_dict()
    except Exception as exc:
        failure = AgentError(
            "unexpected_error",
            str(exc),
            stage="execute",
            details={"exception_type": type(exc).__name__},
            recoverable=False,
        ).to_dict()
    try:
        existing = _read_json_object(manifest_path, stage="execute_failure") if manifest_path.is_file() else {}
        existing.update({"command": "execute", "status": "failed", "success": False, "failure": failure})
        write_json(manifest_path, existing)
    except Exception:
        try:
            _write_failure_manifest(workspace, failure, command="execute")
        except Exception:
            pass
    return AgentOutcome(workspace, manifest_path, "failed", False, 2, failure)


def _run_attempt(
    *,
    config_path: Path,
    mode: str,
    output_dir: Path,
    runner: Runner,
) -> tuple[int, bool, dict[str, Any] | None, Mapping[str, Any] | None]:
    """Execute one low-level attempt and normalize its result contract."""

    try:
        exit_code = int(
            runner(
                [
                    "--config",
                    str(config_path),
                    "--mode",
                    mode,
                    "--output-dir",
                    str(output_dir),
                ]
            )
        )
    except Exception as exc:
        failure = AgentError(
            "runner_exception",
            "low-level runner raised an exception",
            stage="execution",
            details={"exception_type": type(exc).__name__, "error": str(exc)},
            recoverable=False,
        ).to_dict()
        return 2, False, None, failure
    metrics_path = output_dir / "metrics.json"
    metrics: dict[str, Any] | None = None
    if metrics_path.is_file():
        try:
            metrics = _read_json_object(metrics_path, stage="execution_result")
        except AgentError:
            metrics = None
    success = exit_code == 0 and metrics is not None and metrics.get("success") is True
    failure: Mapping[str, Any] | None = None
    if not success:
        if metrics is not None and isinstance(metrics.get("failure"), Mapping):
            failure = dict(metrics["failure"])
        else:
            failure = AgentError(
                "execution_failed",
                "low-level run did not produce successful metrics",
                stage="execution",
                details={
                    "runner_exit_code": exit_code,
                    "metrics_path": str(metrics_path),
                    "metrics_present": metrics is not None,
                },
            ).to_dict()
    return exit_code, success, metrics, failure


def _workspace_retryable_failure(failure: Mapping[str, Any] | None) -> bool:
    """Return whether changing only the object offset can plausibly help."""

    if not isinstance(failure, Mapping):
        return False
    return failure.get("code") in {
        "ik_unreachable",
        "collision",
        "joint_limit_violation",
        "acceptance_failed",
    }


def run_agent(
    *,
    workspace: Path,
    urdf: Path | None = None,
    articraft_record: str | None = None,
    policy: Path = _DEFAULT_POLICY,
    template: Path | None = None,
    semantic_provider: SemanticProvider | None = None,
    semantic_provider_spec: str = "auto",
    door_joint: str | None = None,
    door_link: str | None = None,
    handle_link: str | None = None,
    handle_frame: str | None = None,
    affordances: Path | None = None,
    robot_urdf: Path | None = None,
    goal_angle_deg: float | None = None,
    force: bool = False,
    with_physics: bool = False,
    runner: Runner | None = None,
) -> AgentOutcome:
    """Prepare, search cheap kinematic offsets, then optionally run physics."""

    prepared = prepare_workspace(
        workspace=workspace,
        urdf=urdf,
        articraft_record=articraft_record,
        policy=policy,
        template=template,
        semantic_provider=semantic_provider,
        semantic_provider_spec=semantic_provider_spec,
        door_joint=door_joint,
        door_link=door_link,
        handle_link=handle_link,
        handle_frame=handle_frame,
        affordances=affordances,
        robot_urdf=robot_urdf,
        goal_angle_deg=goal_angle_deg,
        force=force,
    )
    if not prepared.success:
        return prepared

    workspace = prepared.workspace
    manifest_path = prepared.manifest_path
    selected_runner = runner or _default_runner
    try:
        manifest = _read_json_object(manifest_path, stage="search_preflight")
        base_artifact = manifest["artifacts"]["config"]
        base_config_path = Path(str(base_artifact["path"])).resolve()
        if _sha256(base_config_path) != base_artifact["sha256"]:
            raise AgentError(
                "prepared_config_changed",
                "prepared config changed before search started",
                stage="search_preflight",
            )
        base_config = _read_json_object(base_config_path, stage="search_preflight")
        policy_snapshot = manifest["policy"]
        offsets = policy_snapshot["kinematic_object_offset_candidates_m"]
        max_attempts = int(policy_snapshot["search"]["max_kinematic_attempts"])
        attempts_to_run = min(max_attempts, len(offsets))
        first_offset = np.asarray(offsets[0], dtype=float)
        base_position = np.asarray(
            base_config["assets"]["object"]["world_pose"]["position"], dtype=float
        )

        search = {
            "status": "running",
            "max_attempts": max_attempts,
            "candidate_count": len(offsets),
            "attempts": [],
            "selected_attempt": None,
            "physics_requested": bool(with_physics),
            "physics": None,
        }
        manifest["command"] = "run"
        manifest["status"] = "running"
        manifest["success"] = False
        manifest["search"] = search
        write_json(manifest_path, manifest)

        selected: dict[str, Any] | None = None
        for index, raw_offset in enumerate(offsets[:attempts_to_run]):
            offset = np.asarray(raw_offset, dtype=float)
            attempt_id = f"kinematic_{index:03d}"
            attempt_root = workspace / "attempts" / attempt_id
            attempt_config_path = attempt_root / "config.json"
            output_dir = attempt_root / "output"
            attempt_config = copy.deepcopy(base_config)
            position = base_position + offset - first_offset
            attempt_config["assets"]["object"]["world_pose"]["position"] = [
                float(value) for value in position
            ]
            attempt_config["output"]["root"] = str(attempt_root)
            validate_config(attempt_config)
            write_json(attempt_config_path, attempt_config)
            attempt = {
                "attempt_id": attempt_id,
                "mode": "kinematic",
                "object_offset_m": [float(value) for value in offset],
                "decision": {
                    "source": "policy.workspace.kinematic_object_offset_candidates_m",
                    "confidence": 0.8,
                    "override": {"applied": False, "previous_value": None},
                },
                "config": {
                    "path": str(attempt_config_path),
                    "sha256": _sha256(attempt_config_path),
                },
                "output_dir": str(output_dir),
                "status": "running",
                "success": False,
                "exit_code": None,
                "metrics": None,
                "failure": None,
                "animation": None,
            }
            search["attempts"].append(attempt)
            write_json(manifest_path, manifest)
            exit_code, success, metrics, failure = _run_attempt(
                config_path=attempt_config_path,
                mode="kinematic",
                output_dir=output_dir,
                runner=selected_runner,
            )
            metrics_path = output_dir / "metrics.json"
            attempt.update(
                {
                    "status": "succeeded" if success else "failed",
                    "success": success,
                    "exit_code": exit_code,
                    "metrics": (
                        {"path": str(metrics_path), "sha256": _sha256(metrics_path)}
                        if metrics is not None
                        else None
                    ),
                    "failure": failure,
                    "animation": _export_animation_artifact(output_dir) if success else None,
                }
            )
            write_json(manifest_path, manifest)
            if success:
                selected = attempt
                search["selected_attempt"] = attempt_id
                break
            if not _workspace_retryable_failure(failure):
                search["status"] = "failed"
                search["aborted_nonretryable"] = True
                manifest.update(
                    {"status": "failed", "success": False, "failure": failure}
                )
                write_json(manifest_path, manifest)
                return AgentOutcome(
                    workspace,
                    manifest_path,
                    "failed",
                    False,
                    exit_code if exit_code else 2,
                    failure,
                )

        if selected is None:
            failure = AgentError(
                "kinematic_search_exhausted",
                "no policy workspace offset produced a successful kinematic run",
                stage="kinematic_search",
                details={
                    "attempted": len(search["attempts"]),
                    "maximum": max_attempts,
                    "failures": [attempt["failure"] for attempt in search["attempts"]],
                },
            ).to_dict()
            search["status"] = "failed"
            manifest.update({"status": "failed", "success": False, "failure": failure})
            write_json(manifest_path, manifest)
            return AgentOutcome(workspace, manifest_path, "failed", False, 2, failure)

        selected_config_path = Path(selected["config"]["path"])
        manifest["artifacts"]["selected_config"] = copy.deepcopy(selected["config"])
        if with_physics:
            physics_root = workspace / "attempts" / str(selected["attempt_id"]) / "physics_assisted"
            physics = {
                "mode": "physics_assisted",
                "config": copy.deepcopy(selected["config"]),
                "output_dir": str(physics_root),
                "status": "running",
                "success": False,
                "exit_code": None,
                "metrics": None,
                "failure": None,
                "animation": None,
            }
            search["physics"] = physics
            write_json(manifest_path, manifest)
            exit_code, success, metrics, failure = _run_attempt(
                config_path=selected_config_path,
                mode="physics_assisted",
                output_dir=physics_root,
                runner=selected_runner,
            )
            metrics_path = physics_root / "metrics.json"
            physics.update(
                {
                    "status": "succeeded" if success else "failed",
                    "success": success,
                    "exit_code": exit_code,
                    "metrics": (
                        {"path": str(metrics_path), "sha256": _sha256(metrics_path)}
                        if metrics is not None
                        else None
                    ),
                    "failure": failure,
                    "animation": _export_animation_artifact(physics_root) if success else None,
                }
            )
            if not success:
                search["status"] = "failed"
                manifest.update({"status": "failed", "success": False, "failure": failure})
                write_json(manifest_path, manifest)
                return AgentOutcome(
                    workspace,
                    manifest_path,
                    "failed",
                    False,
                    exit_code if exit_code else 2,
                    failure,
                )

        final_animation = (
            search["physics"]["animation"] if with_physics else selected["animation"]
        )
        if final_animation is not None:
            manifest["artifacts"]["animation"] = copy.deepcopy(final_animation)
        search["status"] = "succeeded"
        manifest.update({"status": "succeeded", "success": True, "failure": None})
        write_json(manifest_path, manifest)
        return AgentOutcome(workspace, manifest_path, "succeeded", True, 0)
    except AgentError as exc:
        failure = exc.to_dict()
    except Exception as exc:
        failure = AgentError(
            "unexpected_error",
            str(exc),
            stage="run",
            details={"exception_type": type(exc).__name__},
            recoverable=False,
        ).to_dict()
    try:
        manifest = _read_json_object(manifest_path, stage="run_failure")
        manifest.update({"command": "run", "status": "failed", "success": False, "failure": failure})
        write_json(manifest_path, manifest)
    except Exception:
        pass
    return AgentOutcome(workspace, manifest_path, "failed", False, 2, failure)


def _add_prepare_arguments(command: argparse.ArgumentParser) -> None:
    source = command.add_mutually_exclusive_group(required=True)
    source.add_argument("--urdf", type=Path)
    source.add_argument("--articraft-record")
    command.add_argument("--workdir", required=True, type=Path)
    command.add_argument("--policy", type=Path, default=_DEFAULT_POLICY)
    command.add_argument(
        "--template",
        type=Path,
        help="optional base-config override; policy base_config is used by default",
    )
    command.add_argument("--semantic-provider", default="auto", metavar="auto|local|MODULE:FUNCTION")
    command.add_argument("--door-joint")
    command.add_argument("--door-link")
    command.add_argument("--handle-link")
    command.add_argument("--handle-frame")
    command.add_argument("--affordances", type=Path)
    command.add_argument("--robot-urdf", type=Path)
    command.add_argument("--goal-angle-deg", type=float)
    command.add_argument("--force", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="infer semantics and write config/manifest only")
    _add_prepare_arguments(prepare)
    prepare.add_argument(
        "--dry-run",
        action="store_true",
        help="compatibility flag; prepare is always a non-executing dry run",
    )

    run = subparsers.add_parser(
        "run",
        help="prepare, search policy kinematic offsets, and optionally run physics",
    )
    _add_prepare_arguments(run)
    run.add_argument(
        "--with-physics",
        action="store_true",
        help="run physics_assisted once after the first successful kinematic attempt",
    )

    execute = subparsers.add_parser("execute", help="run a frozen prepared configuration")
    execute.add_argument("--workdir", required=True, type=Path)
    execute.add_argument("--mode", choices=("kinematic", "physics_assisted"), required=True)
    execute.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        outcome = prepare_workspace(
            workspace=args.workdir,
            urdf=args.urdf,
            articraft_record=args.articraft_record,
            policy=args.policy,
            template=args.template,
            semantic_provider_spec=args.semantic_provider,
            door_joint=args.door_joint,
            door_link=args.door_link,
            handle_link=args.handle_link,
            handle_frame=args.handle_frame,
            affordances=args.affordances,
            robot_urdf=args.robot_urdf,
            goal_angle_deg=args.goal_angle_deg,
            force=args.force,
        )
    elif args.command == "execute":
        outcome = execute_workspace(
            workspace=args.workdir,
            mode=args.mode,
            output_dir=args.output_dir,
        )
    else:
        outcome = run_agent(
            workspace=args.workdir,
            urdf=args.urdf,
            articraft_record=args.articraft_record,
            policy=args.policy,
            template=args.template,
            semantic_provider_spec=args.semantic_provider,
            door_joint=args.door_joint,
            door_link=args.door_link,
            handle_link=args.handle_link,
            handle_frame=args.handle_frame,
            affordances=args.affordances,
            robot_urdf=args.robot_urdf,
            goal_angle_deg=args.goal_angle_deg,
            force=args.force,
            with_physics=args.with_physics,
        )
    stream = sys.stdout if outcome.success else sys.stderr
    print(json.dumps(outcome.summary(), ensure_ascii=False, sort_keys=True), file=stream)
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
