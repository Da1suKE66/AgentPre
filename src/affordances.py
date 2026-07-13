"""Affordance frames and deterministic geometry-derived grasp candidates.

All poses in this module are expressed in the named URDF link frame and all
quaternions use ``wxyz`` order.  The module deliberately operates on link and
frame names; simulator body indices are never accepted or exposed.

When an authored affordance frame is unavailable, callers may explicitly opt
into a deterministic geometry fallback by supplying a
:class:`CandidateGenerationConfig`.  There are no hidden gripper widths,
clearances, candidate limits, or primitive sampling settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence, TypeAlias
import xml.etree.ElementTree as ET

import numpy as np

from .transforms import matrix_to_quaternion, pose_matrix, rpy_to_matrix
from .urdf_model import MeshReference, resolve_mesh_path


_NUMERICAL_EPS = 1.0e-12
_UNIT_TOLERANCE = 1.0e-4
_ORTHOGONAL_TOLERANCE = 1.0e-6


class AffordanceError(ValueError):
    """A machine-readable affordance or geometry validation failure."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": dict(self.context)}


@dataclass(frozen=True, slots=True)
class FailureReason:
    """A non-exception failure retained during resolution or selection."""

    code: str
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": dict(self.context)}


@dataclass(frozen=True, slots=True)
class HandleFrame:
    """One authored handle frame, relative to ``link_name``."""

    name: str
    link_name: str
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    gripper_closing_axis: tuple[float, float, float]
    approach_axis: tuple[float, float, float]
    recommended_gripper_width_m: float

    @property
    def transform(self) -> np.ndarray:
        """Return the local handle-frame transform."""

        return pose_matrix(self.position, self.quaternion_wxyz)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "link": self.link_name,
            "position": list(self.position),
            "quaternion_wxyz": list(self.quaternion_wxyz),
            "gripper_closing_axis": list(self.gripper_closing_axis),
            "approach_axis": list(self.approach_axis),
            "recommended_gripper_width_m": self.recommended_gripper_width_m,
        }


@dataclass(frozen=True, slots=True)
class AffordanceSet:
    """Validated contents of one ``affordances.json`` file."""

    path: Path
    schema_version: int
    quaternion_order: str
    frames: Mapping[str, HandleFrame]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def require_frame(self, name: str) -> HandleFrame:
        try:
            return self.frames[name]
        except KeyError as exc:
            raise AffordanceError(
                "FRAME_NOT_FOUND",
                f"Affordance frame not found: {name}",
                frame=name,
                available_frames=sorted(self.frames),
                path=str(self.path),
            ) from exc


@dataclass(frozen=True, slots=True)
class CandidateGenerationConfig:
    """Caller-owned parameters for geometry fallback candidate generation.

    ``width_margin_m`` is added once to the measured total handle span.  It is
    therefore a total opening clearance, not a per-finger clearance.
    """

    width_margin_m: float
    max_gripper_width_m: float
    max_candidates: int
    primitive_radial_samples: int

    def __post_init__(self) -> None:
        for name in ("width_margin_m", "max_gripper_width_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise AffordanceError(
                    "CANDIDATE_CONFIG_INVALID",
                    f"{name} must be numeric",
                    field=name,
                    value=value,
                )
            if not math.isfinite(float(value)):
                raise AffordanceError(
                    "CANDIDATE_CONFIG_INVALID",
                    f"{name} must be finite",
                    field=name,
                    value=value,
                )
        if self.width_margin_m < 0.0:
            raise AffordanceError(
                "CANDIDATE_CONFIG_INVALID",
                "width_margin_m must be non-negative",
                field="width_margin_m",
                value=self.width_margin_m,
            )
        if self.max_gripper_width_m <= 0.0:
            raise AffordanceError(
                "CANDIDATE_CONFIG_INVALID",
                "max_gripper_width_m must be positive",
                field="max_gripper_width_m",
                value=self.max_gripper_width_m,
            )
        if (
            isinstance(self.max_candidates, bool)
            or not isinstance(self.max_candidates, int)
            or self.max_candidates < 1
        ):
            raise AffordanceError(
                "CANDIDATE_CONFIG_INVALID",
                "max_candidates must be a positive integer",
                field="max_candidates",
                value=self.max_candidates,
            )
        if (
            isinstance(self.primitive_radial_samples, bool)
            or not isinstance(self.primitive_radial_samples, int)
            or self.primitive_radial_samples < 4
        ):
            raise AffordanceError(
                "CANDIDATE_CONFIG_INVALID",
                "primitive_radial_samples must be an integer of at least 4",
                field="primitive_radial_samples",
                value=self.primitive_radial_samples,
            )


@dataclass(frozen=True, slots=True)
class HandleGeometry:
    """Local handle geometry and deterministic AABB/PCA features."""

    link_name: str
    vertices_local: np.ndarray = field(repr=False)
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    aabb_center: np.ndarray
    principal_axes: np.ndarray
    principal_extents: np.ndarray
    sources: tuple[str, ...]

    def __post_init__(self) -> None:
        arrays_and_shapes = (
            ("vertices_local", self.vertices_local, (None, 3)),
            ("aabb_min", self.aabb_min, (3,)),
            ("aabb_max", self.aabb_max, (3,)),
            ("aabb_center", self.aabb_center, (3,)),
            ("principal_axes", self.principal_axes, (3, 3)),
            ("principal_extents", self.principal_extents, (3,)),
        )
        for name, value, shape in arrays_and_shapes:
            array = np.asarray(value, dtype=float)
            valid_shape = array.ndim == 2 and array.shape[1] == 3 if shape == (None, 3) else array.shape == shape
            if not valid_shape or not np.isfinite(array).all():
                raise AffordanceError(
                    "GEOMETRY_FEATURE_INVALID",
                    f"{name} has an invalid shape or non-finite values",
                    field=name,
                    shape=array.shape,
                )
            array = array.copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def to_dict(self) -> dict[str, Any]:
        return {
            "link": self.link_name,
            "vertex_count": int(self.vertices_local.shape[0]),
            "aabb_min": self.aabb_min.tolist(),
            "aabb_max": self.aabb_max.tolist(),
            "aabb_center": self.aabb_center.tolist(),
            "principal_axes": self.principal_axes.tolist(),
            "principal_extents": self.principal_extents.tolist(),
            "sources": list(self.sources),
        }


@dataclass(frozen=True, slots=True)
class GraspCandidate:
    """One handle-relative gripper target."""

    candidate_id: str
    rank: int
    link_name: str
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    gripper_closing_axis: tuple[float, float, float]
    approach_axis: tuple[float, float, float]
    gripper_width_m: float
    source: str

    @property
    def transform(self) -> np.ndarray:
        return pose_matrix(self.position, self.quaternion_wxyz)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "rank": self.rank,
            "link": self.link_name,
            "position": list(self.position),
            "quaternion_wxyz": list(self.quaternion_wxyz),
            "gripper_closing_axis": list(self.gripper_closing_axis),
            "approach_axis": list(self.approach_axis),
            "gripper_width_m": self.gripper_width_m,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class CandidateResolution:
    """Result of resolving an authored frame or geometry fallback."""

    requested_frame: str
    used_geometry_fallback: bool
    candidates: tuple[GraspCandidate, ...]
    reasons: tuple[FailureReason, ...] = ()
    geometry: HandleGeometry | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_frame": self.requested_frame,
            "used_geometry_fallback": self.used_geometry_fallback,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "reasons": [reason.to_dict() for reason in self.reasons],
            "geometry": self.geometry.to_dict() if self.geometry is not None else None,
        }


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Optional structured result returned by a candidate-check callback."""

    passed: bool
    reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


CheckReturn: TypeAlias = bool | CheckResult | tuple[bool, str]
CandidateCheck: TypeAlias = Callable[[GraspCandidate], CheckReturn]


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    candidate: GraspCandidate
    reachable: bool
    collision_free: bool | None
    accepted: bool
    reasons: tuple[FailureReason, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "reachable": self.reachable,
            "collision_free": self.collision_free,
            "accepted": self.accepted,
            "reasons": [reason.to_dict() for reason in self.reasons],
        }


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    selected: GraspCandidate | None
    evaluations: tuple[CandidateEvaluation, ...]

    @property
    def ok(self) -> bool:
        return self.selected is not None

    @property
    def failure_reasons(self) -> tuple[FailureReason, ...]:
        return tuple(reason for evaluation in self.evaluations for reason in evaluation.reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "selected": self.selected.to_dict() if self.selected is not None else None,
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
        }


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise AffordanceError(
            "INVALID_NUMBER", f"{field_name} must be numeric", field=field_name, value=value
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AffordanceError(
            "INVALID_NUMBER", f"{field_name} must be numeric", field=field_name, value=value
        ) from exc
    if not math.isfinite(number):
        raise AffordanceError(
            "NONFINITE_NUMBER",
            f"{field_name} must be finite",
            field=field_name,
            value=value,
        )
    return number


def _finite_vector(value: Any, length: int, field_name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise AffordanceError(
            "INVALID_VECTOR",
            f"{field_name} must contain exactly {length} values",
            field=field_name,
            value=value,
        )
    vector = np.asarray(
        [_finite_number(component, f"{field_name}[{index}]") for index, component in enumerate(value)],
        dtype=float,
    )
    return vector


def _unit_vector(value: Any, field_name: str) -> np.ndarray:
    vector = _finite_vector(value, 3, field_name)
    norm = float(np.linalg.norm(vector))
    if norm <= _NUMERICAL_EPS or abs(norm - 1.0) > _UNIT_TOLERANCE:
        raise AffordanceError(
            "AXIS_NOT_NORMALIZED",
            f"{field_name} must be a normalized, non-zero axis",
            field=field_name,
            norm=norm,
        )
    return vector / norm


def _quaternion_wxyz(value: Any, field_name: str) -> np.ndarray:
    quaternion = _finite_vector(value, 4, field_name)
    norm = float(np.linalg.norm(quaternion))
    if norm <= _NUMERICAL_EPS or abs(norm - 1.0) > _UNIT_TOLERANCE:
        raise AffordanceError(
            "QUATERNION_NOT_NORMALIZED",
            f"{field_name} must be a normalized, non-zero wxyz quaternion",
            field=field_name,
            norm=norm,
        )
    quaternion = quaternion / norm
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return quaternion


def _required_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise AffordanceError(
            "INVALID_OBJECT", f"{field_name} must be an object", field=field_name, value=value
        )
    return value


def _parse_handle_frame(name: str, raw: Any) -> HandleFrame:
    field_root = f"frames.{name}"
    data = _required_mapping(raw, field_root)
    link_name = data.get("link")
    if not isinstance(link_name, str) or not link_name.strip():
        raise AffordanceError(
            "FRAME_LINK_INVALID",
            f"{field_root}.link must be a non-empty link name",
            field=f"{field_root}.link",
            value=link_name,
        )
    required = (
        "position",
        "quaternion_wxyz",
        "gripper_closing_axis",
        "approach_axis",
        "recommended_gripper_width_m",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise AffordanceError(
            "FRAME_FIELD_MISSING",
            f"Affordance frame {name!r} is missing required fields",
            frame=name,
            fields=missing,
        )
    position = _finite_vector(data["position"], 3, f"{field_root}.position")
    quaternion = _quaternion_wxyz(
        data["quaternion_wxyz"], f"{field_root}.quaternion_wxyz"
    )
    closing_axis = _unit_vector(
        data["gripper_closing_axis"], f"{field_root}.gripper_closing_axis"
    )
    approach_axis = _unit_vector(data["approach_axis"], f"{field_root}.approach_axis")
    axis_dot = float(np.dot(closing_axis, approach_axis))
    if abs(axis_dot) > _ORTHOGONAL_TOLERANCE:
        raise AffordanceError(
            "FRAME_AXES_NOT_ORTHOGONAL",
            "gripper_closing_axis and approach_axis must be orthogonal",
            frame=name,
            dot=axis_dot,
        )
    width = _finite_number(
        data["recommended_gripper_width_m"],
        f"{field_root}.recommended_gripper_width_m",
    )
    if width <= 0.0:
        raise AffordanceError(
            "GRIPPER_WIDTH_INVALID",
            "recommended_gripper_width_m must be positive",
            frame=name,
            width=width,
        )
    return HandleFrame(
        name=name,
        link_name=link_name.strip(),
        position=tuple(float(value) for value in position),  # type: ignore[arg-type]
        quaternion_wxyz=tuple(float(value) for value in quaternion),  # type: ignore[arg-type]
        gripper_closing_axis=tuple(float(value) for value in closing_axis),  # type: ignore[arg-type]
        approach_axis=tuple(float(value) for value in approach_axis),  # type: ignore[arg-type]
        recommended_gripper_width_m=width,
    )


def load_affordances(path: str | Path) -> AffordanceSet:
    """Load and validate an ``affordances.json`` file.

    The root must explicitly declare ``quaternion_order: \"wxyz\"``.  An empty
    ``frames`` object is valid so the caller can use geometry fallback.
    """

    affordance_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(affordance_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AffordanceError(
            "AFFORDANCES_NOT_FOUND",
            f"Affordances file does not exist: {affordance_path}",
            path=str(affordance_path),
        ) from exc
    except OSError as exc:
        raise AffordanceError(
            "AFFORDANCES_READ_ERROR",
            f"Cannot read affordances file: {exc}",
            path=str(affordance_path),
        ) from exc
    except json.JSONDecodeError as exc:
        raise AffordanceError(
            "AFFORDANCES_JSON_INVALID",
            f"Invalid affordances JSON: {exc}",
            path=str(affordance_path),
            line=exc.lineno,
            column=exc.colno,
        ) from exc
    data = _required_mapping(raw, "affordances")
    schema_version = data.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version < 1
    ):
        raise AffordanceError(
            "SCHEMA_VERSION_INVALID",
            "schema_version must be a positive integer",
            value=schema_version,
        )
    quaternion_order = data.get("quaternion_order")
    if quaternion_order != "wxyz":
        raise AffordanceError(
            "QUATERNION_ORDER_INVALID",
            "quaternion_order must be exactly 'wxyz'",
            value=quaternion_order,
        )
    raw_frames = _required_mapping(data.get("frames"), "frames")
    frames: dict[str, HandleFrame] = {}
    for name, raw_frame in raw_frames.items():
        if not isinstance(name, str) or not name.strip():
            raise AffordanceError(
                "FRAME_NAME_INVALID", "frame names must be non-empty strings", value=name
            )
        frames[name] = _parse_handle_frame(name, raw_frame)
    metadata = {
        key: value
        for key, value in data.items()
        if key not in {"schema_version", "quaternion_order", "frames"}
    }
    return AffordanceSet(
        path=affordance_path,
        schema_version=schema_version,
        quaternion_order="wxyz",
        frames=MappingProxyType(frames),
        metadata=MappingProxyType(metadata),
    )


def load_handle_frame(path: str | Path, frame_name: str) -> HandleFrame:
    """Load one required named frame from an ``affordances.json`` file."""

    return load_affordances(path).require_frame(frame_name)


def _parse_xml_vector(
    text: str | None,
    *,
    field_name: str,
    default: Sequence[float] | None = None,
) -> np.ndarray:
    if text is None:
        if default is None:
            raise AffordanceError(
                "URDF_GEOMETRY_FIELD_MISSING",
                f"Missing required URDF geometry field: {field_name}",
                field=field_name,
            )
        return np.asarray(default, dtype=float)
    parts = text.split()
    if len(parts) != 3:
        raise AffordanceError(
            "INVALID_VECTOR",
            f"{field_name} must contain exactly 3 values",
            field=field_name,
            value=text,
        )
    return np.asarray(
        [_finite_number(part, f"{field_name}[{index}]") for index, part in enumerate(parts)],
        dtype=float,
    )


def _entry_transform(entry: ET.Element, context: str) -> tuple[np.ndarray, np.ndarray]:
    origin = entry.find("origin")
    xyz = _parse_xml_vector(
        origin.get("xyz") if origin is not None else None,
        field_name=f"{context}.origin.xyz",
        default=(0.0, 0.0, 0.0),
    )
    rpy = _parse_xml_vector(
        origin.get("rpy") if origin is not None else None,
        field_name=f"{context}.origin.rpy",
        default=(0.0, 0.0, 0.0),
    )
    return xyz, rpy_to_matrix(rpy)


def _box_vertices(element: ET.Element, context: str) -> np.ndarray:
    size = _parse_xml_vector(element.get("size"), field_name=f"{context}.box.size")
    if np.any(size <= 0.0):
        raise AffordanceError(
            "URDF_GEOMETRY_DIMENSION_INVALID",
            "Box dimensions must be positive",
            field=f"{context}.box.size",
            value=size.tolist(),
        )
    signs = np.asarray(list(itertools.product((-0.5, 0.5), repeat=3)), dtype=float)
    return signs * size


def _cylinder_vertices(
    element: ET.Element, context: str, radial_samples: int
) -> np.ndarray:
    radius = _finite_number(element.get("radius"), f"{context}.cylinder.radius")
    length = _finite_number(element.get("length"), f"{context}.cylinder.length")
    if radius <= 0.0 or length <= 0.0:
        raise AffordanceError(
            "URDF_GEOMETRY_DIMENSION_INVALID",
            "Cylinder radius and length must be positive",
            field=f"{context}.cylinder",
            radius=radius,
            length=length,
        )
    angles = {
        2.0 * math.pi * index / radial_samples for index in range(radial_samples)
    }
    angles.update((0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0))
    vertices = [
        (radius * math.cos(angle), radius * math.sin(angle), z)
        for z in (-length / 2.0, length / 2.0)
        for angle in sorted(angles)
    ]
    return np.asarray(vertices, dtype=float)


def _sphere_vertices(element: ET.Element, context: str) -> np.ndarray:
    radius = _finite_number(element.get("radius"), f"{context}.sphere.radius")
    if radius <= 0.0:
        raise AffordanceError(
            "URDF_GEOMETRY_DIMENSION_INVALID",
            "Sphere radius must be positive",
            field=f"{context}.sphere.radius",
            radius=radius,
        )
    return radius * np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _load_obj_vertices(path: Path) -> np.ndarray:
    vertices: list[tuple[float, float, float]] = []
    try:
        with path.open("r", encoding="utf-8", errors="strict") as stream:
            for line_number, line in enumerate(stream, start=1):
                stripped = line.lstrip()
                if not stripped.startswith("v ") and not stripped.startswith("v\t"):
                    continue
                parts = stripped.split()
                if len(parts) < 4:
                    raise AffordanceError(
                        "OBJ_VERTEX_INVALID",
                        "OBJ vertex must contain at least x, y, and z",
                        path=str(path),
                        line=line_number,
                    )
                vertex = tuple(
                    _finite_number(parts[index], f"OBJ line {line_number} vertex[{index - 1}]")
                    for index in (1, 2, 3)
                )
                vertices.append(vertex)  # type: ignore[arg-type]
    except UnicodeError as exc:
        raise AffordanceError(
            "OBJ_READ_ERROR", f"Cannot decode OBJ mesh: {exc}", path=str(path)
        ) from exc
    except OSError as exc:
        raise AffordanceError(
            "MESH_READ_ERROR", f"Cannot read mesh: {exc}", path=str(path)
        ) from exc
    if not vertices:
        raise AffordanceError("MESH_EMPTY", "OBJ mesh has no vertices", path=str(path))
    return np.asarray(vertices, dtype=float)


def _load_optional_trimesh_vertices(path: Path) -> np.ndarray:
    try:
        import trimesh  # type: ignore[import-not-found]
    except ImportError as exc:
        raise AffordanceError(
            "MESH_FORMAT_UNSUPPORTED",
            "Non-OBJ meshes require the optional trimesh package",
            path=str(path),
            suffix=path.suffix.lower(),
        ) from exc
    try:
        loaded = trimesh.load(path, process=False)
        if isinstance(loaded, trimesh.Scene):
            loaded = loaded.dump(concatenate=True)
        vertices = np.asarray(loaded.vertices, dtype=float)
    except Exception as exc:
        raise AffordanceError(
            "MESH_LOAD_ERROR", f"Cannot load mesh: {exc}", path=str(path)
        ) from exc
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise AffordanceError("MESH_EMPTY", "Mesh has no 3D vertices", path=str(path))
    if not np.isfinite(vertices).all():
        raise AffordanceError(
            "MESH_VERTEX_NONFINITE", "Mesh contains non-finite vertices", path=str(path)
        )
    return vertices


def _mesh_vertices(
    element: ET.Element,
    *,
    context: str,
    urdf_path: Path,
    handle_link_name: str,
    package_paths: Mapping[str, str | Path] | None,
) -> np.ndarray:
    filename = (element.get("filename") or "").strip()
    if not filename:
        raise AffordanceError(
            "MESH_FILENAME_MISSING",
            "URDF mesh geometry is missing filename",
            field=f"{context}.mesh.filename",
        )
    scale = _parse_xml_vector(
        element.get("scale"),
        field_name=f"{context}.mesh.scale",
        default=(1.0, 1.0, 1.0),
    )
    if np.any(scale <= 0.0):
        raise AffordanceError(
            "MESH_SCALE_INVALID",
            "Mesh scale must contain positive values",
            field=f"{context}.mesh.scale",
            scale=scale.tolist(),
        )
    reference = MeshReference(
        link_name=handle_link_name,
        usage="geometry_fallback",
        filename=filename,
        scale=tuple(float(value) for value in scale),  # type: ignore[arg-type]
    )
    path = resolve_mesh_path(reference, urdf_path, package_paths)
    if path is None:
        raise AffordanceError(
            "MESH_URI_UNRESOLVED",
            "Mesh URI cannot be resolved without an explicit package path",
            filename=filename,
            link=handle_link_name,
        )
    if not path.is_file():
        raise AffordanceError(
            "MESH_FILE_NOT_FOUND",
            f"Mesh file does not exist: {path}",
            filename=filename,
            resolved_path=str(path),
            link=handle_link_name,
        )
    vertices = (
        _load_obj_vertices(path)
        if path.suffix.lower() == ".obj"
        else _load_optional_trimesh_vertices(path)
    )
    return vertices * scale


def _canonical_principal_axes(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = vertices - np.mean(vertices, axis=0)
    covariance = (centered.T @ centered) / float(vertices.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(-eigenvalues, kind="stable")
    axes = eigenvectors[:, order].copy()
    for column in range(3):
        axis = axes[:, column]
        pivot = int(np.argmax(np.abs(axis)))
        if axis[pivot] < 0.0:
            axes[:, column] = -axis
    if float(np.linalg.det(axes)) < 0.0:
        axes[:, 2] = -axes[:, 2]
    projections = vertices @ axes
    extents = np.max(projections, axis=0) - np.min(projections, axis=0)
    return axes, extents


def extract_handle_geometry(
    urdf_path: str | Path,
    handle_link_name: str,
    *,
    primitive_radial_samples: int,
    package_paths: Mapping[str, str | Path] | None = None,
) -> HandleGeometry:
    """Extract named-link collision geometry, falling back to visual geometry.

    Box, cylinder, sphere, and mesh geometry are supported.  OBJ vertices are
    parsed without third-party packages; other mesh formats use ``trimesh`` if
    installed.  Geometry origins and mesh scales are applied before AABB/PCA.
    """

    if (
        isinstance(primitive_radial_samples, bool)
        or not isinstance(primitive_radial_samples, int)
        or primitive_radial_samples < 4
    ):
        raise AffordanceError(
            "CANDIDATE_CONFIG_INVALID",
            "primitive_radial_samples must be an integer of at least 4",
            field="primitive_radial_samples",
            value=primitive_radial_samples,
        )
    if not isinstance(handle_link_name, str) or not handle_link_name.strip():
        raise AffordanceError(
            "HANDLE_LINK_INVALID", "handle_link_name must be a non-empty link name"
        )
    path = Path(urdf_path).expanduser().resolve()
    try:
        tree = ET.parse(path)
    except FileNotFoundError as exc:
        raise AffordanceError(
            "URDF_NOT_FOUND", f"URDF file does not exist: {path}", path=str(path)
        ) from exc
    except OSError as exc:
        raise AffordanceError(
            "URDF_READ_ERROR", f"Cannot read URDF file: {exc}", path=str(path)
        ) from exc
    except ET.ParseError as exc:
        raise AffordanceError(
            "URDF_XML_INVALID",
            f"Invalid URDF XML: {exc}",
            path=str(path),
            line=exc.position[0],
            column=exc.position[1],
        ) from exc
    links = [
        element
        for element in tree.getroot().findall("link")
        if (element.get("name") or "").strip() == handle_link_name
    ]
    if not links:
        raise AffordanceError(
            "HANDLE_LINK_NOT_FOUND",
            f"Handle link not found in URDF: {handle_link_name}",
            link=handle_link_name,
            path=str(path),
        )
    if len(links) > 1:
        raise AffordanceError(
            "HANDLE_LINK_NOT_UNIQUE",
            f"Handle link name is not unique: {handle_link_name}",
            link=handle_link_name,
            count=len(links),
        )
    link = links[0]
    usage = "collision"
    entries = link.findall("collision")
    if not entries:
        usage = "visual"
        entries = link.findall("visual")
    if not entries:
        raise AffordanceError(
            "HANDLE_GEOMETRY_MISSING",
            "Handle link has neither collision nor visual geometry",
            link=handle_link_name,
        )

    vertex_groups: list[np.ndarray] = []
    sources: list[str] = []
    for index, entry in enumerate(entries):
        context = f"link[{handle_link_name}].{usage}[{index}]"
        geometry = entry.find("geometry")
        if geometry is None:
            raise AffordanceError(
                "HANDLE_GEOMETRY_MISSING",
                "Handle geometry entry has no <geometry>",
                link=handle_link_name,
                usage=usage,
                index=index,
            )
        supported = [
            (kind, geometry.find(kind))
            for kind in ("box", "cylinder", "sphere", "mesh")
            if geometry.find(kind) is not None
        ]
        if len(supported) != 1:
            raise AffordanceError(
                "HANDLE_GEOMETRY_INVALID",
                "Each handle geometry entry must contain exactly one supported shape",
                link=handle_link_name,
                usage=usage,
                index=index,
                supported_shapes=[kind for kind, _ in supported],
            )
        kind, shape = supported[0]
        assert shape is not None
        if kind == "box":
            vertices = _box_vertices(shape, context)
        elif kind == "cylinder":
            vertices = _cylinder_vertices(shape, context, primitive_radial_samples)
        elif kind == "sphere":
            vertices = _sphere_vertices(shape, context)
        else:
            vertices = _mesh_vertices(
                shape,
                context=context,
                urdf_path=path,
                handle_link_name=handle_link_name,
                package_paths=package_paths,
            )
        translation, rotation = _entry_transform(entry, context)
        transformed = vertices @ rotation.T + translation
        vertex_groups.append(transformed)
        sources.append(f"{usage}[{index}]:{kind}")

    vertices_local = np.concatenate(vertex_groups, axis=0)
    if vertices_local.shape[0] < 3 or not np.isfinite(vertices_local).all():
        raise AffordanceError(
            "HANDLE_GEOMETRY_INSUFFICIENT",
            "Handle geometry must provide at least three finite vertices",
            link=handle_link_name,
            vertex_count=int(vertices_local.shape[0]),
        )
    aabb_min = np.min(vertices_local, axis=0)
    aabb_max = np.max(vertices_local, axis=0)
    aabb_center = (aabb_min + aabb_max) / 2.0
    principal_axes, principal_extents = _canonical_principal_axes(vertices_local)
    return HandleGeometry(
        link_name=handle_link_name,
        vertices_local=vertices_local,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        aabb_center=aabb_center,
        principal_axes=principal_axes,
        principal_extents=principal_extents,
        sources=tuple(sources),
    )


def _candidate_quaternion(closing_axis: np.ndarray, approach_axis: np.ndarray) -> np.ndarray:
    lateral_axis = np.cross(approach_axis, closing_axis)
    lateral_norm = float(np.linalg.norm(lateral_axis))
    if lateral_norm <= _NUMERICAL_EPS:
        raise AffordanceError(
            "CANDIDATE_AXES_INVALID", "Candidate closing and approach axes are parallel"
        )
    lateral_axis /= lateral_norm
    rotation = np.column_stack((closing_axis, lateral_axis, approach_axis))
    return matrix_to_quaternion(rotation)


def generate_handle_candidates_from_geometry(
    geometry: HandleGeometry,
    config: CandidateGenerationConfig,
) -> tuple[GraspCandidate, ...]:
    """Generate ranked candidates from a handle AABB center and PCA axes."""

    specifications: list[tuple[float, int, int, int, int]] = []
    vertices = geometry.vertices_local
    for closing_index in range(3):
        unsigned_closing = geometry.principal_axes[:, closing_index]
        projection = vertices @ unsigned_closing
        measured_span = float(np.max(projection) - np.min(projection))
        required_width = measured_span + float(config.width_margin_m)
        if required_width > float(config.max_gripper_width_m) + _NUMERICAL_EPS:
            continue
        for approach_index in range(3):
            if approach_index == closing_index:
                continue
            for closing_sign_order, closing_sign in enumerate((1, -1)):
                for approach_sign_order, approach_sign in enumerate((1, -1)):
                    specifications.append(
                        (
                            required_width,
                            closing_index,
                            approach_index,
                            closing_sign_order,
                            approach_sign_order,
                        )
                    )
    specifications.sort()
    if not specifications:
        minimum_span = float(np.min(geometry.principal_extents))
        minimum_required = minimum_span + float(config.width_margin_m)
        raise AffordanceError(
            "GRIPPER_WIDTH_EXCEEDED",
            "No geometry-derived candidate fits within max_gripper_width_m",
            link=geometry.link_name,
            minimum_required_width_m=minimum_required,
            max_gripper_width_m=config.max_gripper_width_m,
            width_margin_m=config.width_margin_m,
        )
    candidates: list[GraspCandidate] = []
    for rank, specification in enumerate(specifications[: config.max_candidates]):
        width, closing_index, approach_index, closing_sign_order, approach_sign_order = specification
        closing_sign = (1.0, -1.0)[closing_sign_order]
        approach_sign = (1.0, -1.0)[approach_sign_order]
        closing_axis = closing_sign * geometry.principal_axes[:, closing_index]
        approach_axis = approach_sign * geometry.principal_axes[:, approach_index]
        quaternion = _candidate_quaternion(closing_axis, approach_axis)
        candidate_id = (
            f"geometry:{rank:03d}:c{closing_index}{'p' if closing_sign > 0 else 'n'}:"
            f"a{approach_index}{'p' if approach_sign > 0 else 'n'}"
        )
        candidates.append(
            GraspCandidate(
                candidate_id=candidate_id,
                rank=rank,
                link_name=geometry.link_name,
                position=tuple(float(value) for value in geometry.aabb_center),  # type: ignore[arg-type]
                quaternion_wxyz=tuple(float(value) for value in quaternion),  # type: ignore[arg-type]
                gripper_closing_axis=tuple(float(value) for value in closing_axis),  # type: ignore[arg-type]
                approach_axis=tuple(float(value) for value in approach_axis),  # type: ignore[arg-type]
                gripper_width_m=float(width),
                source="urdf_geometry_aabb_pca",
            )
        )
    return tuple(candidates)


def generate_handle_candidates(
    urdf_path: str | Path,
    handle_link_name: str,
    *,
    config: CandidateGenerationConfig,
    package_paths: Mapping[str, str | Path] | None = None,
) -> tuple[GraspCandidate, ...]:
    """Extract named handle geometry and generate deterministic candidates."""

    geometry = extract_handle_geometry(
        urdf_path,
        handle_link_name,
        primitive_radial_samples=config.primitive_radial_samples,
        package_paths=package_paths,
    )
    return generate_handle_candidates_from_geometry(geometry, config)


def _frame_candidate(frame: HandleFrame) -> GraspCandidate:
    return GraspCandidate(
        candidate_id=f"frame:{frame.name}",
        rank=0,
        link_name=frame.link_name,
        position=frame.position,
        quaternion_wxyz=frame.quaternion_wxyz,
        gripper_closing_axis=frame.gripper_closing_axis,
        approach_axis=frame.approach_axis,
        gripper_width_m=frame.recommended_gripper_width_m,
        source="affordances_json",
    )


def resolve_handle_candidates(
    affordances: AffordanceSet | str | Path,
    requested_frame: str,
    urdf_path: str | Path,
    handle_link_name: str,
    *,
    config: CandidateGenerationConfig,
    package_paths: Mapping[str, str | Path] | None = None,
) -> CandidateResolution:
    """Resolve an authored frame or fall back to named-link URDF geometry.

    A missing requested frame is retained as a structured reason even when the
    geometry fallback succeeds.  A present frame with a mismatched link or an
    infeasible authored width is an error rather than a silent fallback.
    """

    data = load_affordances(affordances) if isinstance(affordances, (str, Path)) else affordances
    if not isinstance(data, AffordanceSet):
        raise AffordanceError(
            "AFFORDANCES_INVALID", "affordances must be a path or AffordanceSet"
        )
    if not isinstance(requested_frame, str) or not requested_frame.strip():
        raise AffordanceError(
            "FRAME_NAME_INVALID", "requested_frame must be a non-empty frame name"
        )
    frame = data.frames.get(requested_frame)
    if frame is not None:
        if frame.link_name != handle_link_name:
            raise AffordanceError(
                "FRAME_LINK_MISMATCH",
                "Requested frame is attached to a different link",
                frame=requested_frame,
                frame_link=frame.link_name,
                configured_handle_link=handle_link_name,
            )
        if frame.recommended_gripper_width_m > config.max_gripper_width_m + _NUMERICAL_EPS:
            raise AffordanceError(
                "GRIPPER_WIDTH_EXCEEDED",
                "Authored frame width exceeds max_gripper_width_m",
                frame=requested_frame,
                required_width_m=frame.recommended_gripper_width_m,
                max_gripper_width_m=config.max_gripper_width_m,
            )
        return CandidateResolution(
            requested_frame=requested_frame,
            used_geometry_fallback=False,
            candidates=(_frame_candidate(frame),),
        )

    reason = FailureReason(
        code="FRAME_NOT_FOUND",
        message=f"Affordance frame not found: {requested_frame}; used URDF handle geometry",
        context=MappingProxyType(
            {"frame": requested_frame, "available_frames": sorted(data.frames)}
        ),
    )
    geometry = extract_handle_geometry(
        urdf_path,
        handle_link_name,
        primitive_radial_samples=config.primitive_radial_samples,
        package_paths=package_paths,
    )
    candidates = generate_handle_candidates_from_geometry(geometry, config)
    return CandidateResolution(
        requested_frame=requested_frame,
        used_geometry_fallback=True,
        candidates=candidates,
        reasons=(reason,),
        geometry=geometry,
    )


def _callback_result(
    callback: CandidateCheck,
    candidate: GraspCandidate,
    *,
    failure_code: str,
    default_message: str,
    exception_code: str,
) -> tuple[bool, FailureReason | None]:
    try:
        raw = callback(candidate)
    except Exception as exc:
        return False, FailureReason(
            code=exception_code,
            message=f"{default_message}: callback raised {type(exc).__name__}: {exc}",
            context=MappingProxyType(
                {"candidate_id": candidate.candidate_id, "exception_type": type(exc).__name__}
            ),
        )
    if isinstance(raw, CheckResult):
        passed = raw.passed
        message = raw.reason
        details = dict(raw.details)
    elif isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], bool) and isinstance(raw[1], str):
        passed = raw[0]
        message = raw[1]
        details = {}
    elif isinstance(raw, bool):
        passed = raw
        message = None
        details = {}
    else:
        return False, FailureReason(
            code=exception_code,
            message=f"{default_message}: callback returned an unsupported result",
            context=MappingProxyType(
                {"candidate_id": candidate.candidate_id, "result_type": type(raw).__name__}
            ),
        )
    if passed:
        return True, None
    details["candidate_id"] = candidate.candidate_id
    return False, FailureReason(
        code=failure_code,
        message=message or default_message,
        context=MappingProxyType(details),
    )


def select_grasp_candidate(
    candidates: Sequence[GraspCandidate],
    *,
    reachability_check: CandidateCheck,
    collision_free_check: CandidateCheck,
) -> CandidateSelection:
    """Select the first feasible candidate in deterministic rank/id order.

    Both callbacks use pass semantics: ``True`` means reachable for
    ``reachability_check`` and collision-free for ``collision_free_check``.
    A callback may instead return ``(passed, reason)`` or :class:`CheckResult`.
    Rejection and callback-error reasons are retained per candidate.
    Collision is not evaluated when reachability already failed.
    """

    if not callable(reachability_check) or not callable(collision_free_check):
        raise AffordanceError(
            "CANDIDATE_CHECK_INVALID",
            "reachability_check and collision_free_check must be callable",
        )
    ordered = sorted(candidates, key=lambda candidate: (candidate.rank, candidate.candidate_id))
    evaluations: list[CandidateEvaluation] = []
    for candidate in ordered:
        reachable, reach_reason = _callback_result(
            reachability_check,
            candidate,
            failure_code="IK_UNREACHABLE",
            default_message="Candidate is not reachable",
            exception_code="REACHABILITY_CHECK_ERROR",
        )
        if not reachable:
            assert reach_reason is not None
            evaluations.append(
                CandidateEvaluation(
                    candidate=candidate,
                    reachable=False,
                    collision_free=None,
                    accepted=False,
                    reasons=(reach_reason,),
                )
            )
            continue
        collision_free, collision_reason = _callback_result(
            collision_free_check,
            candidate,
            failure_code="COLLISION",
            default_message="Candidate is in collision",
            exception_code="COLLISION_CHECK_ERROR",
        )
        if not collision_free:
            assert collision_reason is not None
            evaluations.append(
                CandidateEvaluation(
                    candidate=candidate,
                    reachable=True,
                    collision_free=False,
                    accepted=False,
                    reasons=(collision_reason,),
                )
            )
            continue
        evaluations.append(
            CandidateEvaluation(
                candidate=candidate,
                reachable=True,
                collision_free=True,
                accepted=True,
                reasons=(),
            )
        )
        return CandidateSelection(selected=candidate, evaluations=tuple(evaluations))
    return CandidateSelection(selected=None, evaluations=tuple(evaluations))


__all__ = [
    "AffordanceError",
    "AffordanceSet",
    "CandidateCheck",
    "CandidateEvaluation",
    "CandidateGenerationConfig",
    "CandidateResolution",
    "CandidateSelection",
    "CheckResult",
    "CheckReturn",
    "FailureReason",
    "GraspCandidate",
    "HandleFrame",
    "HandleGeometry",
    "extract_handle_geometry",
    "generate_handle_candidates",
    "generate_handle_candidates_from_geometry",
    "load_affordances",
    "load_handle_frame",
    "resolve_handle_candidates",
    "select_grasp_candidate",
]
