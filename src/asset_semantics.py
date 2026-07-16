"""Fail-closed semantic inference for microwave-like articulated URDF assets.

The runtime historically required callers to author ``door_joint``, ``door_link``
and a handle affordance by hand.  This module supplies the deterministic first
stage of an upper-level agent: it ranks bounded revolute joints using names,
kinematic topology and geometry, then ranks handle geometry on the selected
door (including fixed descendants).  A result is returned only when both ranks
have a clear winner; ambiguous assets raise :class:`SemanticInferenceError`.

The inference is deliberately inspectable.  Every candidate retains its score
and evidence, and task angles record the explicit safe-opening policy rather
than pretending that a desired opening angle is encoded in the mesh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from .transforms import matrix_to_quaternion, rpy_to_matrix
from .urdf_model import Joint, MeshReference, URDFModelError, load_urdf, resolve_mesh_path


_EPS = 1.0e-12
_DOOR_WORDS = ("door", "hinge", "lid", "hatch", "gate")
_ROTARY_CONTROL_WORDS = ("knob", "dial", "selector", "turntable", "spindle", "button")
_HANDLE_WORDS = ("handle", "grip", "pull", "grab", "lever")
_HANDLE_NEGATIVE_WORDS = ("mount", "hinge", "latch", "panel", "glass", "trim", "frame")


class SemanticInferenceError(ValueError):
    """Machine-readable failure raised instead of guessing an ambiguous semantic."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": dict(self.context)}


@dataclass(frozen=True, slots=True)
class InferencePolicy:
    """Explicit policy knobs used after geometry/kinematics are inferred."""

    max_gripper_width_m: float = 0.08
    gripper_width_margin_m: float = 0.01
    preferred_opening_angle_deg: float = 65.0
    joint_limit_reserve_deg: float = 2.0
    minimum_door_score: float = 8.0
    minimum_door_margin: float = 3.0
    minimum_handle_score: float = 7.0
    minimum_handle_margin: float = 2.0

    def __post_init__(self) -> None:
        numeric = (
            "max_gripper_width_m",
            "gripper_width_margin_m",
            "preferred_opening_angle_deg",
            "joint_limit_reserve_deg",
            "minimum_door_score",
            "minimum_door_margin",
            "minimum_handle_score",
            "minimum_handle_margin",
        )
        for name in numeric:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise SemanticInferenceError(
                    "INFERENCE_POLICY_INVALID", f"{name} must be finite and numeric", field=name
                )
        if self.max_gripper_width_m <= 0.0:
            raise SemanticInferenceError(
                "INFERENCE_POLICY_INVALID", "max_gripper_width_m must be positive"
            )
        if self.gripper_width_margin_m < 0.0:
            raise SemanticInferenceError(
                "INFERENCE_POLICY_INVALID", "gripper_width_margin_m must be non-negative"
            )
        if self.preferred_opening_angle_deg <= 0.0 or self.joint_limit_reserve_deg < 0.0:
            raise SemanticInferenceError(
                "INFERENCE_POLICY_INVALID", "opening angle must be positive and reserve non-negative"
            )


@dataclass(frozen=True, slots=True)
class GeometryPart:
    """One collision (or visual fallback) geometry entry in its owning link."""

    link_name: str
    name: str
    usage: str
    shape: str
    vertices_local: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_local, dtype=float)
        if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 3:
            raise SemanticInferenceError(
                "GEOMETRY_INVALID", "geometry must contain at least three 3D vertices",
                link=self.link_name, geometry=self.name,
            )
        if not np.isfinite(vertices).all():
            raise SemanticInferenceError(
                "GEOMETRY_INVALID", "geometry vertices must be finite",
                link=self.link_name, geometry=self.name,
            )
        vertices = vertices.copy()
        vertices.setflags(write=False)
        object.__setattr__(self, "vertices_local", vertices)

    @property
    def center_local(self) -> np.ndarray:
        return 0.5 * (np.min(self.vertices_local, axis=0) + np.max(self.vertices_local, axis=0))

    @property
    def principal_axes_and_extents(self) -> tuple[np.ndarray, np.ndarray]:
        return _principal_axes_and_extents(self.vertices_local)


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    name: str
    score: float
    evidence: tuple[str, ...]
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "evidence": list(self.evidence),
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class InferredHandleFrame:
    name: str
    link_name: str
    geometry_name: str
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    gripper_closing_axis: tuple[float, float, float]
    approach_axis: tuple[float, float, float]
    recommended_gripper_width_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "link": self.link_name,
            "geometry": self.geometry_name,
            "position": list(self.position),
            "quaternion_wxyz": list(self.quaternion_wxyz),
            "gripper_closing_axis": list(self.gripper_closing_axis),
            "approach_axis": list(self.approach_axis),
            "recommended_gripper_width_m": self.recommended_gripper_width_m,
        }


@dataclass(frozen=True, slots=True)
class MicrowaveSemantics:
    urdf_path: str
    door_joint_name: str
    door_link_name: str
    closed_angle_rad: float
    goal_angle_rad: float
    handle: InferredHandleFrame
    door_confidence: float
    handle_confidence: float
    door_candidates: tuple[RankedCandidate, ...]
    handle_candidates: tuple[RankedCandidate, ...]
    warnings: tuple[str, ...]

    @property
    def closed_angle_deg(self) -> float:
        return math.degrees(self.closed_angle_rad)

    @property
    def goal_angle_deg(self) -> float:
        return math.degrees(self.goal_angle_rad)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "urdf_path": self.urdf_path,
            "door_joint": self.door_joint_name,
            "door_link": self.door_link_name,
            "closed_angle_rad": self.closed_angle_rad,
            "closed_angle_deg": self.closed_angle_deg,
            "goal_angle_rad": self.goal_angle_rad,
            "goal_angle_deg": self.goal_angle_deg,
            "handle": self.handle.to_dict(),
            "confidence": {"door": self.door_confidence, "handle": self.handle_confidence},
            "evidence": {
                "door_candidates": [candidate.to_dict() for candidate in self.door_candidates],
                "handle_candidates": [candidate.to_dict() for candidate in self.handle_candidates],
            },
            "warnings": list(self.warnings),
        }

    def to_affordances_payload(self, *, asset_name: str) -> dict[str, Any]:
        """Return a payload accepted by :func:`src.affordances.load_affordances`."""

        frame = self.handle.to_dict()
        frame.pop("name")
        frame.pop("geometry")
        return {
            "schema_version": 1,
            "asset_name": asset_name,
            "quaternion_order": "wxyz",
            "frames": {self.handle.name: frame},
            "metadata": {
                "source": "agentpre_asset_semantics",
                "door_confidence": self.door_confidence,
                "handle_confidence": self.handle_confidence,
            },
        }


def _xml_vector(text: str | None, *, default: Sequence[float], field_name: str) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=float)
    parts = text.split()
    if len(parts) != 3:
        raise SemanticInferenceError(
            "URDF_VECTOR_INVALID", f"{field_name} must have three values", value=text
        )
    try:
        vector = np.asarray([float(value) for value in parts], dtype=float)
    except ValueError as exc:
        raise SemanticInferenceError(
            "URDF_VECTOR_INVALID", f"{field_name} must be numeric", value=text
        ) from exc
    if not np.isfinite(vector).all():
        raise SemanticInferenceError(
            "URDF_VECTOR_INVALID", f"{field_name} must be finite", value=text
        )
    return vector


def _positive_float(text: str | None, field_name: str) -> float:
    try:
        value = float(text)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise SemanticInferenceError(
            "URDF_GEOMETRY_INVALID", f"{field_name} must be numeric", value=text
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise SemanticInferenceError(
            "URDF_GEOMETRY_INVALID", f"{field_name} must be finite and positive", value=text
        )
    return value


def _entry_transform(entry: ET.Element) -> np.ndarray:
    origin = entry.find("origin")
    xyz = _xml_vector(
        origin.get("xyz") if origin is not None else None,
        default=(0.0, 0.0, 0.0), field_name="origin.xyz",
    )
    rpy = _xml_vector(
        origin.get("rpy") if origin is not None else None,
        default=(0.0, 0.0, 0.0), field_name="origin.rpy",
    )
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rpy_to_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def _box_vertices(element: ET.Element) -> np.ndarray:
    size = _xml_vector(element.get("size"), default=(), field_name="box.size")
    if size.shape != (3,) or np.any(size <= 0.0):
        raise SemanticInferenceError("URDF_GEOMETRY_INVALID", "box size must be positive")
    return np.asarray(list(itertools.product((-0.5, 0.5), repeat=3)), dtype=float) * size


def _cylinder_vertices(element: ET.Element) -> np.ndarray:
    radius = _positive_float(element.get("radius"), "cylinder.radius")
    length = _positive_float(element.get("length"), "cylinder.length")
    return np.asarray(
        [
            (radius * math.cos(angle), radius * math.sin(angle), z)
            for z in (-0.5 * length, 0.5 * length)
            for angle in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
        ],
        dtype=float,
    )


def _sphere_vertices(element: ET.Element) -> np.ndarray:
    radius = _positive_float(element.get("radius"), "sphere.radius")
    return radius * np.asarray(
        [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)],
        dtype=float,
    )


def _mesh_vertices(
    element: ET.Element,
    *,
    link_name: str,
    urdf_path: Path,
    package_paths: Mapping[str, str | Path] | None,
) -> np.ndarray:
    filename = (element.get("filename") or "").strip()
    scale = _xml_vector(
        element.get("scale"), default=(1.0, 1.0, 1.0), field_name="mesh.scale"
    )
    reference = MeshReference(
        link_name=link_name, usage="semantic_inference", filename=filename,
        scale=tuple(float(value) for value in scale),
    )
    resolved = resolve_mesh_path(reference, urdf_path, package_paths)
    if resolved is None or not resolved.is_file():
        raise SemanticInferenceError(
            "MESH_UNAVAILABLE", "mesh cannot be resolved for semantic inference",
            link=link_name, filename=filename,
        )
    if resolved.suffix.lower() == ".obj":
        vertices: list[tuple[float, float, float]] = []
        for line_number, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if not fields or fields[0] != "v":
                continue
            if len(fields) < 4:
                raise SemanticInferenceError(
                    "MESH_INVALID", "OBJ vertex is incomplete", path=str(resolved), line=line_number
                )
            vertices.append(tuple(float(value) for value in fields[1:4]))
        if not vertices:
            raise SemanticInferenceError("MESH_INVALID", "OBJ has no vertices", path=str(resolved))
        return np.asarray(vertices, dtype=float) * scale
    try:
        import trimesh  # type: ignore[import-not-found]

        loaded = trimesh.load(resolved, process=False)
        if isinstance(loaded, trimesh.Scene):
            loaded = loaded.dump(concatenate=True)
        vertices = np.asarray(loaded.vertices, dtype=float)
    except Exception as exc:
        raise SemanticInferenceError(
            "MESH_UNAVAILABLE", "non-OBJ mesh requires a readable trimesh installation",
            path=str(resolved), exception=type(exc).__name__,
        ) from exc
    return vertices * scale


def _geometry_vertices(
    geometry: ET.Element,
    *,
    link_name: str,
    urdf_path: Path,
    package_paths: Mapping[str, str | Path] | None,
) -> tuple[str, np.ndarray]:
    shapes = [(name, geometry.find(name)) for name in ("box", "cylinder", "sphere", "mesh")]
    shapes = [(name, element) for name, element in shapes if element is not None]
    if len(shapes) != 1:
        raise SemanticInferenceError(
            "URDF_GEOMETRY_INVALID", "geometry must contain exactly one supported shape",
            link=link_name,
        )
    shape, element = shapes[0]
    assert element is not None
    if shape == "box":
        return shape, _box_vertices(element)
    if shape == "cylinder":
        return shape, _cylinder_vertices(element)
    if shape == "sphere":
        return shape, _sphere_vertices(element)
    return shape, _mesh_vertices(
        element, link_name=link_name, urdf_path=urdf_path, package_paths=package_paths
    )


def _load_geometry_parts(
    root: ET.Element,
    urdf_path: Path,
    package_paths: Mapping[str, str | Path] | None,
) -> tuple[dict[str, tuple[GeometryPart, ...]], list[str]]:
    by_link: dict[str, tuple[GeometryPart, ...]] = {}
    warnings: list[str] = []
    for link in root.findall("link"):
        link_name = (link.get("name") or "").strip()
        collision_entries = link.findall("collision")
        usage = "collision" if collision_entries else "visual"
        entries = collision_entries or link.findall("visual")
        parts: list[GeometryPart] = []
        for index, entry in enumerate(entries):
            geometry = entry.find("geometry")
            part_name = (entry.get("name") or f"{usage}_{index}").strip()
            if geometry is None:
                warnings.append(f"{link_name}/{part_name}: missing geometry")
                continue
            try:
                shape, vertices = _geometry_vertices(
                    geometry, link_name=link_name, urdf_path=urdf_path,
                    package_paths=package_paths,
                )
                transform = _entry_transform(entry)
                with np.errstate(all="ignore"):
                    transformed = (transform[:3, :3] @ vertices.T).T + transform[:3, 3]
                if not np.isfinite(transformed).all():
                    raise SemanticInferenceError(
                        "GEOMETRY_NONFINITE",
                        "geometry transform produced non-finite vertices",
                        link=link_name,
                        geometry=part_name,
                    )
                parts.append(
                    GeometryPart(
                        link_name=link_name, name=part_name, usage=usage,
                        shape=shape, vertices_local=transformed,
                    )
                )
            except SemanticInferenceError as exc:
                warnings.append(f"{link_name}/{part_name}: {exc.code}")
        by_link[link_name] = tuple(parts)
    return by_link, warnings


def _principal_axes_and_extents(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(vertices).all():
        raise SemanticInferenceError(
            "GEOMETRY_NONFINITE", "cannot compute PCA from non-finite vertices"
        )
    with np.errstate(all="ignore"):
        centered = vertices - np.mean(vertices, axis=0)
        covariance = centered.T @ centered / float(vertices.shape[0])
    if not np.isfinite(covariance).all():
        raise SemanticInferenceError(
            "GEOMETRY_NONFINITE", "geometry covariance overflowed or became non-finite"
        )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(-eigenvalues, kind="stable")
    axes = eigenvectors[:, order].copy()
    for column in range(3):
        pivot = int(np.argmax(np.abs(axes[:, column])))
        if axes[pivot, column] < 0.0:
            axes[:, column] *= -1.0
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] *= -1.0
    with np.errstate(all="ignore"):
        projection = vertices @ axes
    if not np.isfinite(projection).all():
        raise SemanticInferenceError(
            "GEOMETRY_NONFINITE", "geometry PCA projection became non-finite"
        )
    extents = np.max(projection, axis=0) - np.min(projection, axis=0)
    return axes, extents


def _joint_transform(joint: Joint) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rpy_to_matrix(joint.origin.rpy)
    transform[:3, 3] = joint.origin.xyz
    return transform


def _fixed_subtree_transforms(model: Any, root_link: str) -> dict[str, np.ndarray]:
    transforms = {root_link: np.eye(4, dtype=float)}
    while True:
        changed = False
        for joint in model.joints.values():
            if joint.joint_type != "fixed" or joint.parent not in transforms or joint.child in transforms:
                continue
            transforms[joint.child] = transforms[joint.parent] @ _joint_transform(joint)
            changed = True
        if not changed:
            return transforms


def _transformed_vertices(part: GeometryPart, transform: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        vertices = (transform[:3, :3] @ part.vertices_local.T).T + transform[:3, 3]
    if not np.isfinite(vertices).all():
        raise SemanticInferenceError(
            "GEOMETRY_NONFINITE", "fixed-link transform produced non-finite vertices",
            link=part.link_name, geometry=part.name,
        )
    return vertices


def _contains_any(value: str, words: Sequence[str]) -> bool:
    lowered = value.lower()
    return any(word in lowered for word in words)


def _bounded_joint_angles(joint: Joint, policy: InferencePolicy) -> tuple[float, float]:
    if joint.joint_type != "revolute" or joint.limit is None:
        raise SemanticInferenceError(
            "DOOR_LIMITS_UNAVAILABLE", "door inference requires a bounded revolute joint",
            joint=joint.name,
        )
    lower, upper = joint.limit.lower, joint.limit.upper
    if lower is None or upper is None or not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise SemanticInferenceError(
            "DOOR_LIMITS_UNAVAILABLE", "door joint requires finite ordered limits", joint=joint.name
        )
    # URDF assets conventionally place the closed pose at zero.  If zero is not
    # inside the interval, the nearest endpoint is the only defensible fallback.
    closed = min(max(0.0, lower), upper)
    endpoints = (lower, upper)
    far = max(endpoints, key=lambda endpoint: abs(endpoint - closed))
    direction = 1.0 if far > closed else -1.0
    available = abs(far - closed)
    reserve = math.radians(policy.joint_limit_reserve_deg)
    preferred = math.radians(policy.preferred_opening_angle_deg)
    opening = min(preferred, max(0.0, available - reserve))
    if opening <= math.radians(5.0):
        raise SemanticInferenceError(
            "DOOR_RANGE_TOO_SMALL", "door joint has insufficient range after reserve",
            joint=joint.name, available_range_rad=available,
        )
    return closed, closed + direction * opening


def _confidence(top_score: float, second_score: float | None, threshold: float) -> float:
    margin = top_score - (second_score if second_score is not None else threshold)
    margin_term = 1.0 / (1.0 + math.exp(-0.75 * margin))
    quality_term = 1.0 / (1.0 + math.exp(-0.5 * (top_score - threshold)))
    return float(margin_term * quality_term)


@dataclass(frozen=True, slots=True)
class _DoorGeometry:
    transforms: Mapping[str, np.ndarray]
    parts: tuple[tuple[GeometryPart, np.ndarray], ...]
    main_part: GeometryPart | None
    main_transform: np.ndarray | None
    aggregate_vertices: np.ndarray | None


def _door_geometry(model: Any, parts: Mapping[str, tuple[GeometryPart, ...]], door_link: str) -> _DoorGeometry:
    transforms = _fixed_subtree_transforms(model, door_link)
    views: list[tuple[GeometryPart, np.ndarray]] = []
    for link_name, transform in transforms.items():
        for part in parts.get(link_name, ()):
            views.append((part, transform))
    if not views:
        return _DoorGeometry(MappingProxyType(transforms), (), None, None, None)
    main_part, main_transform = max(
        views,
        key=lambda view: float(np.prod(np.sort(_principal_axes_and_extents(
            _transformed_vertices(view[0], view[1])
        )[1])[-2:])),
    )
    aggregate = np.concatenate([_transformed_vertices(part, transform) for part, transform in views])
    aggregate.setflags(write=False)
    return _DoorGeometry(
        MappingProxyType(transforms), tuple(views), main_part, main_transform, aggregate
    )


def _rank_door_joints(
    model: Any,
    parts: Mapping[str, tuple[GeometryPart, ...]],
) -> tuple[tuple[RankedCandidate, ...], dict[str, _DoorGeometry]]:
    ranked: list[RankedCandidate] = []
    geometries: dict[str, _DoorGeometry] = {}
    for joint in model.joints.values():
        if joint.joint_type not in {"revolute", "continuous"}:
            continue
        score = 0.0
        evidence: list[str] = []
        if joint.joint_type == "revolute":
            score += 4.0
            evidence.append("bounded revolute topology +4")
        else:
            score -= 2.0
            evidence.append("continuous rotary joint -2")
        if _contains_any(joint.name, _DOOR_WORDS):
            score += 6.0
            evidence.append("joint name has door/hinge semantic +6")
        if _contains_any(joint.child, _DOOR_WORDS):
            score += 6.0
            evidence.append("child link has door semantic +6")
        if _contains_any(joint.name, _ROTARY_CONTROL_WORDS):
            score -= 6.0
            evidence.append("joint name resembles rotary control -6")
        if _contains_any(joint.child, _ROTARY_CONTROL_WORDS):
            score -= 5.0
            evidence.append("child link resembles rotary control -5")
        if joint.limit is not None and joint.limit.lower is not None and joint.limit.upper is not None:
            span = joint.limit.upper - joint.limit.lower
            if 0.35 <= span <= 3.2:
                score += 2.0
                evidence.append("door-like bounded angular range +2")
            elif span < 0.2:
                score -= 2.0
                evidence.append("very small angular range -2")
        geometry = _door_geometry(model, parts, joint.child)
        geometries[joint.name] = geometry
        if geometry.aggregate_vertices is not None:
            _, aggregate_extents = _principal_axes_and_extents(geometry.aggregate_vertices)
            ordered = np.sort(aggregate_extents)[::-1]
            if ordered[0] > _EPS and ordered[1] / ordered[0] >= 0.25 and ordered[2] / ordered[1] <= 0.45:
                score += 3.0
                evidence.append("child subtree is panel-like +3")
            axis = np.asarray(joint.axis, dtype=float)
            norm = float(np.linalg.norm(axis))
            if norm > _EPS:
                axis /= norm
                center = np.mean(geometry.aggregate_vertices, axis=0)
                radial = float(np.linalg.norm(center - axis * np.dot(center, axis)))
                radial_ratio = radial / max(float(ordered[0]), _EPS)
                if radial > 0.10 and radial_ratio > 0.18:
                    score += 3.0
                    evidence.append("panel centroid is offset from hinge axis +3")
            if any(
                _contains_any(part.name, _HANDLE_WORDS) or _contains_any(part.link_name, _HANDLE_WORDS)
                for part, _ in geometry.parts
            ):
                score += 2.0
                evidence.append("fixed subtree contains handle semantics +2")
            details = {
                "joint_type": joint.joint_type,
                "child_link": joint.child,
                "aggregate_extents_m": aggregate_extents.tolist(),
            }
        else:
            evidence.append("no usable child geometry +0")
            details = {"joint_type": joint.joint_type, "child_link": joint.child}
        ranked.append(
            RankedCandidate(
                name=joint.name, score=score, evidence=tuple(evidence),
                details=MappingProxyType(details),
            )
        )
    ranked.sort(key=lambda candidate: (-candidate.score, candidate.name))
    return tuple(ranked), geometries


def _rank_handle_parts(
    door_joint: Joint,
    geometry: _DoorGeometry,
    policy: InferencePolicy,
) -> tuple[tuple[RankedCandidate, ...], dict[str, tuple[GeometryPart, np.ndarray]]]:
    if geometry.aggregate_vertices is None or geometry.main_part is None or geometry.main_transform is None:
        return (), {}
    main_vertices = _transformed_vertices(geometry.main_part, geometry.main_transform)
    main_center = np.mean(main_vertices, axis=0)
    main_axes, main_extents = _principal_axes_and_extents(main_vertices)
    door_normal = main_axes[:, int(np.argmin(main_extents))]
    axis = np.asarray(door_joint.axis, dtype=float)
    axis /= max(float(np.linalg.norm(axis)), _EPS)
    all_radial = np.linalg.norm(
        geometry.aggregate_vertices - np.outer(geometry.aggregate_vertices @ axis, axis), axis=1
    )
    maximum_radius = max(float(np.max(all_radial)), _EPS)
    main_area = max(float(np.prod(np.sort(main_extents)[-2:])), _EPS)
    ranked: list[RankedCandidate] = []
    lookup: dict[str, tuple[GeometryPart, np.ndarray]] = {}
    for part, transform in geometry.parts:
        vertices_door = _transformed_vertices(part, transform)
        center = np.mean(vertices_door, axis=0)
        _, extents = _principal_axes_and_extents(vertices_door)
        ordered = np.sort(extents)[::-1]
        score = 0.0
        evidence: list[str] = []
        semantic_name = f"{part.link_name}/{part.name}"
        if _contains_any(part.name, _HANDLE_WORDS):
            score += 8.0
            evidence.append("geometry name has handle/grip/pull semantic +8")
        if _contains_any(part.link_name, _HANDLE_WORDS):
            score += 6.0
            evidence.append("link name has handle/grip semantic +6")
        if _contains_any(part.name, _HANDLE_NEGATIVE_WORDS):
            score -= 4.0
            evidence.append("geometry name resembles mount/panel/trim -4")
        if part is geometry.main_part and part.link_name == geometry.main_part.link_name:
            score -= 8.0
            evidence.append("largest door panel cannot be the handle -8")
        if ordered[1] > _EPS and ordered[0] / ordered[1] >= 2.2:
            score += 3.0
            evidence.append("geometry is handle-bar elongated +3")
        part_area = float(np.prod(ordered[:2]))
        if part_area / main_area < 0.35:
            score += 1.5
            evidence.append("geometry is small relative to door panel +1.5")
        radial = float(np.linalg.norm(center - axis * np.dot(center, axis)))
        radial_ratio = radial / maximum_radius
        if radial_ratio >= 0.65:
            score += 3.0
            evidence.append("geometry is far from hinge axis +3")
        normal_offset = abs(float(np.dot(center - main_center, door_normal)))
        panel_thickness = max(float(np.min(main_extents)), 0.005)
        if normal_offset >= 0.6 * panel_thickness:
            score += 2.0
            evidence.append("geometry protrudes from door plane +2")
        feasible_span = float(np.min(extents)) + policy.gripper_width_margin_m
        if feasible_span <= policy.max_gripper_width_m + _EPS:
            score += 2.0
            evidence.append("cross-section fits configured gripper +2")
        else:
            score -= 6.0
            evidence.append("cross-section exceeds configured gripper -6")
        key = semantic_name
        if key in lookup:
            key = f"{semantic_name}#{len(lookup)}"
        lookup[key] = (part, transform)
        ranked.append(
            RankedCandidate(
                name=key, score=score, evidence=tuple(evidence),
                details=MappingProxyType(
                    {
                        "link": part.link_name,
                        "geometry": part.name,
                        "shape": part.shape,
                        "center_in_door_link_m": center.tolist(),
                        "principal_extents_m": extents.tolist(),
                        "hinge_radius_ratio": radial_ratio,
                    }
                ),
            )
        )
    ranked.sort(key=lambda candidate: (-candidate.score, candidate.name))
    return tuple(ranked), lookup


def _make_handle_frame(
    selected: RankedCandidate,
    part: GeometryPart,
    door_to_handle_link: np.ndarray,
    door_geometry: _DoorGeometry,
    policy: InferencePolicy,
) -> InferredHandleFrame:
    assert door_geometry.main_part is not None and door_geometry.main_transform is not None
    main_vertices = _transformed_vertices(door_geometry.main_part, door_geometry.main_transform)
    main_center = np.mean(main_vertices, axis=0)
    main_axes, main_extents = _principal_axes_and_extents(main_vertices)
    outward_door = main_axes[:, int(np.argmin(main_extents))]
    handle_center_door = (
        door_to_handle_link[:3, :3] @ part.center_local + door_to_handle_link[:3, 3]
    )
    if float(np.dot(handle_center_door - main_center, outward_door)) < 0.0:
        outward_door = -outward_door
    rotation_door_from_link = door_to_handle_link[:3, :3]
    # ``approach_axis`` follows AgentPre's trajectory convention: it points
    # from the exterior pregrasp toward the door/handle.  Therefore the
    # pregrasp expression ``handle - approach * distance`` stays outside.
    approach_local = rotation_door_from_link.T @ (-outward_door)
    approach_local /= max(float(np.linalg.norm(approach_local)), _EPS)
    axes, _ = part.principal_axes_and_extents
    spans = np.asarray(
        [
            float(np.ptp(part.vertices_local @ axes[:, index]))
            for index in range(3)
        ]
    )
    longitudinal_candidates = sorted(
        range(3),
        key=lambda index: (
            abs(float(np.dot(axes[:, index], approach_local))) > 0.75,
            -spans[index],
            index,
        ),
    )
    longitudinal_local = axes[:, longitudinal_candidates[0]].copy()
    longitudinal_local -= approach_local * float(np.dot(longitudinal_local, approach_local))
    longitudinal_norm = float(np.linalg.norm(longitudinal_local))
    if longitudinal_norm <= _EPS:
        raise SemanticInferenceError(
            "HANDLE_FRAME_DEGENERATE", "cannot derive orthogonal handle axes", candidate=selected.name
        )
    longitudinal_local /= longitudinal_norm
    pivot = int(np.argmax(np.abs(longitudinal_local)))
    if longitudinal_local[pivot] < 0.0:
        longitudinal_local *= -1.0
    closing_local = np.cross(approach_local, longitudinal_local)
    closing_local /= max(float(np.linalg.norm(closing_local)), _EPS)
    # Frame convention: local X closes the gripper, local Y approaches the
    # handle, and local Z follows the bar.  On the current Articraft record
    # this deterministically yields identity orientation (X, +Y, Z).
    rotation = np.column_stack((closing_local, approach_local, longitudinal_local))
    quaternion = matrix_to_quaternion(rotation)
    width = float(np.ptp(part.vertices_local @ closing_local)) + policy.gripper_width_margin_m
    if width > policy.max_gripper_width_m + _EPS:
        raise SemanticInferenceError(
            "HANDLE_TOO_WIDE", "selected handle exceeds configured gripper width",
            candidate=selected.name, required_width_m=width,
            max_gripper_width_m=policy.max_gripper_width_m,
        )
    return InferredHandleFrame(
        name="inferred_pull_grip",
        link_name=part.link_name,
        geometry_name=part.name,
        position=tuple(float(value) for value in part.center_local),
        quaternion_wxyz=tuple(float(value) for value in quaternion),
        gripper_closing_axis=(1.0, 0.0, 0.0),
        approach_axis=(0.0, 1.0, 0.0),
        recommended_gripper_width_m=width,
    )


def infer_microwave_semantics(
    urdf_path: str | Path,
    *,
    policy: InferencePolicy | None = None,
    package_paths: Mapping[str, str | Path] | None = None,
) -> MicrowaveSemantics:
    """Infer door and handle semantics, refusing low-score or tied candidates.

    This function performs no simulation and writes no files.  The returned
    affordance can be fed to reachability/collision planning as the next stage;
    semantic confidence is not a substitute for those physical checks.
    """

    active_policy = policy or InferencePolicy()
    path = Path(urdf_path).expanduser().resolve()
    try:
        model = load_urdf(path)
        root = ET.parse(path).getroot()
    except (URDFModelError, ET.ParseError, OSError) as exc:
        raise SemanticInferenceError(
            "URDF_LOAD_FAILED", f"cannot load URDF for semantic inference: {exc}", path=str(path)
        ) from exc
    parts, warnings = _load_geometry_parts(root, path, package_paths)
    door_candidates, door_geometries = _rank_door_joints(model, parts)
    if not door_candidates:
        raise SemanticInferenceError(
            "DOOR_NOT_FOUND", "URDF has no revolute or continuous joint candidates", path=str(path)
        )
    top_door = door_candidates[0]
    second_door_score = door_candidates[1].score if len(door_candidates) > 1 else None
    door_margin = top_door.score - (
        second_door_score if second_door_score is not None else active_policy.minimum_door_score
    )
    if top_door.score < active_policy.minimum_door_score or (
        second_door_score is not None and door_margin < active_policy.minimum_door_margin
    ):
        raise SemanticInferenceError(
            "DOOR_INFERENCE_AMBIGUOUS",
            "door candidate does not clear score and margin thresholds",
            candidates=[candidate.to_dict() for candidate in door_candidates[:5]],
            minimum_score=active_policy.minimum_door_score,
            minimum_margin=active_policy.minimum_door_margin,
        )
    door_joint = model.require_joint(top_door.name)
    closed, goal = _bounded_joint_angles(door_joint, active_policy)
    door_geometry = door_geometries[door_joint.name]
    handle_candidates, handle_lookup = _rank_handle_parts(door_joint, door_geometry, active_policy)
    if not handle_candidates:
        raise SemanticInferenceError(
            "HANDLE_NOT_FOUND", "selected door subtree has no usable geometry", door_joint=door_joint.name
        )
    top_handle = handle_candidates[0]
    second_handle_score = handle_candidates[1].score if len(handle_candidates) > 1 else None
    handle_margin = top_handle.score - (
        second_handle_score if second_handle_score is not None else active_policy.minimum_handle_score
    )
    if top_handle.score < active_policy.minimum_handle_score or (
        second_handle_score is not None and handle_margin < active_policy.minimum_handle_margin
    ):
        raise SemanticInferenceError(
            "HANDLE_INFERENCE_AMBIGUOUS",
            "handle candidate does not clear score and margin thresholds",
            door_joint=door_joint.name,
            candidates=[candidate.to_dict() for candidate in handle_candidates[:8]],
            minimum_score=active_policy.minimum_handle_score,
            minimum_margin=active_policy.minimum_handle_margin,
        )
    part, door_to_handle_link = handle_lookup[top_handle.name]
    handle = _make_handle_frame(
        top_handle, part, door_to_handle_link, door_geometry, active_policy
    )
    door_confidence = _confidence(
        top_door.score, second_door_score, active_policy.minimum_door_score
    )
    handle_confidence = _confidence(
        top_handle.score, second_handle_score, active_policy.minimum_handle_score
    )
    return MicrowaveSemantics(
        urdf_path=str(path), door_joint_name=door_joint.name,
        door_link_name=door_joint.child, closed_angle_rad=closed, goal_angle_rad=goal,
        handle=handle, door_confidence=door_confidence,
        handle_confidence=handle_confidence,
        door_candidates=door_candidates, handle_candidates=handle_candidates,
        warnings=tuple(warnings),
    )


def infer_task_semantics(
    urdf_path: str | Path,
    *,
    policy: InferencePolicy | None = None,
    package_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Compatibility provider for the upper-agent semantic decision protocol.

    The precise authored frame is returned as ``affordance_payload`` so a
    same-link handle (as in the Articraft microwave) does not accidentally ask
    downstream geometry fallback to treat the entire door as a handle.
    """

    result = infer_microwave_semantics(
        urdf_path, policy=policy, package_paths=package_paths
    )
    door_rationale = "; ".join(result.door_candidates[0].evidence)
    handle_rationale = "; ".join(result.handle_candidates[0].evidence)

    def decision(value: Any, confidence: float, rationale: str) -> dict[str, Any]:
        return {
            "value": value,
            "source": "urdf_topology_geometry_inference",
            "confidence": confidence,
            "rationale": rationale,
        }

    payload = result.to_affordances_payload(asset_name=Path(urdf_path).stem)
    return {
        "decisions": {
            "door_joint": decision(
                result.door_joint_name, result.door_confidence, door_rationale
            ),
            "door_link": decision(
                result.door_link_name, result.door_confidence,
                "child link of the clear door-joint winner",
            ),
            "handle_link": decision(
                result.handle.link_name, result.handle_confidence, handle_rationale
            ),
            "handle_frame": decision(
                result.handle.name, result.handle_confidence,
                "frame reconstructed from the selected named/geometry handle part",
            ),
            "closed_angle_deg": decision(
                result.closed_angle_deg, result.door_confidence,
                "zero-clamped closed pose inside the finite door-joint limits",
            ),
            "goal_angle_deg": decision(
                result.goal_angle_deg, result.door_confidence,
                "preferred safe opening angle clipped by the joint-limit reserve",
            ),
        },
        "affordance_payload": payload,
        "semantic_evidence": result.to_dict()["evidence"],
        "warnings": list(result.warnings),
    }


# Short alias accepted by provider loaders that prefer a generic name.
infer_semantics = infer_task_semantics


__all__ = [
    "InferencePolicy",
    "InferredHandleFrame",
    "MicrowaveSemantics",
    "RankedCandidate",
    "SemanticInferenceError",
    "infer_semantics",
    "infer_task_semantics",
    "infer_microwave_semantics",
]
