"""Auditable, name-based collision checks for deterministic trajectories.

The checker in this module is deliberately independent of simulator body and
joint indices.  It reads every ``<collision>`` geometry from the configured
robot and object URDFs, resolves it by link name, and builds rotation-aware
oriented bounds from caller-supplied named-link forward kinematics.

World AABBs and deterministic sweep-and-prune are retained strictly as the
broad phase.  Every broad-phase candidate is then checked with the 15-axis OBB
separating-axis test (three axes from each box plus nine cross products;
degenerate cross axes are skipped).  The configured margin is applied once to
the narrow-phase projected separation.  Reports retain both the broad-phase
candidate and the final ``obb_overlap``/``within_margin`` decision, so a world
AABB false positive is visible without becoming a trajectory collision.

Only cross-asset robot/object pairs are evaluated.  Intended grasp contact is
ignored only when both link names are present in ``allowed_contact_links``;
for example, ``panda_leftfinger``--``handle`` may be allowed while an arm link
touching ``handle`` remains a collision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from .door_kinematics import forward_kinematics
from .affordances import CheckResult, GraspCandidate
from .transforms import compose_transforms, decompose_pose, pose_matrix, rpy_to_matrix
from .urdf_model import MeshReference, URDFModel, load_urdf, resolve_mesh_path


BACKEND_NAME = "named_urdf_cross_asset_sap_obb_v2"
_EPS = 1.0e-12
_SAT_AXIS_EPS = 1.0e-10
_NARROW_PHASE_NAME = "obb_sat_15_axes"


class CollisionError(ValueError):
    """Machine-readable failure which prevents an auditable collision check."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": dict(self.context)}


@dataclass(frozen=True, slots=True)
class CollisionShape:
    """One conservative local bound attached to a named URDF link."""

    asset: str
    link_name: str
    shape_id: str
    geometry_type: str
    local_transform: np.ndarray = field(repr=False)
    bounds_min: np.ndarray = field(repr=False)
    bounds_max: np.ndarray = field(repr=False)
    source: str

    def __post_init__(self) -> None:
        transform = _validate_transform(self.local_transform, f"{self.shape_id}.local_transform")
        minimum = _vector(self.bounds_min, 3, f"{self.shape_id}.bounds_min")
        maximum = _vector(self.bounds_max, 3, f"{self.shape_id}.bounds_max")
        if np.any(maximum < minimum):
            raise CollisionError(
                "GEOMETRY_BOUNDS_INVALID",
                "collision shape maximum bound is below its minimum bound",
                shape_id=self.shape_id,
                bounds_min=minimum.tolist(),
                bounds_max=maximum.tolist(),
            )
        for name, value in (
            ("local_transform", transform),
            ("bounds_min", minimum),
            ("bounds_max", maximum),
        ):
            frozen = np.asarray(value, dtype=float).copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)

    def world_bounds(self, link_world_transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return a conservative world AABB for this shape."""

        world_geometry = compose_transforms(
            _validate_transform(link_world_transform, f"link[{self.link_name}]"),
            self.local_transform,
        )
        corners = _bounds_corners(self.bounds_min, self.bounds_max)
        world_vertices = (
            world_geometry[:3, :3] @ corners.T
        ).T + world_geometry[:3, 3]
        return np.min(world_vertices, axis=0), np.max(world_vertices, axis=0)

    def world_obb(self, link_world_transform: np.ndarray) -> "OrientedBounds":
        """Return the authored local bound as a rotation-aware world OBB."""

        world_geometry = compose_transforms(
            _validate_transform(link_world_transform, f"link[{self.link_name}]"),
            self.local_transform,
        )
        local_center = 0.5 * (self.bounds_min + self.bounds_max)
        return OrientedBounds(
            center=world_geometry[:3, :3] @ local_center + world_geometry[:3, 3],
            axes=world_geometry[:3, :3],
            half_extents=0.5 * (self.bounds_max - self.bounds_min),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "link": self.link_name,
            "shape_id": self.shape_id,
            "geometry_type": self.geometry_type,
            "local_transform": self.local_transform.tolist(),
            "bounds_min": self.bounds_min.tolist(),
            "bounds_max": self.bounds_max.tolist(),
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class OrientedBounds:
    """World-space OBB represented by center, orthonormal axes, and half extents."""

    center: np.ndarray = field(repr=False)
    axes: np.ndarray = field(repr=False)
    half_extents: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        center = _vector(self.center, 3, "obb.center")
        half_extents = _vector(self.half_extents, 3, "obb.half_extents")
        if np.any(half_extents < 0.0):
            raise CollisionError(
                "GEOMETRY_BOUNDS_INVALID",
                "OBB half extents must be non-negative",
                half_extents=half_extents.tolist(),
            )
        try:
            axes = np.asarray(self.axes, dtype=float)
        except (TypeError, ValueError) as exc:
            raise CollisionError(
                "OBB_AXES_INVALID", "OBB axes must be a numeric 3x3 rotation"
            ) from exc
        if axes.shape != (3, 3) or not np.isfinite(axes).all():
            raise CollisionError(
                "OBB_AXES_INVALID",
                "OBB axes must be a finite 3x3 rotation",
                shape=list(axes.shape),
            )
        if not np.allclose(axes.T @ axes, np.eye(3), atol=1.0e-7, rtol=0.0):
            raise CollisionError("OBB_AXES_INVALID", "OBB axes are not orthonormal")
        if not math.isclose(float(np.linalg.det(axes)), 1.0, abs_tol=1.0e-7):
            raise CollisionError("OBB_AXES_INVALID", "OBB axes determinant is not +1")
        for name, value in (
            ("center", center),
            ("axes", axes),
            ("half_extents", half_extents),
        ):
            frozen = np.asarray(value, dtype=float).copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)


@dataclass(frozen=True, slots=True)
class CollisionPair:
    """One audited broad-phase candidate and its OBB narrow-phase outcome."""

    robot_link: str
    robot_shape: str
    object_link: str
    object_shape: str
    reason: str
    signed_clearance_m: float
    configured_margin_m: float
    allowed: bool
    broad_phase_candidate: bool = True
    broad_phase_signed_clearance_m: float | None = None
    narrow_phase: str = _NARROW_PHASE_NAME
    tested_separating_axes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_link": self.robot_link,
            "robot_shape": self.robot_shape,
            "object_link": self.object_link,
            "object_shape": self.object_shape,
            "reason": self.reason,
            "signed_clearance_m": self.signed_clearance_m,
            "configured_margin_m": self.configured_margin_m,
            "allowed": self.allowed,
            "broad_phase_candidate": self.broad_phase_candidate,
            "broad_phase_signed_clearance_m": self.broad_phase_signed_clearance_m,
            "narrow_phase": self.narrow_phase,
            "narrow_phase_outcome": self.reason,
            "tested_separating_axes": self.tested_separating_axes,
        }


@dataclass(frozen=True, slots=True)
class FrameCollisionResult:
    """Auditable result for one candidate or trajectory frame."""

    backend: str
    frame_index: int | None
    collision: bool
    pairs: tuple[CollisionPair, ...]
    allowed_pairs: tuple[CollisionPair, ...]
    checked_shape_pairs: int
    potential_shape_pairs: int
    reasons: tuple[str, ...]
    broad_phase_candidates: tuple[CollisionPair, ...] = ()
    sap_axis_candidate_pairs: int = 0
    obb_tested_shape_pairs: int = 0

    @property
    def collision_free(self) -> bool:
        return not self.collision

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "frame_index": self.frame_index,
            "collision": self.collision,
            "collision_free": self.collision_free,
            "broad_phase": "sap_world_aabb",
            "narrow_phase": _NARROW_PHASE_NAME,
            "checked_shape_pairs": self.checked_shape_pairs,
            "potential_shape_pairs": self.potential_shape_pairs,
            "sap_axis_candidate_pairs": self.sap_axis_candidate_pairs,
            "broad_phase_candidate_pairs": len(self.broad_phase_candidates),
            "obb_tested_shape_pairs": self.obb_tested_shape_pairs,
            "broad_phase_candidates": [
                pair.to_dict() for pair in self.broad_phase_candidates
            ],
            "pairs": [pair.to_dict() for pair in self.pairs],
            "allowed_pairs": [pair.to_dict() for pair in self.allowed_pairs],
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class CollisionReport:
    """Ordered frame results and the exact boolean flags consumed by metrics."""

    backend: str
    broad_phase: str
    margin_m: float
    allowed_contact_links: tuple[str, ...]
    flags: tuple[bool, ...]
    frames: tuple[FrameCollisionResult, ...]
    narrow_phase: str = _NARROW_PHASE_NAME

    @property
    def collision_frame_count(self) -> int:
        return sum(self.flags)

    @property
    def collision_frame_ratio(self) -> float:
        return self.collision_frame_count / len(self.flags) if self.flags else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "scope": "cross_asset_robot_object",
            "conservative": True,
            "broad_phase": self.broad_phase,
            "broad_phase_backend": "sap_world_aabb",
            "narrow_phase": self.narrow_phase,
            "margin_application": "once_on_obb_projected_separation",
            "margin_m": self.margin_m,
            "allowed_contact_links": list(self.allowed_contact_links),
            "frame_count": len(self.frames),
            "collision_frame_count": self.collision_frame_count,
            "collision_frame_ratio": self.collision_frame_ratio,
            "flags": list(self.flags),
            "frames": [frame.to_dict() for frame in self.frames],
        }


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise CollisionError(
            "INVALID_NUMBER", f"{field_name} must be numeric", field=field_name, value=value
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CollisionError(
            "INVALID_NUMBER", f"{field_name} must be numeric", field=field_name, value=value
        ) from exc
    if not math.isfinite(result):
        raise CollisionError(
            "NONFINITE_NUMBER", f"{field_name} must be finite", field=field_name, value=value
        )
    return result


def _vector(value: Any, size: int, field_name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CollisionError(
            "INVALID_VECTOR", f"{field_name} must contain {size} numeric values", field=field_name
        ) from exc
    if result.shape != (size,) or not np.isfinite(result).all():
        raise CollisionError(
            "INVALID_VECTOR",
            f"{field_name} must have shape ({size},) and contain only finite values",
            field=field_name,
            shape=list(result.shape),
        )
    return result


def _validate_transform(value: Any, field_name: str) -> np.ndarray:
    try:
        transform = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CollisionError(
            "TRANSFORM_INVALID", f"{field_name} must be a numeric 4x4 transform", field=field_name
        ) from exc
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise CollisionError(
            "TRANSFORM_INVALID",
            f"{field_name} must be a finite 4x4 transform",
            field=field_name,
            shape=list(transform.shape),
        )
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-9, rtol=0.0):
        raise CollisionError(
            "TRANSFORM_INVALID", f"{field_name} has an invalid homogeneous last row", field=field_name
        )
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-7, rtol=0.0):
        raise CollisionError(
            "TRANSFORM_INVALID", f"{field_name} rotation is not orthonormal", field=field_name
        )
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-7):
        raise CollisionError(
            "TRANSFORM_INVALID", f"{field_name} rotation determinant is not +1", field=field_name
        )
    return transform


def _parse_vector(
    text: str | None,
    *,
    field_name: str,
    default: tuple[float, float, float] | None = None,
) -> np.ndarray:
    if text is None:
        if default is None:
            raise CollisionError(
                "URDF_ATTRIBUTE_MISSING", f"missing {field_name}", field=field_name
            )
        return np.asarray(default, dtype=float)
    parts = text.split()
    if len(parts) != 3:
        raise CollisionError(
            "URDF_VECTOR_INVALID",
            f"{field_name} must contain exactly three values",
            field=field_name,
            value=text,
        )
    return np.asarray(
        [_finite_number(part, f"{field_name}[{index}]") for index, part in enumerate(parts)],
        dtype=float,
    )


def _origin_transform(entry: ET.Element, context: str) -> np.ndarray:
    origin = entry.find("origin")
    xyz = _parse_vector(
        origin.get("xyz") if origin is not None else None,
        field_name=f"{context}.origin.xyz",
        default=(0.0, 0.0, 0.0),
    )
    rpy = _parse_vector(
        origin.get("rpy") if origin is not None else None,
        field_name=f"{context}.origin.rpy",
        default=(0.0, 0.0, 0.0),
    )
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rpy_to_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def _bounds_corners(minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=float,
    )


def _load_obj_vertices(path: Path) -> np.ndarray:
    vertices: list[tuple[float, float, float]] = []
    try:
        with path.open("r", encoding="utf-8", errors="strict") as stream:
            for line_number, line in enumerate(stream, start=1):
                stripped = line.lstrip()
                if not stripped.startswith(("v ", "v\t")):
                    continue
                parts = stripped.split()
                if len(parts) < 4:
                    raise CollisionError(
                        "OBJ_VERTEX_INVALID",
                        "OBJ vertex must contain x, y, and z",
                        path=str(path),
                        line=line_number,
                    )
                values = tuple(
                    _finite_number(parts[index], f"OBJ[{line_number}][{index - 1}]")
                    for index in (1, 2, 3)
                )
                vertices.append(values)  # type: ignore[arg-type]
    except UnicodeError as exc:
        raise CollisionError(
            "MESH_READ_ERROR", f"cannot decode OBJ mesh: {exc}", path=str(path)
        ) from exc
    except OSError as exc:
        raise CollisionError(
            "MESH_READ_ERROR", f"cannot read mesh: {exc}", path=str(path)
        ) from exc
    if not vertices:
        raise CollisionError("MESH_EMPTY", "OBJ mesh has no vertices", path=str(path))
    return np.asarray(vertices, dtype=float)


def _load_mesh_vertices(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".obj":
        return _load_obj_vertices(path)
    try:
        import trimesh  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CollisionError(
            "MESH_FORMAT_UNSUPPORTED",
            "non-OBJ collision meshes require the declared trimesh dependency",
            path=str(path),
            suffix=path.suffix.lower(),
        ) from exc
    try:
        loaded = trimesh.load(path, process=False)
        if isinstance(loaded, trimesh.Scene):
            loaded = loaded.dump(concatenate=True)
        vertices = np.asarray(loaded.vertices, dtype=float)
    except Exception as exc:
        raise CollisionError(
            "MESH_LOAD_ERROR", f"cannot load collision mesh: {exc}", path=str(path)
        ) from exc
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise CollisionError("MESH_EMPTY", "collision mesh has no 3D vertices", path=str(path))
    if not np.isfinite(vertices).all():
        raise CollisionError(
            "MESH_VERTEX_NONFINITE", "collision mesh has non-finite vertices", path=str(path)
        )
    return vertices


def _infer_package_paths(urdf_path: Path, package_names: set[str]) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    ancestors = (urdf_path.parent, *urdf_path.parents)
    for package_name in sorted(package_names):
        for ancestor in ancestors:
            if ancestor.name == package_name:
                resolved[package_name] = ancestor
                break
            package_xml = ancestor / "package.xml"
            if package_xml.is_file():
                try:
                    package_root = ET.parse(package_xml).getroot()
                    authored_name = (package_root.findtext("name") or "").strip()
                except (OSError, ET.ParseError):
                    authored_name = ""
                if authored_name == package_name:
                    resolved[package_name] = ancestor
                    break
    return resolved


def _mesh_bounds(
    mesh: ET.Element,
    *,
    urdf_path: Path,
    link_name: str,
    package_paths: Mapping[str, str | Path],
    context: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    filename = (mesh.get("filename") or "").strip()
    if not filename:
        raise CollisionError(
            "MESH_FILENAME_MISSING", "collision mesh filename is missing", context=context
        )
    scale = _parse_vector(
        mesh.get("scale"), field_name=f"{context}.mesh.scale", default=(1.0, 1.0, 1.0)
    )
    if np.any(scale <= 0.0):
        raise CollisionError(
            "MESH_SCALE_INVALID",
            "collision mesh scale must contain positive values",
            context=context,
            scale=scale.tolist(),
        )
    reference = MeshReference(
        link_name=link_name,
        usage="collision",
        filename=filename,
        scale=tuple(float(value) for value in scale),  # type: ignore[arg-type]
    )
    resolved = resolve_mesh_path(reference, urdf_path, package_paths)
    if resolved is None:
        raise CollisionError(
            "MESH_URI_UNRESOLVED",
            "collision mesh URI could not be resolved by package name",
            filename=filename,
            link=link_name,
        )
    if not resolved.is_file():
        raise CollisionError(
            "MESH_FILE_NOT_FOUND",
            f"collision mesh file does not exist: {resolved}",
            filename=filename,
            resolved_path=str(resolved),
            link=link_name,
        )
    vertices = _load_mesh_vertices(resolved) * scale
    return np.min(vertices, axis=0), np.max(vertices, axis=0), str(resolved)


def load_collision_shapes(
    urdf_path: str | Path,
    *,
    asset: str,
    package_paths: Mapping[str, str | Path] | None = None,
) -> tuple[CollisionShape, ...]:
    """Load every authored collision shape without falling back to visuals."""

    path = Path(urdf_path).expanduser().resolve()
    # The shared parser checks duplicate names and malformed core URDF fields.
    model = load_urdf(path)
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:  # normally caught by load_urdf
        raise CollisionError("URDF_READ_ERROR", f"cannot parse URDF: {exc}", path=str(path)) from exc

    package_names: set[str] = set()
    for mesh in root.findall("link/collision/geometry/mesh"):
        filename = (mesh.get("filename") or "").strip()
        if filename.startswith("package://"):
            remainder = filename[len("package://") :]
            package_name = remainder.split("/", 1)[0]
            if package_name:
                package_names.add(package_name)
    resolved_packages: dict[str, str | Path] = {
        name: value for name, value in (package_paths or {}).items()
    }
    for name, value in _infer_package_paths(path, package_names).items():
        resolved_packages.setdefault(name, value)

    shapes: list[CollisionShape] = []
    link_elements = {(link.get("name") or "").strip(): link for link in root.findall("link")}
    for link_name in model.link_names:
        link = link_elements[link_name]
        for index, entry in enumerate(link.findall("collision")):
            context = f"link[{link_name}].collision[{index}]"
            geometry = entry.find("geometry")
            if geometry is None:
                raise CollisionError(
                    "COLLISION_GEOMETRY_MISSING",
                    "collision entry has no geometry",
                    asset=asset,
                    link=link_name,
                    index=index,
                )
            candidates = [
                (kind, geometry.find(kind))
                for kind in ("box", "cylinder", "sphere", "mesh")
                if geometry.find(kind) is not None
            ]
            if len(candidates) != 1:
                raise CollisionError(
                    "COLLISION_GEOMETRY_INVALID",
                    "collision entry must contain exactly one supported geometry",
                    asset=asset,
                    link=link_name,
                    index=index,
                    supported_shapes=[kind for kind, _ in candidates],
                )
            kind, element = candidates[0]
            assert element is not None
            source = f"{path}#{context}:{kind}"
            if kind == "box":
                size = _parse_vector(element.get("size"), field_name=f"{context}.box.size")
                if np.any(size <= 0.0):
                    raise CollisionError(
                        "GEOMETRY_DIMENSION_INVALID",
                        "box dimensions must be positive",
                        context=context,
                        size=size.tolist(),
                    )
                minimum, maximum = -0.5 * size, 0.5 * size
            elif kind == "cylinder":
                radius = _finite_number(element.get("radius"), f"{context}.cylinder.radius")
                length = _finite_number(element.get("length"), f"{context}.cylinder.length")
                if radius <= 0.0 or length <= 0.0:
                    raise CollisionError(
                        "GEOMETRY_DIMENSION_INVALID",
                        "cylinder radius and length must be positive",
                        context=context,
                        radius=radius,
                        length=length,
                    )
                minimum = np.asarray([-radius, -radius, -0.5 * length], dtype=float)
                maximum = -minimum
            elif kind == "sphere":
                radius = _finite_number(element.get("radius"), f"{context}.sphere.radius")
                if radius <= 0.0:
                    raise CollisionError(
                        "GEOMETRY_DIMENSION_INVALID",
                        "sphere radius must be positive",
                        context=context,
                        radius=radius,
                    )
                minimum = np.full(3, -radius, dtype=float)
                maximum = np.full(3, radius, dtype=float)
            else:
                minimum, maximum, resolved_mesh = _mesh_bounds(
                    element,
                    urdf_path=path,
                    link_name=link_name,
                    package_paths=resolved_packages,
                    context=context,
                )
                source = f"{source}:{resolved_mesh}"
            shapes.append(
                CollisionShape(
                    asset=asset,
                    link_name=link_name,
                    shape_id=f"{asset}:{link_name}:collision[{index}]",
                    geometry_type=kind,
                    local_transform=_origin_transform(entry, context),
                    bounds_min=minimum,
                    bounds_max=maximum,
                    source=source,
                )
            )
    if not shapes:
        raise CollisionError(
            "COLLISION_GEOMETRY_MISSING",
            "URDF has no authored collision geometry; visuals are not collision data",
            asset=asset,
            path=str(path),
        )
    return tuple(shapes)


def named_link_fk(
    model_or_path: URDFModel | str | Path,
    world_transform: np.ndarray,
    joint_positions_by_name: Mapping[str, float],
) -> dict[str, np.ndarray]:
    """Run dependency-light URDF FK and return ``link name -> T_world_link``.

    The underlying routine supports fixed, revolute/continuous, and prismatic
    joints and never accepts an integer joint or body index.
    """

    model = model_or_path if isinstance(model_or_path, URDFModel) else load_urdf(model_or_path)
    return forward_kinematics(model, world_transform, joint_positions_by_name)


def _signed_aabb_clearance(
    a_min: np.ndarray,
    a_max: np.ndarray,
    b_min: np.ndarray,
    b_max: np.ndarray,
) -> float:
    """Positive Euclidean separation or negative minimum overlap depth."""

    gaps = np.maximum(np.maximum(a_min - b_max, b_min - a_max), 0.0)
    if np.any(gaps > 0.0):
        return float(np.linalg.norm(gaps))
    overlaps = np.minimum(a_max, b_max) - np.maximum(a_min, b_min)
    return -float(np.min(overlaps))


def _obb_sat_signed_clearance(
    first: OrientedBounds,
    second: OrientedBounds,
) -> tuple[float, int]:
    """Return OBB projected separation and number of non-degenerate SAT axes.

    Positive separation means at least one separating axis exists.  Zero means
    touching, and a negative value is the least projected overlap among the
    tested axes.  The caller compares the result to the configured margin once.
    Local AABB-derived OBBs require the complete box SAT basis: three axes from
    each box and all nine pairwise cross products.
    """

    candidate_axes = [first.axes[:, index] for index in range(3)]
    candidate_axes.extend(second.axes[:, index] for index in range(3))
    candidate_axes.extend(
        np.cross(first.axes[:, first_index], second.axes[:, second_index])
        for first_index in range(3)
        for second_index in range(3)
    )
    center_delta = second.center - first.center
    maximum_separation = -math.inf
    tested_axes = 0
    for authored_axis in candidate_axes:
        norm = float(np.linalg.norm(authored_axis))
        if norm <= _SAT_AXIS_EPS:
            # Parallel box axes produce a zero cross product and do not define
            # an additional separating direction.
            continue
        axis = authored_axis / norm
        first_radius = float(
            np.dot(first.half_extents, np.abs(first.axes.T @ axis))
        )
        second_radius = float(
            np.dot(second.half_extents, np.abs(second.axes.T @ axis))
        )
        projected_distance = abs(float(np.dot(center_delta, axis)))
        separation = projected_distance - first_radius - second_radius
        maximum_separation = max(maximum_separation, separation)
        tested_axes += 1
    if tested_axes == 0 or not math.isfinite(maximum_separation):
        raise CollisionError(
            "OBB_SAT_INVALID",
            "OBB separating-axis test produced no finite non-degenerate axes",
        )
    return maximum_separation, tested_axes


class NamedAABBCollisionChecker:
    """SAP broad phase plus OBB-SAT narrow phase, keyed entirely by names.

    The historical class name is retained as a public runner adapter.  AABB
    results never directly set collision flags in this backend version.
    """

    backend = BACKEND_NAME

    def __init__(
        self,
        robot_urdf: str | Path,
        object_urdf: str | Path,
        *,
        broad_phase: str,
        margin_m: float,
        allowed_contact_links: Sequence[str],
        robot_package_paths: Mapping[str, str | Path] | None = None,
        object_package_paths: Mapping[str, str | Path] | None = None,
    ) -> None:
        if not isinstance(broad_phase, str) or broad_phase.strip().lower() != "sap":
            raise CollisionError(
                "BROAD_PHASE_UNSUPPORTED",
                "named SAP/OBB collision backend requires broad_phase='sap'",
                broad_phase=broad_phase,
                supported=["sap"],
            )
        self.broad_phase = "sap"
        margin = _finite_number(margin_m, "margin_m")
        if margin < 0.0:
            raise CollisionError(
                "MARGIN_INVALID", "collision margin must be non-negative", margin_m=margin
            )
        if isinstance(allowed_contact_links, (str, bytes)):
            raise CollisionError(
                "ALLOWED_LINKS_INVALID", "allowed_contact_links must be a sequence of link names"
            )
        names = tuple(str(name).strip() for name in allowed_contact_links)
        if any(not name for name in names) or len(set(names)) != len(names):
            raise CollisionError(
                "ALLOWED_LINKS_INVALID",
                "allowed_contact_links must contain unique non-empty link names",
                values=list(allowed_contact_links),
            )
        self.margin_m = margin
        self.allowed_contact_links = tuple(sorted(names))
        self._allowed = frozenset(names)
        self.robot_shapes = load_collision_shapes(
            robot_urdf, asset="robot", package_paths=robot_package_paths
        )
        self.object_shapes = load_collision_shapes(
            object_urdf, asset="object", package_paths=object_package_paths
        )
        known_links = {shape.link_name for shape in self.robot_shapes + self.object_shapes}
        unknown = sorted(self._allowed - known_links)
        if unknown:
            raise CollisionError(
                "ALLOWED_LINK_NOT_FOUND",
                "allowed contact link is not present among named collision links",
                unknown_links=unknown,
                known_collision_links=sorted(known_links),
            )

    @staticmethod
    def _validate_link_transforms(
        transforms: Mapping[str, np.ndarray],
        shapes: tuple[CollisionShape, ...],
        asset: str,
    ) -> None:
        if not isinstance(transforms, Mapping):
            raise CollisionError(
                "LINK_TRANSFORMS_INVALID",
                f"{asset}_link_transforms must be a name-to-transform mapping",
                asset=asset,
            )
        required = {shape.link_name for shape in shapes}
        missing = sorted(required - set(transforms))
        if missing:
            raise CollisionError(
                "LINK_TRANSFORM_MISSING",
                "collision check cannot continue with missing named-link transforms",
                asset=asset,
                missing_links=missing,
            )
        for name in sorted(required):
            _validate_transform(transforms[name], f"{asset}_link_transforms[{name!r}]")

    def check_frame(
        self,
        robot_link_transforms: Mapping[str, np.ndarray],
        object_link_transforms: Mapping[str, np.ndarray],
        *,
        frame_index: int | None = None,
    ) -> FrameCollisionResult:
        """Check all configured cross-asset shape pairs for one frame."""

        if frame_index is not None and (
            isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0
        ):
            raise CollisionError(
                "FRAME_INDEX_INVALID", "frame_index must be a non-negative integer or None"
            )
        self._validate_link_transforms(robot_link_transforms, self.robot_shapes, "robot")
        self._validate_link_transforms(object_link_transforms, self.object_shapes, "object")
        robot_bounds = {
            shape.shape_id: shape.world_bounds(robot_link_transforms[shape.link_name])
            for shape in self.robot_shapes
        }
        object_bounds = {
            shape.shape_id: shape.world_bounds(object_link_transforms[shape.link_name])
            for shape in self.object_shapes
        }
        robot_obbs = {
            shape.shape_id: shape.world_obb(robot_link_transforms[shape.link_name])
            for shape in self.robot_shapes
        }
        object_obbs = {
            shape.shape_id: shape.world_obb(object_link_transforms[shape.link_name])
            for shape in self.object_shapes
        }
        # Deterministic sweep-and-prune on world X.  Full 3-D conservative
        # AABB clearance is evaluated only for pairs whose X intervals are
        # within the configured margin.
        object_sweep = sorted(
            self.object_shapes,
            key=lambda shape: (object_bounds[shape.shape_id][0][0], shape.shape_id),
        )
        collisions: list[CollisionPair] = []
        allowed: list[CollisionPair] = []
        broad_phase_candidates: list[CollisionPair] = []
        sap_axis_candidates = 0
        checked = 0
        for robot_shape in self.robot_shapes:
            r_min, r_max = robot_bounds[robot_shape.shape_id]
            for object_shape in object_sweep:
                o_min, o_max = object_bounds[object_shape.shape_id]
                if o_min[0] > r_max[0] + self.margin_m + _EPS:
                    break
                if o_max[0] < r_min[0] - self.margin_m - _EPS:
                    continue
                sap_axis_candidates += 1
                broad_phase_clearance = _signed_aabb_clearance(
                    r_min, r_max, o_min, o_max
                )
                if broad_phase_clearance > self.margin_m + _EPS:
                    continue
                checked += 1
                clearance, tested_axes = _obb_sat_signed_clearance(
                    robot_obbs[robot_shape.shape_id],
                    object_obbs[object_shape.shape_id],
                )
                is_allowed = (
                    robot_shape.link_name in self._allowed
                    and object_shape.link_name in self._allowed
                )
                if clearance <= _EPS:
                    reason = "obb_overlap"
                elif clearance <= self.margin_m + _EPS:
                    reason = "within_margin"
                else:
                    reason = "obb_separated"
                pair = CollisionPair(
                    robot_link=robot_shape.link_name,
                    robot_shape=robot_shape.shape_id,
                    object_link=object_shape.link_name,
                    object_shape=object_shape.shape_id,
                    reason=reason,
                    signed_clearance_m=clearance,
                    configured_margin_m=self.margin_m,
                    allowed=is_allowed,
                    broad_phase_candidate=True,
                    broad_phase_signed_clearance_m=broad_phase_clearance,
                    narrow_phase=_NARROW_PHASE_NAME,
                    tested_separating_axes=tested_axes,
                )
                broad_phase_candidates.append(pair)
                if reason == "obb_separated":
                    continue
                (allowed if is_allowed else collisions).append(pair)
        collisions.sort(key=lambda pair: (pair.robot_shape, pair.object_shape))
        allowed.sort(key=lambda pair: (pair.robot_shape, pair.object_shape))
        broad_phase_candidates.sort(
            key=lambda pair: (pair.robot_shape, pair.object_shape)
        )
        reasons = tuple(sorted({pair.reason for pair in collisions}))
        return FrameCollisionResult(
            backend=self.backend,
            frame_index=frame_index,
            collision=bool(collisions),
            pairs=tuple(collisions),
            allowed_pairs=tuple(allowed),
            checked_shape_pairs=checked,
            potential_shape_pairs=len(self.robot_shapes) * len(self.object_shapes),
            reasons=reasons,
            broad_phase_candidates=tuple(broad_phase_candidates),
            sap_axis_candidate_pairs=sap_axis_candidates,
            obb_tested_shape_pairs=checked,
        )

    def check_candidate(
        self,
        robot_link_transforms: Mapping[str, np.ndarray],
        object_link_transforms: Mapping[str, np.ndarray],
    ) -> FrameCollisionResult:
        """Candidate-selection alias with the same fail-closed semantics."""

        return self.check_frame(robot_link_transforms, object_link_transforms)

    def check_trajectory(
        self,
        robot_link_transforms: Sequence[Mapping[str, np.ndarray]],
        object_link_transforms: Sequence[Mapping[str, np.ndarray]],
    ) -> CollisionReport:
        """Check ordered named-link FK maps and return exact per-frame flags."""

        if len(robot_link_transforms) != len(object_link_transforms):
            raise CollisionError(
                "TRAJECTORY_LENGTH_MISMATCH",
                "robot and object link-transform trajectories must have equal length",
                robot_frame_count=len(robot_link_transforms),
                object_frame_count=len(object_link_transforms),
            )
        if not robot_link_transforms:
            raise CollisionError(
                "TRAJECTORY_EMPTY", "collision trajectory must contain at least one frame"
            )
        frames = tuple(
            self.check_frame(robot_frame, object_frame, frame_index=index)
            for index, (robot_frame, object_frame) in enumerate(
                zip(robot_link_transforms, object_link_transforms, strict=True)
            )
        )
        return CollisionReport(
            backend=self.backend,
            broad_phase=self.broad_phase,
            margin_m=self.margin_m,
            allowed_contact_links=self.allowed_contact_links,
            flags=tuple(frame.collision for frame in frames),
            frames=frames,
        )


class NamedAABBCollisionEvaluator:
    """Adapter from the runner's task-level API to named-link collision FK.

    The runner provides arm coordinates in the same order as the configured
    arm *names*.  This adapter immediately reconstructs a name-to-coordinate
    mapping, adds the two named Franka finger joints at half of the configured
    total opening, computes URDF FK, and delegates to
    :class:`NamedAABBCollisionChecker`.
    """

    def __init__(self, config: Any, kinematics: Any, backend: Any) -> None:
        self.config = config
        self.kinematics = kinematics
        self.ik_backend = backend
        self.arm_joint_names = tuple(
            str(name) for name in config.get("assets.robot.arm_joint_names")
        )
        self.finger_joint_names = tuple(
            str(name) for name in config.get("assets.robot.finger_joint_names")
        )
        if len(self.finger_joint_names) != 2 or len(set(self.finger_joint_names)) != 2:
            raise CollisionError(
                "FINGER_JOINT_CONFIG_INVALID",
                "Franka collision FK requires exactly two unique named finger joints",
                finger_joint_names=list(self.finger_joint_names),
            )
        if not self.arm_joint_names or len(set(self.arm_joint_names)) != len(
            self.arm_joint_names
        ):
            raise CollisionError(
                "ARM_JOINT_CONFIG_INVALID",
                "collision FK requires unique configured arm joint names",
                arm_joint_names=list(self.arm_joint_names),
            )
        self.robot_model = load_urdf(config.asset_path("robot"))
        robot_pose = config.get("assets.robot.world_pose")
        self.robot_world_transform = pose_matrix(
            robot_pose["position"], robot_pose["orientation_wxyz"]
        )
        self.checker = NamedAABBCollisionChecker(
            config.asset_path("robot"),
            config.asset_path("object"),
            broad_phase=str(config.get("collision.broad_phase")),
            margin_m=float(config.get("collision.margin_m")),
            allowed_contact_links=tuple(
                str(name) for name in config.get("collision.allowed_contact_links")
            ),
        )
        self.candidate_results: dict[str, FrameCollisionResult] = {}
        self.last_trajectory_report: CollisionReport | None = None

    def _joint_positions(
        self, arm_joint_q: Sequence[float], gripper_width_m: float
    ) -> dict[str, float]:
        try:
            arm = np.asarray(arm_joint_q, dtype=float)
        except (TypeError, ValueError) as exc:
            raise CollisionError(
                "ARM_JOINT_VALUES_INVALID",
                "arm joint values must be numeric",
            ) from exc
        if arm.shape != (len(self.arm_joint_names),) or not np.isfinite(arm).all():
            raise CollisionError(
                "ARM_JOINT_VALUES_INVALID",
                "arm joint values must match configured names and be finite",
                expected=len(self.arm_joint_names),
                shape=list(arm.shape),
            )
        width = _finite_number(gripper_width_m, "gripper_width_m")
        if width < 0.0:
            raise CollisionError(
                "GRIPPER_WIDTH_INVALID",
                "gripper width must be non-negative",
                gripper_width_m=width,
            )
        named = {
            name: float(value)
            for name, value in zip(self.arm_joint_names, arm, strict=True)
        }
        # Franka's two URDF prismatic finger coordinates are each half of the
        # total symmetric opening; names, rather than integer indices, select them.
        for name in self.finger_joint_names:
            named[name] = 0.5 * width
        return named

    def _robot_fk(
        self, arm_joint_q: Sequence[float], gripper_width_m: float
    ) -> dict[str, np.ndarray]:
        return named_link_fk(
            self.robot_model,
            self.robot_world_transform,
            self._joint_positions(arm_joint_q, gripper_width_m),
        )

    def candidate_is_collision_free(
        self,
        candidate: GraspCandidate,
        gripper_world: np.ndarray,
    ) -> CheckResult:
        """Solve candidate IK, reconstruct all named-link poses, and check it."""

        try:
            position, orientation_wxyz = decompose_pose(gripper_world)
            trajectory = self.ik_backend.solve_waypoints([position], [orientation_wxyz])
            waypoints = tuple(getattr(trajectory, "waypoints", ()))
            if len(waypoints) != 1:
                return CheckResult(
                    False,
                    "candidate collision IK returned an invalid waypoint count",
                    {"waypoint_count": len(waypoints), "backend": self.checker.backend},
                )
            waypoint = waypoints[0]
            validation = getattr(waypoint, "validation", None)
            if not bool(getattr(validation, "success", False)):
                return CheckResult(
                    False,
                    "candidate collision FK unavailable because IK validation failed",
                    {
                        "failed_checks": list(
                            getattr(validation, "failed_checks", ())
                        ),
                        "backend": self.checker.backend,
                    },
                )
            arm_joint_q = tuple(getattr(waypoint, "arm_joint_positions"))
            robot_links = self._robot_fk(arm_joint_q, candidate.gripper_width_m)
            closed_angle_rad = math.radians(
                float(self.config.get("task.closed_angle_deg"))
            )
            object_links = self.kinematics.link_transforms(closed_angle_rad)
            result = self.checker.check_candidate(robot_links, object_links)
            self.candidate_results[candidate.candidate_id] = result
            details = result.to_dict()
            details["candidate_id"] = candidate.candidate_id
            return CheckResult(
                result.collision_free,
                None
                if result.collision_free
                else "candidate has a conservative named-geometry collision",
                details,
            )
        except Exception as exc:
            details: dict[str, Any] = {
                "backend": self.checker.backend,
                "exception_type": type(exc).__name__,
                "error": str(exc),
            }
            if isinstance(exc, CollisionError):
                details["collision_error"] = exc.to_dict()
            return CheckResult(
                False,
                "candidate collision check could not be completed",
                details,
            )

    def trajectory_collision_flags(
        self,
        plan: Any,
        arm_joint_q: np.ndarray,
    ) -> np.ndarray:
        """Return checked flags for every task frame, never placeholder zeros."""

        arm_rows = np.asarray(arm_joint_q, dtype=float)
        frame_count = len(plan.phase_names)
        expected = (frame_count, len(self.arm_joint_names))
        if arm_rows.shape != expected:
            raise CollisionError(
                "TRAJECTORY_JOINT_SHAPE_INVALID",
                "arm trajectory shape does not match phase frames and configured names",
                expected=list(expected),
                actual=list(arm_rows.shape),
            )
        widths = np.asarray(plan.gripper_width_m, dtype=float)
        door_angles = np.asarray(plan.door_angle_rad, dtype=float)
        if widths.shape != (frame_count,) or door_angles.shape != (frame_count,):
            raise CollisionError(
                "TRAJECTORY_STATE_SHAPE_INVALID",
                "gripper width and door angle must contain one value per frame",
                frame_count=frame_count,
                width_shape=list(widths.shape),
                door_shape=list(door_angles.shape),
            )
        robot_frames = [
            self._robot_fk(arm_rows[index], float(widths[index]))
            for index in range(frame_count)
        ]
        object_frames = [
            self.kinematics.link_transforms(float(door_angles[index]))
            for index in range(frame_count)
        ]
        report = self.checker.check_trajectory(robot_frames, object_frames)
        self.last_trajectory_report = report
        return np.asarray(report.flags, dtype=bool)


def build_collision_evaluator(
    config: Any,
    kinematics: Any,
    backend: Any,
) -> NamedAABBCollisionEvaluator:
    """Build the concrete evaluator requested by :mod:`src.run`."""

    return NamedAABBCollisionEvaluator(config, kinematics, backend)


__all__ = [
    "BACKEND_NAME",
    "CollisionError",
    "CollisionPair",
    "CollisionReport",
    "CollisionShape",
    "FrameCollisionResult",
    "NamedAABBCollisionChecker",
    "NamedAABBCollisionEvaluator",
    "OrientedBounds",
    "build_collision_evaluator",
    "load_collision_shapes",
    "named_link_fk",
]
