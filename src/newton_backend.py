"""Newton v1.3.0 adapter for deterministic Franka-only inverse kinematics.

All project-facing quaternions use ``wxyz``.  Conversion to Newton's ``xyzw``
layout happens only at this module boundary.  The numerical validation helpers
remain importable without Newton/Warp so local checks do not require the remote
simulation environment.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import FailureCode, PipelineError
from .ik_objectives import (
    JointNominalObjective,
    JointReferenceObjective,
    build_scalar_dof_to_coord_map,
)
from .transforms import (
    compose_transforms,
    decompose_pose,
    normalize_quaternion,
    pose_matrix,
)
from .urdf_model import URDFModelError, load_urdf


try:  # Newton is installed on the remote runtime, not necessarily on laptops.
    import newton
    import warp as wp
except (ImportError, OSError) as exc:  # Native extension loading can raise OSError.
    newton = None  # type: ignore[assignment]
    wp = None  # type: ignore[assignment]
    _NEWTON_IMPORT_ERROR: BaseException | None = exc
else:
    _NEWTON_IMPORT_ERROR = None


@dataclass(frozen=True, slots=True)
class ResolvedLabel:
    """A uniquely resolved Newton model label."""

    index: int
    label: str


@dataclass(frozen=True, slots=True)
class PoseError:
    """Post-FK Cartesian error measured from the realized TCP pose."""

    position_m: float
    orientation_rad: float

    @property
    def orientation_deg(self) -> float:
        return math.degrees(self.orientation_rad)


@dataclass(frozen=True, slots=True)
class JointLimitViolation:
    """One realized scalar DoF outside its configured Newton limits."""

    dof_index: int
    coord_index: int
    value: float
    lower: float
    upper: float
    magnitude: float


@dataclass(frozen=True, slots=True)
class JointVelocityViolation:
    """One proposed scalar-DoF step exceeding its URDF velocity limit."""

    dof_index: int
    coord_index: int
    previous: float
    candidate: float
    delta: float
    max_delta: float
    requested_velocity: float
    velocity_limit: float

    @property
    def limit_ratio(self) -> float:
        if self.velocity_limit == 0.0:
            return math.inf if self.requested_velocity > 0.0 else 0.0
        return self.requested_velocity / self.velocity_limit


class JointMotionInfeasibleError(ValueError):
    """Raised when position/velocity/acceleration/jerk bounds have no overlap."""

    def __init__(self, message: str, *, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = dict(details)


@dataclass(frozen=True, slots=True)
class JointMotionProjection:
    """One projected joint state and its finite-difference motion state."""

    joint_q: np.ndarray
    joint_velocity: np.ndarray
    joint_acceleration: np.ndarray
    joint_jerk: np.ndarray
    projected_dof_indices: tuple[int, ...]
    diagnostics: tuple["JointMotionProjectionDiagnostic", ...] = ()


@dataclass(frozen=True, slots=True)
class JointMotionProjectionDiagnostic:
    """Raw-to-realized motion evidence for one projected scalar DoF."""

    dof_index: int
    coord_index: int
    raw_candidate_q: float
    projected_q: float
    q_correction_rad: float
    raw_requested_velocity_rad_s: float
    raw_requested_acceleration_rad_s2: float
    raw_requested_jerk_rad_s3: float
    projected_velocity_rad_s: float
    projected_acceleration_rad_s2: float
    projected_jerk_rad_s3: float
    trigger_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WaypointValidation:
    """Independent acceptance decision made after Newton forward kinematics."""

    success: bool
    pose_error: PoseError
    joint_limit_violations: tuple[JointLimitViolation, ...]
    has_nonfinite: bool
    failed_checks: tuple[str, ...]
    failure_codes: tuple[FailureCode, ...]

    @property
    def max_joint_limit_violation(self) -> float:
        if not self.joint_limit_violations:
            return 0.0
        return max(item.magnitude for item in self.joint_limit_violations)


@dataclass(frozen=True, slots=True)
class IKWaypointResult:
    """Solver output and its independently verified realized pose."""

    waypoint_index: int
    joint_positions: tuple[float, ...]
    arm_joint_positions: tuple[float, ...]
    target_position: tuple[float, float, float]
    target_orientation_wxyz: tuple[float, float, float, float]
    actual_position: tuple[float, float, float]
    actual_orientation_wxyz: tuple[float, float, float, float]
    objective_cost: float
    validation: WaypointValidation
    velocity_projection_applied: bool = False
    raw_joint_velocity_violations: tuple[JointVelocityViolation, ...] = ()
    motion_projection_applied: bool = False
    projected_joint_dof_indices: tuple[int, ...] = ()
    motion_projection_diagnostics: tuple[
        JointMotionProjectionDiagnostic, ...
    ] = ()


@dataclass(frozen=True, slots=True)
class IKTrajectoryResult:
    """Ordered waypoint results from one deterministic warm-started solve."""

    waypoints: tuple[IKWaypointResult, ...]

    @property
    def success_count(self) -> int:
        return sum(result.validation.success for result in self.waypoints)

    @property
    def success_rate(self) -> float:
        if not self.waypoints:
            return 0.0
        return self.success_count / len(self.waypoints)

    @property
    def all_successful(self) -> bool:
        return bool(self.waypoints) and self.success_count == len(self.waypoints)

    def meets_success_rate(self, minimum: float) -> bool:
        threshold = float(minimum)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("minimum success rate must be finite and in [0, 1]")
        return self.success_rate >= threshold


def _vector(
    values: Iterable[float],
    size: int,
    name: str,
    *,
    require_finite: bool,
) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {size} numeric values") from exc
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    if require_finite and not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _nonnegative_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def resolve_unique_label(
    labels: Sequence[str], configured_name: str, *, kind: str
) -> ResolvedLabel:
    """Resolve one Newton label by exact name or ``/...`` suffix.

    URDF import commonly prefixes labels with the robot name.  Exact and
    suffix matches are considered together and the result must be unique;
    first-match behavior would silently select the wrong articulation.
    """

    if not isinstance(configured_name, str) or not configured_name.strip():
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            f"configured {kind} name must be a non-empty string",
            stage="newton_model",
            details={"kind": kind, "configured_name": configured_name},
        )
    name = configured_name.strip()
    suffix = f"/{name}"
    matches = [
        ResolvedLabel(index=index, label=label)
        for index, label in enumerate(labels)
        if isinstance(label, str) and (label == name or label.endswith(suffix))
    ]
    if not matches:
        raise PipelineError(
            FailureCode.FRAME_MISSING,
            f"Newton {kind} label not found: {name}",
            stage="newton_model",
            details={"kind": kind, "configured_name": name, "labels": list(labels)},
        )
    if len(matches) != 1:
        raise PipelineError(
            FailureCode.NAME_NOT_UNIQUE,
            f"Newton {kind} label is ambiguous: {name}",
            stage="newton_model",
            details={
                "kind": kind,
                "configured_name": name,
                "matches": [
                    {"index": match.index, "label": match.label} for match in matches
                ],
            },
        )
    return matches[0]


def quaternion_wxyz_to_xyzw(quaternion_wxyz: Iterable[float]) -> np.ndarray:
    """Reorder one finite quaternion from project ``wxyz`` to Newton ``xyzw``."""

    w, x, y, z = _vector(
        quaternion_wxyz, 4, "quaternion_wxyz", require_finite=True
    )
    return np.asarray([x, y, z, w], dtype=float)


def quaternion_xyzw_to_wxyz(quaternion_xyzw: Iterable[float]) -> np.ndarray:
    """Reorder one finite quaternion from Newton ``xyzw`` to project ``wxyz``."""

    x, y, z, w = _vector(
        quaternion_xyzw, 4, "quaternion_xyzw", require_finite=True
    )
    return np.asarray([w, x, y, z], dtype=float)


def quaternion_angle_rad_xyzw(
    actual_xyzw: Iterable[float], target_xyzw: Iterable[float]
) -> float:
    """Return shortest angular distance using ``2*acos(abs(dot(q1, q2)))``."""

    actual = _vector(actual_xyzw, 4, "actual_xyzw", require_finite=True)
    target = _vector(target_xyzw, 4, "target_xyzw", require_finite=True)
    actual_norm = float(np.linalg.norm(actual))
    target_norm = float(np.linalg.norm(target))
    if actual_norm == 0.0 or target_norm == 0.0:
        raise ValueError("orientation quaternions must have non-zero norm")
    dot = abs(float(np.dot(actual / actual_norm, target / target_norm)))
    return 2.0 * math.acos(float(np.clip(dot, 0.0, 1.0)))


def compute_pose_error(
    actual_position: Iterable[float],
    actual_orientation_xyzw: Iterable[float],
    target_position: Iterable[float],
    target_orientation_xyzw: Iterable[float],
) -> PoseError:
    """Measure realized-vs-target position and orientation errors."""

    actual_pos = _vector(actual_position, 3, "actual_position", require_finite=True)
    target_pos = _vector(target_position, 3, "target_position", require_finite=True)
    return PoseError(
        position_m=float(np.linalg.norm(actual_pos - target_pos)),
        orientation_rad=quaternion_angle_rad_xyzw(
            actual_orientation_xyzw, target_orientation_xyzw
        ),
    )


_REPEATED_TARGET_POSITION_ATOL_M = 1.0e-10
_REPEATED_TARGET_QUATERNION_ATOL = 1.0e-10


def _cartesian_targets_equivalent(
    previous_position: np.ndarray,
    previous_orientation_wxyz: np.ndarray,
    current_position: np.ndarray,
    current_orientation_wxyz: np.ndarray,
) -> bool:
    """Return whether two normalized Cartesian targets are numerically identical.

    The tolerance is deliberately far tighter than IK acceptance: this only
    recognizes a generated hold waypoint, never ordinary slow Cartesian
    motion.  Unit quaternions use the double cover, so ``q`` and ``-q`` are
    treated as the same target orientation.
    """

    if not np.allclose(
        previous_position,
        current_position,
        rtol=0.0,
        atol=_REPEATED_TARGET_POSITION_ATOL_M,
    ):
        return False
    return bool(
        np.allclose(
            previous_orientation_wxyz,
            current_orientation_wxyz,
            rtol=0.0,
            atol=_REPEATED_TARGET_QUATERNION_ATOL,
        )
        or np.allclose(
            previous_orientation_wxyz,
            -current_orientation_wxyz,
            rtol=0.0,
            atol=_REPEATED_TARGET_QUATERNION_ATOL,
        )
    )


def joint_limit_violations(
    joint_q: Sequence[float] | np.ndarray,
    joint_limit_lower: Sequence[float] | np.ndarray,
    joint_limit_upper: Sequence[float] | np.ndarray,
    dof_to_coord: Sequence[int] | np.ndarray,
    *,
    tolerance: float,
) -> tuple[JointLimitViolation, ...]:
    """Return true post-solve scalar limit violations without projecting ``q``."""

    tol = _nonnegative_finite(tolerance, "joint-limit tolerance")
    q = np.asarray(joint_q, dtype=float)
    lower = np.asarray(joint_limit_lower, dtype=float)
    upper = np.asarray(joint_limit_upper, dtype=float)
    mapping_raw = np.asarray(dof_to_coord)
    if q.ndim != 1:
        raise ValueError(f"joint_q must be one-dimensional, got {q.shape}")
    if lower.ndim != 1 or upper.shape != lower.shape:
        raise ValueError("joint limit arrays must be one-dimensional and have equal shape")
    if mapping_raw.ndim != 1 or len(mapping_raw) != len(lower):
        raise ValueError("dof_to_coord must be one-dimensional and match joint limits")
    try:
        mapping = mapping_raw.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("dof_to_coord must contain integer indices") from exc
    if not np.array_equal(mapping_raw, mapping):
        raise ValueError("dof_to_coord must contain integer indices")
    if np.any(mapping < 0) or np.any(mapping >= len(q)):
        raise ValueError("dof_to_coord contains an out-of-range coordinate index")
    if np.isnan(lower).any() or np.isnan(upper).any():
        raise ValueError("joint limits cannot contain NaN")
    if np.any(lower > upper):
        raise ValueError("joint lower limits cannot exceed upper limits")

    violations: list[JointLimitViolation] = []
    for dof_index, coord_index_value in enumerate(mapping):
        coord_index = int(coord_index_value)
        value = float(q[coord_index])
        if not math.isfinite(value):
            continue
        low = float(lower[dof_index])
        high = float(upper[dof_index])
        magnitude = max(low - value, value - high, 0.0)
        if magnitude > tol:
            violations.append(
                JointLimitViolation(
                    dof_index=dof_index,
                    coord_index=coord_index,
                    value=value,
                    lower=low,
                    upper=high,
                    magnitude=magnitude,
                )
            )
    return tuple(violations)


def project_scalar_joint_limits(
    joint_q: Sequence[float] | np.ndarray,
    joint_limit_lower: Sequence[float] | np.ndarray,
    joint_limit_upper: Sequence[float] | np.ndarray,
    dof_to_coord: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Project scalar joint coordinates onto finite URDF hard limits.

    Newton's floating-point optimizer can finish a few micro-radians outside a
    limit even with a joint-limit objective.  Projection makes the hard
    constraint exact; callers must still run fresh FK afterwards because a
    projected pose can fail the Cartesian tolerance.
    """

    coordinates = np.asarray(joint_q, dtype=float)
    lower = np.asarray(joint_limit_lower, dtype=float)
    upper = np.asarray(joint_limit_upper, dtype=float)
    mapping = np.asarray(dof_to_coord, dtype=np.int64)
    if coordinates.ndim != 1:
        raise ValueError("joint_q must be one-dimensional")
    if lower.shape != upper.shape or lower.shape != mapping.shape:
        raise ValueError("joint limits and dof_to_coord must have matching shapes")
    if np.any(mapping < 0) or np.any(mapping >= len(coordinates)):
        raise ValueError("dof_to_coord contains an out-of-range coordinate index")
    projected = coordinates.copy()
    if not np.isfinite(projected).all():
        return projected
    for dof_index, coord_index in enumerate(mapping):
        value = projected[int(coord_index)]
        if math.isfinite(float(lower[dof_index])):
            value = max(value, float(lower[dof_index]))
        if math.isfinite(float(upper[dof_index])):
            value = min(value, float(upper[dof_index]))
        projected[int(coord_index)] = value
    return projected


def _joint_velocity_inputs(
    candidate_joint_q: Sequence[float] | np.ndarray,
    previous_joint_q: Sequence[float] | np.ndarray,
    joint_velocity_limits: Sequence[float] | np.ndarray,
    dof_to_coord: Sequence[int] | np.ndarray,
    active_mask: Sequence[float] | np.ndarray | None,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    candidate = np.asarray(candidate_joint_q, dtype=float)
    previous = np.asarray(previous_joint_q, dtype=float)
    limits = np.asarray(joint_velocity_limits, dtype=float)
    mapping_raw = np.asarray(dof_to_coord)
    if candidate.ndim != 1 or previous.shape != candidate.shape:
        raise ValueError(
            "candidate_joint_q and previous_joint_q must be equal-length vectors"
        )
    if not np.isfinite(previous).all():
        raise ValueError("previous_joint_q must contain only finite values")
    if limits.ndim != 1 or mapping_raw.shape != limits.shape:
        raise ValueError(
            "joint_velocity_limits and dof_to_coord must be equal-length vectors"
        )
    try:
        mapping = mapping_raw.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("dof_to_coord must contain integer indices") from exc
    if not np.array_equal(mapping_raw, mapping):
        raise ValueError("dof_to_coord must contain integer indices")
    if np.any(mapping < 0) or np.any(mapping >= len(candidate)):
        raise ValueError("dof_to_coord contains an out-of-range coordinate index")
    if np.isnan(limits).any() or np.any(limits < 0.0):
        raise ValueError("joint_velocity_limits must be non-negative and cannot contain NaN")
    if active_mask is None:
        mask = np.ones(limits.shape, dtype=float)
    else:
        mask = np.asarray(active_mask, dtype=float)
        if mask.shape != limits.shape or not np.isfinite(mask).all():
            raise ValueError(
                "active_mask must be finite and match joint_velocity_limits"
            )
        if np.any((mask != 0.0) & (mask != 1.0)):
            raise ValueError("active_mask entries must be exactly zero or one")
    timestep = float(dt_s)
    if not math.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    return candidate, previous, limits, mapping, mask, timestep


def joint_velocity_violations(
    candidate_joint_q: Sequence[float] | np.ndarray,
    previous_joint_q: Sequence[float] | np.ndarray,
    joint_velocity_limits: Sequence[float] | np.ndarray,
    dof_to_coord: Sequence[int] | np.ndarray,
    *,
    dt_s: float,
    active_mask: Sequence[float] | np.ndarray | None = None,
    tolerance_rad: float = 0.0,
) -> tuple[JointVelocityViolation, ...]:
    """Return proposed per-step velocity-limit violations before projection."""

    tolerance = _nonnegative_finite(tolerance_rad, "velocity-step tolerance")
    candidate, previous, limits, mapping, mask, timestep = _joint_velocity_inputs(
        candidate_joint_q,
        previous_joint_q,
        joint_velocity_limits,
        dof_to_coord,
        active_mask,
        dt_s,
    )
    if not np.isfinite(candidate).all():
        return ()

    violations: list[JointVelocityViolation] = []
    for dof_index, coord_index_value in enumerate(mapping):
        if mask[dof_index] == 0.0:
            continue
        coord_index = int(coord_index_value)
        delta = abs(
            float(candidate[coord_index]) - float(previous[coord_index])
        )
        velocity_limit = float(limits[dof_index])
        max_delta = velocity_limit * timestep
        if delta > max_delta + tolerance:
            violations.append(
                JointVelocityViolation(
                    dof_index=dof_index,
                    coord_index=coord_index,
                    previous=float(previous[coord_index]),
                    candidate=float(candidate[coord_index]),
                    delta=delta,
                    max_delta=max_delta,
                    requested_velocity=delta / timestep,
                    velocity_limit=velocity_limit,
                )
            )
    return tuple(violations)


def project_scalar_joint_velocity_limits(
    candidate_joint_q: Sequence[float] | np.ndarray,
    previous_joint_q: Sequence[float] | np.ndarray,
    joint_velocity_limits: Sequence[float] | np.ndarray,
    dof_to_coord: Sequence[int] | np.ndarray,
    *,
    dt_s: float,
    active_mask: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Project a proposed scalar-joint step onto URDF velocity bounds.

    The interval is centered on the previous finite state and has half-width
    ``velocity_limit * dt_s``.  Inactive DoFs are left untouched.  A projected
    result must undergo fresh FK and Cartesian validation at the call site.
    """

    candidate, previous, limits, mapping, mask, timestep = _joint_velocity_inputs(
        candidate_joint_q,
        previous_joint_q,
        joint_velocity_limits,
        dof_to_coord,
        active_mask,
        dt_s,
    )
    projected = candidate.copy()
    if not np.isfinite(projected).all():
        return projected
    for dof_index, coord_index_value in enumerate(mapping):
        if mask[dof_index] == 0.0:
            continue
        coord_index = int(coord_index_value)
        max_delta = float(limits[dof_index]) * timestep
        low = float(previous[coord_index]) - max_delta
        high = float(previous[coord_index]) + max_delta
        projected[coord_index] = float(np.clip(projected[coord_index], low, high))
    return projected


_MOTION_LIMIT_SAFETY_SCALE = 1.0 - 1.0e-9


def _float32_in_closed_interval(
    desired_value: float,
    lower: float,
    upper: float,
    *,
    dof_index: int,
    coord_index: int,
) -> float:
    """Return the nearest finite float32 value inside one closed interval.

    Casting a continuous projection directly to float32 can round outward past
    a hard bound.  Moving by one representable value toward the interval is
    sufficient after round-to-nearest; if that value crosses the opposite
    bound, the interval contains no finite float32 point and must fail closed.
    """

    if math.isnan(lower) or math.isnan(upper) or lower > upper:
        raise ValueError("float32 projection interval must be ordered and not NaN")
    clipped = min(max(float(desired_value), lower), upper)
    with np.errstate(over="ignore", invalid="ignore"):
        value32 = np.float32(clipped)
    value = float(value32)

    if not math.isfinite(value):
        value32 = np.nextafter(value32, np.float32(0.0))
        value = float(value32)
    if value < lower:
        value32 = np.nextafter(value32, np.float32(math.inf))
        value = float(value32)
    elif value > upper:
        value32 = np.nextafter(value32, np.float32(-math.inf))
        value = float(value32)

    if not math.isfinite(value) or value < lower or value > upper:
        raise JointMotionInfeasibleError(
            "motion interval contains no finite float32 joint coordinate",
            details={
                "constraint": "float32_representability",
                "feasibility_scope": "one_step_from_previous_state",
                "dof_index": int(dof_index),
                "coord_index": int(coord_index),
                "desired_q": float(desired_value),
                "float32_interval_lower_q": float(lower),
                "float32_interval_upper_q": float(upper),
            },
        )
    return value


def project_scalar_joint_motion_limits(
    candidate_joint_q: Sequence[float] | np.ndarray,
    previous_joint_q: Sequence[float] | np.ndarray,
    previous_joint_velocity: Sequence[float] | np.ndarray,
    previous_joint_acceleration: Sequence[float] | np.ndarray,
    joint_position_lower: Sequence[float] | np.ndarray,
    joint_position_upper: Sequence[float] | np.ndarray,
    joint_velocity_limits: Sequence[float] | np.ndarray,
    dof_to_coord: Sequence[int] | np.ndarray,
    *,
    dt_s: float,
    max_acceleration_rad_s2: float,
    max_jerk_rad_s3: float,
    active_mask: Sequence[float] | np.ndarray | None = None,
) -> JointMotionProjection:
    """Project one candidate onto the complete discrete joint-motion interval.

    For each active scalar DoF, the feasible next velocity is the intersection
    of the URDF velocity bound, the configured acceleration bound, the jerk
    interval around the previous acceleration, and the joint-position bounds.
    Empty intersections raise :class:`JointMotionInfeasibleError` rather than
    silently violating one constraint.  A tiny inward numerical margin keeps
    finite differences below strict metric thresholds after round-off.
    Feasibility is intentionally local to the supplied previous state; this
    greedy projector does not certify that an earlier state left enough future
    braking distance before a position bound.
    """

    candidate, previous, velocity_limits, mapping, mask, timestep = (
        _joint_velocity_inputs(
            candidate_joint_q,
            previous_joint_q,
            joint_velocity_limits,
            dof_to_coord,
            active_mask,
            dt_s,
        )
    )
    if not np.isfinite(candidate).all():
        raise ValueError("candidate_joint_q must contain only finite values")

    previous_velocity = np.asarray(previous_joint_velocity, dtype=float)
    previous_acceleration = np.asarray(previous_joint_acceleration, dtype=float)
    lower = np.asarray(joint_position_lower, dtype=float)
    upper = np.asarray(joint_position_upper, dtype=float)
    dof_shape = velocity_limits.shape
    if previous_velocity.shape != dof_shape:
        raise ValueError("previous_joint_velocity must match joint DoF layout")
    if previous_acceleration.shape != dof_shape:
        raise ValueError("previous_joint_acceleration must match joint DoF layout")
    if not np.isfinite(previous_velocity).all():
        raise ValueError("previous_joint_velocity must contain only finite values")
    if not np.isfinite(previous_acceleration).all():
        raise ValueError("previous_joint_acceleration must contain only finite values")
    if lower.shape != dof_shape or upper.shape != dof_shape:
        raise ValueError("joint position limits must match joint DoF layout")
    if np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower > upper):
        raise ValueError("joint position limits must be ordered and cannot contain NaN")

    acceleration_limit = float(max_acceleration_rad_s2)
    jerk_limit = float(max_jerk_rad_s3)
    if not math.isfinite(acceleration_limit) or acceleration_limit <= 0.0:
        raise ValueError("max_acceleration_rad_s2 must be finite and positive")
    if not math.isfinite(jerk_limit) or jerk_limit <= 0.0:
        raise ValueError("max_jerk_rad_s3 must be finite and positive")
    acceleration_limit *= _MOTION_LIMIT_SAFETY_SCALE
    jerk_limit *= _MOTION_LIMIT_SAFETY_SCALE

    with np.errstate(over="ignore", invalid="ignore"):
        projected = candidate.astype(np.float32).astype(float)
    if not np.isfinite(projected).all():
        bad_coord_indices = np.flatnonzero(~np.isfinite(projected)).tolist()
        raise JointMotionInfeasibleError(
            "candidate joint coordinates are not representable as finite float32",
            details={
                "constraint": "float32_representability",
                "feasibility_scope": "one_step_from_previous_state",
                "coord_indices": bad_coord_indices,
            },
        )
    next_velocity = np.empty(dof_shape, dtype=float)
    next_acceleration = np.empty(dof_shape, dtype=float)
    next_jerk = np.empty(dof_shape, dtype=float)
    projected_dofs: list[int] = []
    diagnostics: list[JointMotionProjectionDiagnostic] = []

    for dof_index, coord_index_value in enumerate(mapping):
        coord_index = int(coord_index_value)
        previous_q = float(previous[coord_index])
        desired_q = float(candidate[coord_index])
        desired_velocity = (desired_q - previous_q) / timestep
        desired_acceleration = (
            desired_velocity - float(previous_velocity[dof_index])
        ) / timestep
        desired_jerk = (
            desired_acceleration - float(previous_acceleration[dof_index])
        ) / timestep
        trigger_reasons: list[str] = []

        if mask[dof_index] == 0.0:
            selected_velocity = desired_velocity
            feasible_q_lower = -math.inf
            feasible_q_upper = math.inf
        else:
            raw_velocity_limit = float(velocity_limits[dof_index])
            if not math.isfinite(raw_velocity_limit) or raw_velocity_limit <= 0.0:
                raise ValueError(
                    "active joint velocity limits must be finite and positive"
                )
            velocity_limit = raw_velocity_limit * _MOTION_LIMIT_SAFETY_SCALE
            previous_v = float(previous_velocity[dof_index])
            previous_a = float(previous_acceleration[dof_index])

            acceleration_lower = max(
                -acceleration_limit,
                previous_a - jerk_limit * timestep,
            )
            acceleration_upper = min(
                acceleration_limit,
                previous_a + jerk_limit * timestep,
            )
            if acceleration_lower > acceleration_upper:
                raise JointMotionInfeasibleError(
                    "joint acceleration and jerk intervals do not overlap",
                    details={
                        "constraint": "acceleration_jerk",
                        "feasibility_scope": "one_step_from_previous_state",
                        "dof_index": dof_index,
                        "coord_index": coord_index,
                        "previous_acceleration_rad_s2": previous_a,
                        "acceleration_lower_rad_s2": acceleration_lower,
                        "acceleration_upper_rad_s2": acceleration_upper,
                    },
                )

            acceleration_velocity_lower = (
                previous_v + acceleration_lower * timestep
            )
            acceleration_velocity_upper = (
                previous_v + acceleration_upper * timestep
            )
            velocity_lower = max(
                -velocity_limit,
                acceleration_velocity_lower,
            )
            velocity_upper = min(
                velocity_limit,
                acceleration_velocity_upper,
            )
            if velocity_lower > velocity_upper:
                raise JointMotionInfeasibleError(
                    "joint velocity and acceleration/jerk intervals do not overlap",
                    details={
                        "constraint": "velocity_acceleration_jerk",
                        "feasibility_scope": "one_step_from_previous_state",
                        "dof_index": dof_index,
                        "coord_index": coord_index,
                        "previous_velocity_rad_s": previous_v,
                        "velocity_lower_rad_s": velocity_lower,
                        "velocity_upper_rad_s": velocity_upper,
                        "velocity_limit_rad_s": raw_velocity_limit,
                    },
                )

            position_velocity_lower = (
                float(lower[dof_index]) - previous_q
            ) / timestep
            position_velocity_upper = (
                float(upper[dof_index]) - previous_q
            ) / timestep
            feasible_velocity_lower = max(
                velocity_lower,
                position_velocity_lower,
            )
            feasible_velocity_upper = min(
                velocity_upper,
                position_velocity_upper,
            )
            if feasible_velocity_lower > feasible_velocity_upper:
                raise JointMotionInfeasibleError(
                    "joint position and motion intervals do not overlap",
                    details={
                        "constraint": "position_motion",
                        "feasibility_scope": "one_step_from_previous_state",
                        "dof_index": dof_index,
                        "coord_index": coord_index,
                        "previous_q": previous_q,
                        "position_lower": float(lower[dof_index]),
                        "position_upper": float(upper[dof_index]),
                        "feasible_velocity_lower_rad_s": feasible_velocity_lower,
                        "feasible_velocity_upper_rad_s": feasible_velocity_upper,
                    },
                )
            selected_velocity = float(
                np.clip(
                    desired_velocity,
                    feasible_velocity_lower,
                    feasible_velocity_upper,
                )
            )

            if desired_q < float(lower[dof_index]) or desired_q > float(
                upper[dof_index]
            ):
                trigger_reasons.append("position_limit")
            if abs(desired_velocity) > velocity_limit:
                trigger_reasons.append("velocity_limit")
            if abs(desired_acceleration) > acceleration_limit:
                trigger_reasons.append("acceleration_limit")
            if abs(desired_jerk) > jerk_limit:
                trigger_reasons.append("jerk_limit")
            feasible_q_lower = previous_q + feasible_velocity_lower * timestep
            feasible_q_upper = previous_q + feasible_velocity_upper * timestep

        continuous_q = previous_q + selected_velocity * timestep
        selected_q = _float32_in_closed_interval(
            continuous_q,
            feasible_q_lower,
            feasible_q_upper,
            dof_index=dof_index,
            coord_index=coord_index,
        )
        projected[coord_index] = selected_q
        realized_velocity = (selected_q - previous_q) / timestep
        realized_acceleration = (
            realized_velocity - float(previous_velocity[dof_index])
        ) / timestep
        realized_jerk = (
            realized_acceleration
            - float(previous_acceleration[dof_index])
        ) / timestep
        next_velocity[dof_index] = realized_velocity
        next_acceleration[dof_index] = realized_acceleration
        next_jerk[dof_index] = realized_jerk

        if mask[dof_index] != 0.0:
            post_projection_violations: list[dict[str, float | str]] = []
            if not float(lower[dof_index]) <= selected_q <= float(
                upper[dof_index]
            ):
                post_projection_violations.append(
                    {
                        "kind": "position",
                        "actual": selected_q,
                        "lower": float(lower[dof_index]),
                        "upper": float(upper[dof_index]),
                    }
                )
            if abs(realized_velocity) > raw_velocity_limit:
                post_projection_violations.append(
                    {
                        "kind": "velocity",
                        "actual": abs(realized_velocity),
                        "limit": raw_velocity_limit,
                    }
                )
            if abs(realized_acceleration) > float(max_acceleration_rad_s2):
                post_projection_violations.append(
                    {
                        "kind": "acceleration",
                        "actual": abs(realized_acceleration),
                        "limit": float(max_acceleration_rad_s2),
                    }
                )
            if abs(realized_jerk) > float(max_jerk_rad_s3):
                post_projection_violations.append(
                    {
                        "kind": "jerk",
                        "actual": abs(realized_jerk),
                        "limit": float(max_jerk_rad_s3),
                    }
                )
            if post_projection_violations:
                raise JointMotionInfeasibleError(
                    "float32-realized joint state violates a hard motion bound",
                    details={
                        "constraint": "float32_post_projection_validation",
                        "feasibility_scope": "one_step_from_previous_state",
                        "dof_index": dof_index,
                        "coord_index": coord_index,
                        "violations": post_projection_violations,
                    },
                )

        if selected_q != desired_q:
            if selected_q != continuous_q:
                trigger_reasons.append("float32_quantization")
            if not trigger_reasons:
                trigger_reasons.append("combined_motion_interval")
            projected_dofs.append(dof_index)
            diagnostics.append(
                JointMotionProjectionDiagnostic(
                    dof_index=dof_index,
                    coord_index=coord_index,
                    raw_candidate_q=desired_q,
                    projected_q=selected_q,
                    q_correction_rad=selected_q - desired_q,
                    raw_requested_velocity_rad_s=desired_velocity,
                    raw_requested_acceleration_rad_s2=desired_acceleration,
                    raw_requested_jerk_rad_s3=desired_jerk,
                    projected_velocity_rad_s=realized_velocity,
                    projected_acceleration_rad_s2=realized_acceleration,
                    projected_jerk_rad_s3=realized_jerk,
                    trigger_reasons=tuple(dict.fromkeys(trigger_reasons)),
                )
            )

    return JointMotionProjection(
        joint_q=projected,
        joint_velocity=next_velocity,
        joint_acceleration=next_acceleration,
        joint_jerk=next_jerk,
        projected_dof_indices=tuple(projected_dofs),
        diagnostics=tuple(diagnostics),
    )


def validate_waypoint_solution(
    joint_q: Sequence[float] | np.ndarray,
    actual_position: Iterable[float],
    actual_orientation_xyzw: Iterable[float],
    target_position: Iterable[float],
    target_orientation_xyzw: Iterable[float],
    joint_limit_lower: Sequence[float] | np.ndarray,
    joint_limit_upper: Sequence[float] | np.ndarray,
    dof_to_coord: Sequence[int] | np.ndarray,
    *,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    joint_limit_tolerance: float,
) -> WaypointValidation:
    """Validate one IK output from realized FK, limits, and finite checks."""

    position_tol = _nonnegative_finite(position_tolerance_m, "position tolerance")
    orientation_tol = _nonnegative_finite(
        orientation_tolerance_rad, "orientation tolerance"
    )
    limit_tol = _nonnegative_finite(joint_limit_tolerance, "joint-limit tolerance")

    q = np.asarray(joint_q, dtype=float)
    if q.ndim != 1:
        raise ValueError(f"joint_q must be one-dimensional, got {q.shape}")
    actual_pos = _vector(
        actual_position, 3, "actual_position", require_finite=False
    )
    actual_quat = _vector(
        actual_orientation_xyzw, 4, "actual_orientation_xyzw", require_finite=False
    )
    target_pos = _vector(target_position, 3, "target_position", require_finite=True)
    target_quat = _vector(
        target_orientation_xyzw, 4, "target_orientation_xyzw", require_finite=True
    )
    if float(np.linalg.norm(target_quat)) == 0.0:
        raise ValueError("target_orientation_xyzw must have non-zero norm")

    has_nonfinite = not (
        np.isfinite(q).all()
        and np.isfinite(actual_pos).all()
        and np.isfinite(actual_quat).all()
    )
    actual_quat_norm = (
        float(np.linalg.norm(actual_quat)) if np.isfinite(actual_quat).all() else 0.0
    )
    if has_nonfinite or actual_quat_norm == 0.0:
        has_nonfinite = True
        pose_error = PoseError(position_m=math.inf, orientation_rad=math.inf)
        violations: tuple[JointLimitViolation, ...] = ()
    else:
        pose_error = compute_pose_error(
            actual_pos, actual_quat, target_pos, target_quat
        )
        violations = joint_limit_violations(
            q,
            joint_limit_lower,
            joint_limit_upper,
            dof_to_coord,
            tolerance=limit_tol,
        )

    failed_checks: list[str] = []
    failure_codes: list[FailureCode] = []
    if has_nonfinite:
        failed_checks.append("non_finite")
        failure_codes.append(FailureCode.NUMERICAL_INSTABILITY)
    if pose_error.position_m > position_tol:
        failed_checks.append("position_error")
        if FailureCode.IK_UNREACHABLE not in failure_codes:
            failure_codes.append(FailureCode.IK_UNREACHABLE)
    if pose_error.orientation_rad > orientation_tol:
        failed_checks.append("orientation_error")
        if FailureCode.IK_UNREACHABLE not in failure_codes:
            failure_codes.append(FailureCode.IK_UNREACHABLE)
    if violations:
        failed_checks.append("joint_limit")
        failure_codes.append(FailureCode.JOINT_LIMIT)

    return WaypointValidation(
        success=not failed_checks,
        pose_error=pose_error,
        joint_limit_violations=violations,
        has_nonfinite=has_nonfinite,
        failed_checks=tuple(failed_checks),
        failure_codes=tuple(failure_codes),
    )


@dataclass(frozen=True, slots=True)
class NewtonIKParameters:
    """All robot, solver, and acceptance inputs required by the adapter."""

    robot_urdf: Path
    robot_world_position: tuple[float, float, float]
    robot_world_orientation_wxyz: tuple[float, float, float, float]
    end_effector_link: str
    end_effector_offset_position: tuple[float, float, float]
    end_effector_offset_orientation_wxyz: tuple[float, float, float, float]
    arm_joint_names: tuple[str, ...]
    nominal_arm_joint_positions: tuple[float, ...]
    device: str
    optimizer: str
    jacobian: str
    seed: int
    iterations: int
    step_size: float
    lambda_initial: float
    position_weight: float
    rotation_weight: float
    joint_limit_weight: float
    nominal_posture_weight: float
    position_tolerance_m: float
    orientation_tolerance_rad: float
    joint_limit_tolerance: float
    control_limit_margin_rad: float
    enable_self_collisions: bool
    max_joint_acceleration_rad_s2: float = 7.5
    max_joint_jerk_rad_s3: float = 450.0
    continuity_weight: float = 0.01
    waypoint_dt_s: float = 1.0 / 60.0

    def __post_init__(self) -> None:
        _vector(
            self.robot_world_position,
            3,
            "robot_world_position",
            require_finite=True,
        )
        normalize_quaternion(self.robot_world_orientation_wxyz)
        _vector(
            self.end_effector_offset_position,
            3,
            "end_effector_offset_position",
            require_finite=True,
        )
        normalize_quaternion(self.end_effector_offset_orientation_wxyz)
        if not self.end_effector_link.strip():
            raise ValueError("end_effector_link must be non-empty")
        if not self.arm_joint_names or any(not name.strip() for name in self.arm_joint_names):
            raise ValueError("arm_joint_names must contain non-empty names")
        if len(set(self.arm_joint_names)) != len(self.arm_joint_names):
            raise ValueError("arm_joint_names must be unique")
        if len(self.arm_joint_names) != len(self.nominal_arm_joint_positions):
            raise ValueError("nominal arm positions must match arm_joint_names")
        if not all(math.isfinite(float(value)) for value in self.nominal_arm_joint_positions):
            raise ValueError("nominal arm positions must be finite")
        device = self.device.lower()
        if device != "cpu" and device != "cuda" and not (
            device.startswith("cuda:") and device[5:].isdigit()
        ):
            raise ValueError("Newton IK device must be cpu, cuda, or cuda:<index>")
        if self.optimizer.lower() != "lm":
            raise ValueError("Newton IK adapter requires optimizer='lm'")
        if self.jacobian.lower() != "analytic":
            raise ValueError("Newton IK adapter requires jacobian='analytic'")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            not isinstance(self.iterations, int)
            or isinstance(self.iterations, bool)
            or self.iterations < 1
        ):
            raise ValueError("iterations must be a positive integer")
        for name in (
            "step_size",
            "lambda_initial",
            "position_weight",
            "rotation_weight",
            "joint_limit_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        _nonnegative_finite(self.nominal_posture_weight, "nominal_posture_weight")
        continuity_weight = float(self.continuity_weight)
        if not math.isfinite(continuity_weight) or continuity_weight <= 0.0:
            raise ValueError("continuity_weight must be finite and positive")
        waypoint_dt = float(self.waypoint_dt_s)
        if not math.isfinite(waypoint_dt) or waypoint_dt <= 0.0:
            raise ValueError("waypoint_dt_s must be finite and positive")
        acceleration_limit = float(self.max_joint_acceleration_rad_s2)
        if not math.isfinite(acceleration_limit) or acceleration_limit <= 0.0:
            raise ValueError(
                "max_joint_acceleration_rad_s2 must be finite and positive"
            )
        jerk_limit = float(self.max_joint_jerk_rad_s3)
        if not math.isfinite(jerk_limit) or jerk_limit <= 0.0:
            raise ValueError("max_joint_jerk_rad_s3 must be finite and positive")
        _nonnegative_finite(self.position_tolerance_m, "position_tolerance_m")
        _nonnegative_finite(self.orientation_tolerance_rad, "orientation_tolerance_rad")
        _nonnegative_finite(self.joint_limit_tolerance, "joint_limit_tolerance")
        _nonnegative_finite(
            self.control_limit_margin_rad, "control_limit_margin_rad"
        )
        if not isinstance(self.enable_self_collisions, bool):
            raise ValueError("enable_self_collisions must be a boolean")

    @classmethod
    def from_project_config(
        cls,
        config: Any,
        *,
        joint_limit_tolerance: float,
    ) -> "NewtonIKParameters":
        """Build explicit parameters from :class:`src.config.ProjectConfig`.

        Joint-limit acceptance tolerance is required at the call site so the
        adapter never hides a numerical threshold.  The checked-in runner
        supplies ``ik.joint_limit_tolerance_rad`` from project configuration.
        """

        robot_pose = config.get("assets.robot.world_pose")
        ee_offset = config.get("assets.robot.end_effector_offset")
        self_collision_value = config.get("collision.enabled")
        if not isinstance(self_collision_value, bool):
            raise ValueError("collision.enabled must be a boolean")
        return cls(
            robot_urdf=config.asset_path("robot"),
            robot_world_position=tuple(float(v) for v in robot_pose["position"]),
            robot_world_orientation_wxyz=tuple(
                float(v) for v in robot_pose["orientation_wxyz"]
            ),
            end_effector_link=str(config.get("assets.robot.end_effector_link")),
            end_effector_offset_position=tuple(
                float(v) for v in ee_offset["position"]
            ),
            end_effector_offset_orientation_wxyz=tuple(
                float(v) for v in ee_offset["orientation_wxyz"]
            ),
            arm_joint_names=tuple(config.get("assets.robot.arm_joint_names")),
            nominal_arm_joint_positions=tuple(
                float(v) for v in config.get("assets.robot.default_joint_positions")
            ),
            device=str(config.get("runtime.device")),
            optimizer=str(config.get("ik.optimizer")),
            jacobian=str(config.get("ik.jacobian")),
            seed=int(config.get("seed")),
            iterations=config.get("ik.iterations"),
            step_size=float(config.get("ik.step_size")),
            lambda_initial=float(config.get("ik.lambda_initial")),
            position_weight=float(config.get("ik.position_weight")),
            rotation_weight=float(config.get("ik.rotation_weight")),
            joint_limit_weight=float(config.get("ik.joint_limit_weight")),
            nominal_posture_weight=float(config.get("ik.nominal_posture_weight")),
            continuity_weight=float(config.get("ik.continuity_weight")),
            position_tolerance_m=float(config.get("thresholds.position_error_m")),
            orientation_tolerance_rad=math.radians(
                float(config.get("thresholds.orientation_error_deg"))
            ),
            joint_limit_tolerance=float(joint_limit_tolerance),
            control_limit_margin_rad=float(
                config.get("ik.control_limit_margin_rad")
            ),
            enable_self_collisions=self_collision_value,
            max_joint_acceleration_rad_s2=float(
                config.get("thresholds.max_joint_acceleration_rad_s2")
            ),
            max_joint_jerk_rad_s3=float(
                config.get("thresholds.max_joint_jerk_rad_s3")
            ),
            waypoint_dt_s=float(config.get("simulation.dt")),
        )


def newton_backend_available() -> bool:
    """Return whether both Newton and Warp imported successfully."""

    return _NEWTON_IMPORT_ERROR is None


def require_newton_backend() -> None:
    """Raise a structured dependency failure if Newton cannot be used."""

    if _NEWTON_IMPORT_ERROR is not None:
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "Newton IK backend requires newton==1.3.0 and warp-lang==1.14.0",
            stage="newton_import",
            details={
                "exception_type": type(_NEWTON_IMPORT_ERROR).__name__,
                "exception": str(_NEWTON_IMPORT_ERROR),
            },
        ) from _NEWTON_IMPORT_ERROR


class NewtonFrankaIKBackend:
    """Franka-only Newton LM/analytic IK with true post-FK validation."""

    def __init__(self, parameters: NewtonIKParameters) -> None:
        require_newton_backend()
        self.parameters = parameters
        robot_urdf = Path(parameters.robot_urdf).expanduser().resolve()
        if not robot_urdf.is_file():
            raise PipelineError(
                FailureCode.ASSET_MISSING,
                f"Franka URDF does not exist: {robot_urdf}",
                stage="newton_model",
                details={"path": str(robot_urdf)},
            )
        try:
            robot_description = load_urdf(robot_urdf)
        except URDFModelError as exc:
            raise PipelineError(
                FailureCode.ASSET_INVALID,
                "Franka URDF could not be parsed for joint velocity limits",
                stage="newton_model",
                details=exc.to_dict(),
            ) from exc

        # This flag changes model/control array layout and must precede building.
        newton.use_coord_layout_targets = True
        device = parameters.device.lower()
        self.device = device
        base_quat_wxyz = normalize_quaternion(
            parameters.robot_world_orientation_wxyz
        )
        base_quat_xyzw = quaternion_wxyz_to_xyzw(base_quat_wxyz)
        base_position = _vector(
            parameters.robot_world_position,
            3,
            "robot_world_position",
            require_finite=True,
        )

        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder()
            builder.add_urdf(
                str(robot_urdf),
                xform=wp.transform(
                    wp.vec3(*base_position.tolist()), wp.quat(*base_quat_xyzw.tolist())
                ),
                floating=False,
                hide_visuals=True,
                parse_visuals_as_colliders=False,
                enable_self_collisions=parameters.enable_self_collisions,
                collapse_fixed_joints=False,
                collapse_massless_fixed_root=False,
                override_root_xform=True,
            )
            # No microwave/object is added here: IK operates on one Franka model.
            self.model = builder.finalize(device=device, requires_grad=False)

        if not self.model.use_coord_layout_targets:
            raise PipelineError(
                FailureCode.CONFIG_INVALID,
                "Newton model did not retain coordinate-layout joint targets",
                stage="newton_model",
            )
        if self.model.articulation_count != 1:
            raise PipelineError(
                FailureCode.ASSET_INVALID,
                "Franka-only IK URDF must produce exactly one articulation",
                stage="newton_model",
                details={"articulation_count": self.model.articulation_count},
            )

        self.end_effector = resolve_unique_label(
            self.model.body_label,
            parameters.end_effector_link,
            kind="body",
        )
        q_start = self.model.joint_q_start.numpy().astype(np.int64)
        qd_start = self.model.joint_qd_start.numpy().astype(np.int64)
        dof_dim = self.model.joint_dof_dim.numpy().astype(np.int64)
        try:
            self.dof_to_coord = build_scalar_dof_to_coord_map(
                q_start, qd_start, dof_dim
            )
        except ValueError as exc:
            raise PipelineError(
                FailureCode.CONFIG_INVALID,
                "Franka IK model contains a non-scalar movable joint",
                stage="newton_model",
                details={"error": str(exc)},
            ) from exc

        nominal_q = self.model.joint_q.numpy().astype(np.float32, copy=True)
        active_mask = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        arm_coord_indices: list[int] = []
        arm_dof_indices: list[int] = []
        arm_velocity_limits: list[float] = []
        resolved_arm_joint_indices: set[int] = set()
        for joint_name, nominal_value in zip(
            parameters.arm_joint_names,
            parameters.nominal_arm_joint_positions,
            strict=True,
        ):
            resolved = resolve_unique_label(
                self.model.joint_label, joint_name, kind="joint"
            )
            if resolved.index in resolved_arm_joint_indices:
                raise PipelineError(
                    FailureCode.NAME_NOT_UNIQUE,
                    "configured arm joint names resolve to the same Newton joint",
                    stage="newton_model",
                    details={"joint": joint_name, "label": resolved.label},
                )
            resolved_arm_joint_indices.add(resolved.index)
            coord_begin = int(q_start[resolved.index])
            coord_end = int(q_start[resolved.index + 1])
            dof_begin = int(qd_start[resolved.index])
            dof_end = int(qd_start[resolved.index + 1])
            if coord_end - coord_begin != 1 or dof_end - dof_begin != 1:
                raise PipelineError(
                    FailureCode.CONFIG_INVALID,
                    f"configured arm joint is not scalar: {joint_name}",
                    stage="newton_model",
                    details={
                        "joint": joint_name,
                        "label": resolved.label,
                        "coordinate_count": coord_end - coord_begin,
                        "dof_count": dof_end - dof_begin,
                    },
                )
            nominal_q[coord_begin] = float(nominal_value)
            active_mask[dof_begin] = 1.0
            arm_coord_indices.append(coord_begin)
            arm_dof_indices.append(dof_begin)
            urdf_joint = robot_description.joints.get(joint_name)
            velocity_limit = (
                None
                if urdf_joint is None or urdf_joint.limit is None
                else urdf_joint.limit.velocity
            )
            if (
                velocity_limit is None
                or not math.isfinite(float(velocity_limit))
                or float(velocity_limit) <= 0.0
            ):
                raise PipelineError(
                    FailureCode.ASSET_INVALID,
                    "configured arm joint has no finite positive URDF velocity limit",
                    stage="newton_model",
                    details={"joint": joint_name, "velocity": velocity_limit},
                )
            arm_velocity_limits.append(float(velocity_limit))

        if not np.isfinite(nominal_q).all():
            raise PipelineError(
                FailureCode.NUMERICAL_INSTABILITY,
                "Newton model produced non-finite initial joint coordinates",
                stage="newton_model",
            )
        self.nominal_joint_q = nominal_q
        self.arm_coord_indices = tuple(arm_coord_indices)
        self.arm_dof_indices = tuple(arm_dof_indices)
        self._velocity_active_mask = active_mask.astype(float, copy=True)
        self._joint_velocity_limits = np.full(
            self.model.joint_dof_count, math.inf, dtype=float
        )
        for dof_index, velocity_limit in zip(
            self.arm_dof_indices, arm_velocity_limits, strict=True
        ):
            self._joint_velocity_limits[dof_index] = velocity_limit
        self._joint_limit_lower = self.model.joint_limit_lower.numpy().astype(float)
        self._joint_limit_upper = self.model.joint_limit_upper.numpy().astype(float)
        self._control_limit_lower = self._joint_limit_lower.copy()
        self._control_limit_upper = self._joint_limit_upper.copy()
        for dof_index in self.arm_dof_indices:
            self._control_limit_lower[dof_index] += (
                self.parameters.control_limit_margin_rad
            )
            self._control_limit_upper[dof_index] -= (
                self.parameters.control_limit_margin_rad
            )
            if not (
                math.isfinite(self._control_limit_lower[dof_index])
                and math.isfinite(self._control_limit_upper[dof_index])
                and self._control_limit_lower[dof_index]
                < self._control_limit_upper[dof_index]
            ):
                raise PipelineError(
                    FailureCode.CONFIG_INVALID,
                    "configured IK control-limit margin collapses a joint range",
                    stage="newton_model",
                    details={
                        "dof_index": int(dof_index),
                        "margin_rad": self.parameters.control_limit_margin_rad,
                    },
                )
        self._control_limit_lower_device = wp.array(
            self._control_limit_lower,
            dtype=wp.float32,
            device=device,
        )
        self._control_limit_upper_device = wp.array(
            self._control_limit_upper,
            dtype=wp.float32,
            device=device,
        )
        self._tcp_offset_position = _vector(
            parameters.end_effector_offset_position,
            3,
            "end_effector_offset_position",
            require_finite=True,
        )
        self._tcp_offset_orientation_wxyz = normalize_quaternion(
            parameters.end_effector_offset_orientation_wxyz
        )

        initial_target_positions = wp.array(
            np.zeros((1, 3), dtype=np.float32), dtype=wp.vec3, device=device
        )
        initial_target_rotations = wp.array(
            np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            dtype=wp.vec4,
            device=device,
        )
        tcp_offset_xyzw = quaternion_wxyz_to_xyzw(
            self._tcp_offset_orientation_wxyz
        )
        self.position_objective = newton.ik.IKObjectivePosition(
            self.end_effector.index,
            wp.vec3(*self._tcp_offset_position.tolist()),
            initial_target_positions,
            weight=parameters.position_weight,
        )
        self.rotation_objective = newton.ik.IKObjectiveRotation(
            self.end_effector.index,
            wp.quat(*tcp_offset_xyzw.tolist()),
            initial_target_rotations,
            canonicalize_quat_err=True,
            weight=parameters.rotation_weight,
        )
        limit_objective = newton.ik.IKObjectiveJointLimit(
            self._control_limit_lower_device,
            self._control_limit_upper_device,
            weight=parameters.joint_limit_weight,
        )
        nominal_objective = JointNominalObjective(
            nominal_q,
            self.dof_to_coord,
            active_mask,
            cost_weight=parameters.nominal_posture_weight,
        )
        self.continuity_objective = JointReferenceObjective(
            nominal_q,
            self.dof_to_coord,
            active_mask,
            cost_weight=parameters.continuity_weight,
        )
        self.objectives = (
            self.position_objective,
            self.rotation_objective,
            limit_objective,
            nominal_objective,
            self.continuity_objective,
        )
        self.solver = newton.ik.IKSolver(
            self.model,
            n_problems=1,
            objectives=self.objectives,
            optimizer=newton.ik.IKOptimizer.LM,
            jacobian_mode=newton.ik.IKJacobianType.ANALYTIC,
            sampler=newton.ik.IKSampler.NONE,
            n_seeds=1,
            rng_seed=parameters.seed,
            lambda_initial=parameters.lambda_initial,
        )
        self._fk_state = self.model.state()
        self._zero_joint_qd = wp.zeros(
            self.model.joint_dof_count, dtype=wp.float32, device=device
        )

    def _forward_kinematics_tcp(
        self, joint_q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        q_device = wp.array(
            np.asarray(joint_q, dtype=np.float32),
            dtype=wp.float32,
            device=self.device,
        )
        newton.eval_fk(
            self.model, q_device, self._zero_joint_qd, self._fk_state
        )
        body_transform = np.asarray(
            self._fk_state.body_q.numpy()[self.end_effector.index], dtype=float
        )
        if body_transform.size != 7:
            raise PipelineError(
                FailureCode.PHYSICS_UNAVAILABLE,
                "Newton body transform does not use the expected seven-value layout",
                stage="newton_fk",
                details={"shape": list(body_transform.shape)},
            )
        body_transform = body_transform.reshape(7)
        body_position = body_transform[:3]
        # Preserve numerical failures for the post-FK validator instead of
        # letting strict transform helpers raise before a structured result is
        # produced.
        if not np.isfinite(body_transform).all():
            return (
                np.full(3, math.nan, dtype=float),
                np.full(4, math.nan, dtype=float),
            )
        body_orientation_wxyz = np.asarray(
            [
                body_transform[6],
                body_transform[3],
                body_transform[4],
                body_transform[5],
            ],
            dtype=float,
        )
        if float(np.linalg.norm(body_orientation_wxyz)) == 0.0:
            return body_position.copy(), body_orientation_wxyz
        world_tcp = compose_transforms(
            pose_matrix(body_position, body_orientation_wxyz),
            pose_matrix(
                self._tcp_offset_position,
                self._tcp_offset_orientation_wxyz,
            ),
        )
        return decompose_pose(world_tcp)

    def forward_kinematics_tcp(
        self, joint_q: Sequence[float] | np.ndarray
    ) -> np.ndarray:
        """Return ``T_world_tcp`` for a complete Newton coordinate vector.

        This public wrapper keeps callers away from simulator body indices and
        preserves the project's ``wxyz`` convention at the adapter boundary.
        """

        coordinates = np.asarray(joint_q, dtype=float)
        if coordinates.shape != (self.model.joint_coord_count,):
            raise ValueError(
                "joint_q must match model.joint_coord_count: "
                f"expected {(self.model.joint_coord_count,)}, got {coordinates.shape}"
            )
        position, orientation_wxyz = self._forward_kinematics_tcp(coordinates)
        if not (np.isfinite(position).all() and np.isfinite(orientation_wxyz).all()):
            return np.full((4, 4), math.nan, dtype=float)
        return pose_matrix(position, orientation_wxyz)

    def initial_tcp_transform(self) -> np.ndarray:
        """Return the nominal TCP transform used to begin the first segment."""

        return self.forward_kinematics_tcp(self.nominal_joint_q)

    def solve_waypoints(
        self,
        target_positions: Sequence[Sequence[float]],
        target_orientations_wxyz: Sequence[Sequence[float]],
        *,
        initial_joint_q: Sequence[float] | np.ndarray | None = None,
    ) -> IKTrajectoryResult:
        """Solve ordered Cartesian targets with the previous result as the seed.

        Newton's scalar objective cost is recorded only as a diagnostic.  Each
        solve is regularized against the previous finite joint state.  Ordered
        trajectories additionally project every arm step onto the intersection
        of position, URDF velocity, configured acceleration, and configured
        jerk bounds before fresh FK validation.
        """

        if len(target_positions) != len(target_orientations_wxyz):
            raise ValueError("target position and orientation counts must match")
        if initial_joint_q is None:
            initial = self.nominal_joint_q.copy()
        else:
            initial = np.asarray(initial_joint_q, dtype=np.float32)
            if initial.shape != (self.model.joint_coord_count,):
                raise ValueError(
                    "initial_joint_q must match model.joint_coord_count: "
                    f"expected {(self.model.joint_coord_count,)}, got {initial.shape}"
                )
            if not np.isfinite(initial).all():
                raise ValueError("initial_joint_q must contain only finite values")

        working = wp.array(
            initial.reshape(1, -1), dtype=wp.float32, device=self.device
        )
        enforce_motion_limits = (
            len(target_positions) > 1 or initial_joint_q is not None
        )
        # Independent candidate reachability has no previous trajectory frame,
        # so temporal continuity must not pin it to the nominal posture.
        # Ordered solves restore the configured arm mask before stepping.
        self.continuity_objective.set_active_mask(
            self._velocity_active_mask
            if enforce_motion_limits
            else np.zeros_like(self._velocity_active_mask)
        )
        previous_joint_velocity = np.zeros(
            self.model.joint_dof_count, dtype=float
        )
        previous_joint_acceleration = np.zeros(
            self.model.joint_dof_count, dtype=float
        )
        previous_motion_q = initial.astype(float, copy=True)
        previous_target_position: np.ndarray | None = None
        previous_target_wxyz: np.ndarray | None = None
        previous_target_had_finite_candidate = False
        previous_target_objective_cost: float | None = None
        results: list[IKWaypointResult] = []
        for waypoint_index, (position_values, orientation_values) in enumerate(
            zip(target_positions, target_orientations_wxyz, strict=True)
        ):
            target_position = _vector(
                position_values,
                3,
                f"target_positions[{waypoint_index}]",
                require_finite=True,
            )
            target_wxyz = normalize_quaternion(orientation_values)
            target_xyzw = quaternion_wxyz_to_xyzw(target_wxyz)
            repeated_target = bool(
                previous_target_had_finite_candidate
                and previous_target_position is not None
                and previous_target_wxyz is not None
                and _cartesian_targets_equivalent(
                    previous_target_position,
                    previous_target_wxyz,
                    target_position,
                    target_wxyz,
                )
            )
            self.position_objective.set_target_position(
                0, wp.vec3(*target_position.tolist())
            )
            self.rotation_objective.set_target_rotation(
                0, wp.vec4(*target_xyzw.tolist())
            )

            previous_finite = previous_motion_q.copy()
            self.continuity_objective.set_reference_q(previous_finite)
            if repeated_target:
                # A stationary Cartesian target leaves a 7-DoF null space in
                # which the nominal-posture term can move on every LM call.
                # Request the previous projected q instead.  The motion
                # projector below may still advance it briefly to dissipate
                # residual velocity/acceleration within the hard jerk bound;
                # once at rest it reproduces the float32 q exactly.
                candidate = previous_finite.copy()
                if previous_target_objective_cost is None or not math.isfinite(
                    previous_target_objective_cost
                ):
                    raise PipelineError(
                        FailureCode.NUMERICAL_INSTABILITY,
                        "stationary IK hold has no finite prior LM objective cost",
                        stage="ik_solve",
                        details={"waypoint_index": waypoint_index},
                    )
                # No optimizer call is made for a repeated target, so this is
                # explicitly the last LM diagnostic for the same target rather
                # than a newly evaluated cost at the held/projected q.
                objective_cost = previous_target_objective_cost
            else:
                self.solver.step(
                    working,
                    working,
                    iterations=self.parameters.iterations,
                    step_size=self.parameters.step_size,
                )
                candidate = working.numpy()[0].astype(float, copy=True)
                costs = self.solver.costs.numpy()
                objective_cost = float(costs[0]) if len(costs) else math.nan
            raw_velocity_violations: tuple[JointVelocityViolation, ...] = ()
            projected_dof_indices: tuple[int, ...] = ()
            projection_diagnostics: tuple[
                JointMotionProjectionDiagnostic, ...
            ] = ()
            candidate_joint_velocity = previous_joint_velocity
            candidate_joint_acceleration = previous_joint_acceleration

            if np.isfinite(candidate).all():
                if enforce_motion_limits:
                    raw_velocity_violations = joint_velocity_violations(
                        candidate,
                        previous_finite,
                        self._joint_velocity_limits,
                        self.dof_to_coord,
                        dt_s=self.parameters.waypoint_dt_s,
                        active_mask=self._velocity_active_mask,
                    )
                    try:
                        motion_projection = project_scalar_joint_motion_limits(
                            candidate,
                            previous_finite,
                            previous_joint_velocity,
                            previous_joint_acceleration,
                            self._control_limit_lower,
                            self._control_limit_upper,
                            self._joint_velocity_limits,
                            self.dof_to_coord,
                            dt_s=self.parameters.waypoint_dt_s,
                            max_acceleration_rad_s2=(
                                self.parameters.max_joint_acceleration_rad_s2
                            ),
                            max_jerk_rad_s3=self.parameters.max_joint_jerk_rad_s3,
                            active_mask=self._velocity_active_mask,
                        )
                    except JointMotionInfeasibleError as exc:
                        raise PipelineError(
                            FailureCode.IK_UNREACHABLE,
                            "current projected motion state has no feasible next joint state",
                            stage="ik_motion_limits",
                            details={
                                "waypoint_index": waypoint_index,
                                "feasibility_scope": (
                                    "one_step_from_current_projected_state"
                                ),
                                **exc.details,
                            },
                        ) from exc
                    candidate = motion_projection.joint_q
                    candidate_joint_velocity = motion_projection.joint_velocity
                    candidate_joint_acceleration = (
                        motion_projection.joint_acceleration
                    )
                    projected_dof_indices = (
                        motion_projection.projected_dof_indices
                    )
                    projection_diagnostics = motion_projection.diagnostics
                else:
                    candidate = project_scalar_joint_limits(
                        candidate,
                        self._control_limit_lower,
                        self._control_limit_upper,
                        self.dof_to_coord,
                    )
                    continuous_candidate = candidate.copy()
                    with np.errstate(over="ignore", invalid="ignore"):
                        candidate = candidate.astype(np.float32).astype(float)
                    if not np.isfinite(candidate).all():
                        raise JointMotionInfeasibleError(
                            "projected joint coordinates are not representable as finite float32",
                            details={
                                "constraint": "float32_representability",
                                "feasibility_scope": "single_waypoint",
                            },
                        )
                    for dof_index, coord_index_value in enumerate(
                        self.dof_to_coord
                    ):
                        coord_index = int(coord_index_value)
                        candidate[coord_index] = _float32_in_closed_interval(
                            float(continuous_candidate[coord_index]),
                            float(self._control_limit_lower[dof_index]),
                            float(self._control_limit_upper[dof_index]),
                            dof_index=dof_index,
                            coord_index=coord_index,
                        )
                # The exact position/velocity-feasible projection, not an
                # optimizer overshoot or alternate IK branch, becomes the warm
                # start for the next waypoint.
                working = wp.array(
                    candidate.astype(np.float32, copy=False).reshape(1, -1),
                    dtype=wp.float32,
                    device=self.device,
                )
                actual_position, actual_wxyz = self._forward_kinematics_tcp(candidate)
                if np.isfinite(actual_wxyz).all():
                    actual_xyzw = quaternion_wxyz_to_xyzw(actual_wxyz)
                else:
                    actual_xyzw = np.full(4, math.nan, dtype=float)
            else:
                actual_position = np.full(3, math.nan, dtype=float)
                actual_wxyz = np.full(4, math.nan, dtype=float)
                actual_xyzw = np.full(4, math.nan, dtype=float)

            validation = validate_waypoint_solution(
                candidate,
                actual_position,
                actual_xyzw,
                target_position,
                target_xyzw,
                self._joint_limit_lower,
                self._joint_limit_upper,
                self.dof_to_coord,
                position_tolerance_m=self.parameters.position_tolerance_m,
                orientation_tolerance_rad=self.parameters.orientation_tolerance_rad,
                joint_limit_tolerance=self.parameters.joint_limit_tolerance,
            )
            results.append(
                IKWaypointResult(
                    waypoint_index=waypoint_index,
                    joint_positions=tuple(float(value) for value in candidate),
                    arm_joint_positions=tuple(
                        float(candidate[index]) for index in self.arm_coord_indices
                    ),
                    target_position=tuple(float(value) for value in target_position),
                    target_orientation_wxyz=tuple(
                        float(value) for value in target_wxyz
                    ),
                    actual_position=tuple(float(value) for value in actual_position),
                    actual_orientation_wxyz=tuple(
                        float(value) for value in actual_wxyz
                    ),
                    objective_cost=objective_cost,
                    validation=validation,
                    velocity_projection_applied=bool(raw_velocity_violations),
                    raw_joint_velocity_violations=raw_velocity_violations,
                    motion_projection_applied=bool(projected_dof_indices),
                    projected_joint_dof_indices=projected_dof_indices,
                    motion_projection_diagnostics=projection_diagnostics,
                )
            )

            # A non-finite joint candidate cannot seed the next waypoint.
            # A finite candidate whose FK failed still remains the commanded
            # trajectory state, preserving discrete motion continuity.
            if not np.isfinite(candidate).all():
                working = wp.array(
                    previous_finite.reshape(1, -1),
                    dtype=wp.float32,
                    device=self.device,
                )
            elif enforce_motion_limits:
                previous_motion_q = candidate.astype(float, copy=True)
                previous_joint_velocity = candidate_joint_velocity
                previous_joint_acceleration = candidate_joint_acceleration

            previous_target_position = target_position.copy()
            previous_target_wxyz = target_wxyz.copy()
            previous_target_had_finite_candidate = bool(
                np.isfinite(candidate).all()
            )
            previous_target_objective_cost = objective_cost

        return IKTrajectoryResult(waypoints=tuple(results))


# Short alias for callers that are not robot-name specific.
NewtonIKBackend = NewtonFrankaIKBackend


__all__ = [
    "IKTrajectoryResult",
    "IKWaypointResult",
    "JointLimitViolation",
    "JointMotionInfeasibleError",
    "JointMotionProjection",
    "JointMotionProjectionDiagnostic",
    "JointVelocityViolation",
    "NewtonFrankaIKBackend",
    "NewtonIKBackend",
    "NewtonIKParameters",
    "PoseError",
    "ResolvedLabel",
    "WaypointValidation",
    "compute_pose_error",
    "joint_limit_violations",
    "joint_velocity_violations",
    "newton_backend_available",
    "quaternion_angle_rad_xyzw",
    "quaternion_wxyz_to_xyzw",
    "quaternion_xyzw_to_wxyz",
    "project_scalar_joint_limits",
    "project_scalar_joint_motion_limits",
    "project_scalar_joint_velocity_limits",
    "require_newton_backend",
    "resolve_unique_label",
    "validate_waypoint_solution",
]
