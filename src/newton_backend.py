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
from .ik_objectives import JointNominalObjective, build_scalar_dof_to_coord_map
from .transforms import (
    compose_transforms,
    decompose_pose,
    normalize_quaternion,
    pose_matrix,
)


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
    enable_self_collisions: bool

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
        if self.device.lower() != "cpu":
            raise ValueError("deterministic first-stage Newton IK requires explicit CPU")
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
        _nonnegative_finite(self.position_tolerance_m, "position_tolerance_m")
        _nonnegative_finite(self.orientation_tolerance_rad, "orientation_tolerance_rad")
        _nonnegative_finite(self.joint_limit_tolerance, "joint_limit_tolerance")
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
            position_tolerance_m=float(config.get("thresholds.position_error_m")),
            orientation_tolerance_rad=math.radians(
                float(config.get("thresholds.orientation_error_deg"))
            ),
            joint_limit_tolerance=float(joint_limit_tolerance),
            enable_self_collisions=self_collision_value,
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

        if not np.isfinite(nominal_q).all():
            raise PipelineError(
                FailureCode.NUMERICAL_INSTABILITY,
                "Newton model produced non-finite initial joint coordinates",
                stage="newton_model",
            )
        self.nominal_joint_q = nominal_q
        self.arm_coord_indices = tuple(arm_coord_indices)
        self.arm_dof_indices = tuple(arm_dof_indices)
        self._joint_limit_lower = self.model.joint_limit_lower.numpy().astype(float)
        self._joint_limit_upper = self.model.joint_limit_upper.numpy().astype(float)
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
            self.model.joint_limit_lower,
            self.model.joint_limit_upper,
            weight=parameters.joint_limit_weight,
        )
        nominal_objective = JointNominalObjective(
            nominal_q,
            self.dof_to_coord,
            active_mask,
            cost_weight=parameters.nominal_posture_weight,
        )
        self.objectives = (
            self.position_objective,
            self.rotation_objective,
            limit_objective,
            nominal_objective,
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

    def solve_waypoints(
        self,
        target_positions: Sequence[Sequence[float]],
        target_orientations_wxyz: Sequence[Sequence[float]],
        *,
        initial_joint_q: Sequence[float] | np.ndarray | None = None,
    ) -> IKTrajectoryResult:
        """Solve ordered Cartesian targets with the previous result as the seed.

        Newton's scalar objective cost is recorded only as a diagnostic.  A
        waypoint succeeds exclusively when a fresh FK call passes finite,
        Cartesian-error, and hard joint-limit checks.
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
            self.position_objective.set_target_position(
                0, wp.vec3(*target_position.tolist())
            )
            self.rotation_objective.set_target_rotation(
                0, wp.vec4(*target_xyzw.tolist())
            )

            previous_finite = working.numpy()[0].astype(np.float32, copy=True)
            self.solver.step(
                working,
                working,
                iterations=self.parameters.iterations,
                step_size=self.parameters.step_size,
            )
            candidate = working.numpy()[0].astype(float, copy=True)
            costs = self.solver.costs.numpy()
            objective_cost = float(costs[0]) if len(costs) else math.nan

            if np.isfinite(candidate).all():
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
                )
            )

            # A NaN candidate cannot seed the next waypoint.  Preserve the last
            # finite warm start while retaining the failed candidate in output.
            if validation.has_nonfinite:
                working = wp.array(
                    previous_finite.reshape(1, -1),
                    dtype=wp.float32,
                    device=self.device,
                )

        return IKTrajectoryResult(waypoints=tuple(results))


# Short alias for callers that are not robot-name specific.
NewtonIKBackend = NewtonFrankaIKBackend


__all__ = [
    "IKTrajectoryResult",
    "IKWaypointResult",
    "JointLimitViolation",
    "NewtonFrankaIKBackend",
    "NewtonIKBackend",
    "NewtonIKParameters",
    "PoseError",
    "ResolvedLabel",
    "WaypointValidation",
    "compute_pose_error",
    "joint_limit_violations",
    "newton_backend_available",
    "quaternion_angle_rad_xyzw",
    "quaternion_wxyz_to_xyzw",
    "quaternion_xyzw_to_wxyz",
    "require_newton_backend",
    "resolve_unique_label",
    "validate_waypoint_solution",
]
