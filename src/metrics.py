"""JSON-safe metrics and acceptance gates for deterministic rollouts.

All pose arrays contain 4x4 local-to-world transforms.  Upstream pose APIs and
configuration use ``wxyz`` quaternions; a backend that uses another ordering
must convert explicitly before producing these transforms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


class MetricsInputError(ValueError):
    """Raised when metric inputs or thresholds have incompatible structure."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "context": dict(self.context)}


@dataclass(frozen=True, slots=True)
class MetricThresholds:
    """Every acceptance threshold required by :func:`compute_metrics`.

    There are deliberately no defaults: every value must come from the task
    configuration so changing policy never requires changing metric code.
    """

    min_ik_success_rate: float
    position_error_m: float
    orientation_error_deg: float
    final_door_angle_deg: float
    grasp_position_drift_m: float
    grasp_orientation_drift_deg: float
    max_joint_limit_violations: int
    max_joint_limit_violation_frame_ratio: float
    max_collision_frame_ratio: float
    require_nan_free: bool

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "MetricThresholds":
        required = tuple(cls.__dataclass_fields__)
        missing = [name for name in required if name not in values]
        if missing:
            raise MetricsInputError(
                "THRESHOLD_MISSING",
                "metrics thresholds are missing required fields",
                fields=missing,
            )

        def finite_number(name: str) -> float:
            raw = values[name]
            if isinstance(raw, bool):
                raise MetricsInputError(
                    "THRESHOLD_INVALID", f"threshold {name} must be numeric", field=name, value=raw
                )
            try:
                result = float(raw)
            except (TypeError, ValueError) as exc:
                raise MetricsInputError(
                    "THRESHOLD_INVALID", f"threshold {name} must be numeric", field=name, value=raw
                ) from exc
            if not math.isfinite(result):
                raise MetricsInputError(
                    "THRESHOLD_INVALID", f"threshold {name} must be finite", field=name, value=raw
                )
            return result

        nonnegative_names = (
            "position_error_m",
            "orientation_error_deg",
            "final_door_angle_deg",
            "grasp_position_drift_m",
            "grasp_orientation_drift_deg",
            "max_joint_limit_violation_frame_ratio",
            "max_collision_frame_ratio",
        )
        parsed: dict[str, Any] = {name: finite_number(name) for name in nonnegative_names}
        for name in nonnegative_names:
            if parsed[name] < 0.0:
                raise MetricsInputError(
                    "THRESHOLD_INVALID",
                    f"threshold {name} must be non-negative",
                    field=name,
                    value=parsed[name],
                )

        parsed["min_ik_success_rate"] = finite_number("min_ik_success_rate")
        if not 0.0 <= parsed["min_ik_success_rate"] <= 1.0:
            raise MetricsInputError(
                "THRESHOLD_INVALID",
                "min_ik_success_rate must be in [0, 1]",
                field="min_ik_success_rate",
                value=parsed["min_ik_success_rate"],
            )
        for name in ("max_joint_limit_violation_frame_ratio", "max_collision_frame_ratio"):
            if parsed[name] > 1.0:
                raise MetricsInputError(
                    "THRESHOLD_INVALID",
                    f"threshold {name} must be in [0, 1]",
                    field=name,
                    value=parsed[name],
                )

        maximum_violations = values["max_joint_limit_violations"]
        if (
            not isinstance(maximum_violations, (int, np.integer))
            or isinstance(maximum_violations, (bool, np.bool_))
            or int(maximum_violations) < 0
        ):
            raise MetricsInputError(
                "THRESHOLD_INVALID",
                "max_joint_limit_violations must be a non-negative integer",
                field="max_joint_limit_violations",
                value=maximum_violations,
            )
        parsed["max_joint_limit_violations"] = int(maximum_violations)

        require_nan_free = values["require_nan_free"]
        if not isinstance(require_nan_free, (bool, np.bool_)):
            raise MetricsInputError(
                "THRESHOLD_INVALID",
                "require_nan_free must be boolean",
                field="require_nan_free",
                value=require_nan_free,
            )
        parsed["require_nan_free"] = bool(require_nan_free)
        return cls(**parsed)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _numeric_array(name: str, values: Any) -> np.ndarray:
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MetricsInputError(
            "ARRAY_NOT_NUMERIC", f"{name} must be numeric", array=name
        ) from exc


def _frame_vector(name: str, values: Any, frame_count: int) -> np.ndarray:
    array = _numeric_array(name, values)
    if array.shape != (frame_count,):
        raise MetricsInputError(
            "ARRAY_SHAPE_INVALID",
            f"{name} must have shape ({frame_count},)",
            array=name,
            expected=[frame_count],
            actual=list(array.shape),
        )
    return array


def _flag_vector(name: str, values: Any, frame_count: int) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (frame_count,):
        raise MetricsInputError(
            "ARRAY_SHAPE_INVALID",
            f"{name} must have shape ({frame_count},)",
            array=name,
            expected=[frame_count],
            actual=list(array.shape),
        )
    if np.issubdtype(array.dtype, np.bool_):
        return array.astype(bool, copy=False)
    numeric = _numeric_array(name, values)
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
        raise MetricsInputError(
            "FLAG_ARRAY_INVALID",
            f"{name} must contain only booleans or numeric 0/1 values",
            array=name,
        )
    return numeric.astype(bool)


def _transform_array(name: str, values: Any, frame_count: int) -> np.ndarray:
    array = _numeric_array(name, values)
    if array.shape != (frame_count, 4, 4):
        raise MetricsInputError(
            "ARRAY_SHAPE_INVALID",
            f"{name} must have shape ({frame_count}, 4, 4)",
            array=name,
            expected=[frame_count, 4, 4],
            actual=list(array.shape),
        )
    return array


def _joint_limits(name: str, values: Any, joint_shape: tuple[int, int]) -> np.ndarray:
    array = _numeric_array(name, values)
    frame_count, joint_count = joint_shape
    if array.shape == (joint_count,):
        return np.broadcast_to(array, joint_shape)
    if array.shape == joint_shape:
        return array
    raise MetricsInputError(
        "ARRAY_SHAPE_INVALID",
        f"{name} must have shape ({joint_count},) or {joint_shape}",
        array=name,
        expected=[[joint_count], [frame_count, joint_count]],
        actual=list(array.shape),
    )


def _finite_stat(values: np.ndarray, operation: str) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    if operation == "median":
        return float(np.median(finite))
    if operation == "max":
        return float(np.max(finite))
    raise AssertionError(operation)


def _pose_errors(
    target_gripper_world: np.ndarray, achieved_gripper_world: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = len(target_gripper_world)
    position = np.full(frame_count, np.nan, dtype=float)
    orientation = np.full(frame_count, np.nan, dtype=float)
    finite = np.isfinite(target_gripper_world).all(axis=(1, 2)) & np.isfinite(
        achieved_gripper_world
    ).all(axis=(1, 2))
    indices = np.flatnonzero(finite)
    if indices.size:
        position[indices] = np.linalg.norm(
            achieved_gripper_world[indices, :3, 3] - target_gripper_world[indices, :3, 3],
            axis=1,
        )
        for index in indices:
            relative_rotation = (
                target_gripper_world[index, :3, :3].T
                @ achieved_gripper_world[index, :3, :3]
            )
            cosine = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
            orientation[index] = math.degrees(math.acos(cosine))
    return position, orientation


def _handle_relative_transform(handle_world: np.ndarray, gripper_world: np.ndarray) -> np.ndarray:
    rotation = handle_world[:3, :3]
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation.T @ gripper_world[:3, :3]
    result[:3, 3] = rotation.T @ (gripper_world[:3, 3] - handle_world[:3, 3])
    return result


def _grasp_drift_errors(
    phase_names: np.ndarray,
    handle_world: np.ndarray,
    target_gripper_world: np.ndarray,
    achieved_gripper_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    mask = np.isin(phase_names, ["close", "actuate"])
    eligible = np.flatnonzero(mask)
    position = np.full(len(phase_names), np.nan, dtype=float)
    orientation = np.full(len(phase_names), np.nan, dtype=float)
    reference: np.ndarray | None = None
    for index in eligible:
        if np.isfinite(handle_world[index]).all() and np.isfinite(
            target_gripper_world[index]
        ).all():
            reference = _handle_relative_transform(
                handle_world[index], target_gripper_world[index]
            )
            break
    if reference is None:
        return position, orientation, int(eligible.size)
    for index in eligible:
        if not (
            np.isfinite(handle_world[index]).all()
            and np.isfinite(achieved_gripper_world[index]).all()
        ):
            continue
        actual_relative = _handle_relative_transform(
            handle_world[index], achieved_gripper_world[index]
        )
        position[index] = float(
            np.linalg.norm(actual_relative[:3, 3] - reference[:3, 3])
        )
        relative_rotation = reference[:3, :3].T @ actual_relative[:3, :3]
        cosine = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
        orientation[index] = math.degrees(math.acos(cosine))
    return position, orientation, int(eligible.size)


def _at_least(value: float | int | None, threshold: float | int) -> dict[str, Any]:
    passed = value is not None and math.isfinite(float(value)) and float(value) >= float(threshold)
    return {"value": value, "operator": ">=", "threshold": threshold, "passed": bool(passed)}


def _at_most(value: float | int | None, threshold: float | int) -> dict[str, Any]:
    passed = value is not None and math.isfinite(float(value)) and float(value) <= float(threshold)
    return {"value": value, "operator": "<=", "threshold": threshold, "passed": bool(passed)}


def compute_metrics(
    *,
    phase_names: Sequence[str] | np.ndarray,
    door_angle_rad: Any,
    handle_world: Any,
    target_gripper_world: Any,
    achieved_gripper_world: Any,
    joint_q: Any,
    joint_lower: Any,
    joint_upper: Any,
    collision_flags: Any,
    ik_success_flags: Any,
    target_door_angle_rad: float,
    thresholds: Mapping[str, Any] | MetricThresholds,
) -> dict[str, Any]:
    """Compute rollout metrics and configuration-driven acceptance gates.

    Grasp drift is evaluated only on ``close`` and ``actuate`` frames.  It is
    the change in the actual handle-to-gripper transform relative to the first
    planned handle-to-gripper transform in those phases.
    """

    names = np.asarray(phase_names)
    if names.ndim != 1 or len(names) == 0:
        raise MetricsInputError(
            "PHASE_ARRAY_INVALID",
            "phase_names must be a non-empty one-dimensional array",
            actual=list(names.shape),
        )
    names = names.astype(str)
    frame_count = len(names)
    door = _frame_vector("door_angle_rad", door_angle_rad, frame_count)
    handle = _transform_array("handle_world", handle_world, frame_count)
    target = _transform_array("target_gripper_world", target_gripper_world, frame_count)
    achieved = _transform_array("achieved_gripper_world", achieved_gripper_world, frame_count)
    q = _numeric_array("joint_q", joint_q)
    if q.ndim != 2 or q.shape[0] != frame_count or q.shape[1] == 0:
        raise MetricsInputError(
            "ARRAY_SHAPE_INVALID",
            "joint_q must have shape (frame_count, nonzero_joint_count)",
            array="joint_q",
            expected=[frame_count, "J>0"],
            actual=list(q.shape),
        )
    lower = _joint_limits("joint_lower", joint_lower, q.shape)
    upper = _joint_limits("joint_upper", joint_upper, q.shape)
    finite_limit_pairs = np.isfinite(lower) & np.isfinite(upper)
    if np.any(finite_limit_pairs & (lower > upper)):
        raise MetricsInputError(
            "JOINT_LIMIT_ORDER_INVALID",
            "joint_lower must not exceed joint_upper",
        )
    collision = _flag_vector("collision_flags", collision_flags, frame_count)
    ik_success = _flag_vector("ik_success_flags", ik_success_flags, frame_count)
    parsed_thresholds = (
        thresholds
        if isinstance(thresholds, MetricThresholds)
        else MetricThresholds.from_mapping(thresholds)
    )
    try:
        target_door = float(target_door_angle_rad)
    except (TypeError, ValueError) as exc:
        raise MetricsInputError(
            "TARGET_DOOR_ANGLE_INVALID", "target_door_angle_rad must be numeric"
        ) from exc
    if not math.isfinite(target_door):
        raise MetricsInputError(
            "TARGET_DOOR_ANGLE_INVALID", "target_door_angle_rad must be finite"
        )

    numeric_arrays = (door, handle, target, achieved, q, lower, upper)
    has_nan = any(bool(np.isnan(array).any()) for array in numeric_arrays)
    has_infinite = any(bool(np.isinf(array).any()) for array in numeric_arrays)

    position_errors, orientation_errors = _pose_errors(target, achieved)
    grasp_position_drift, grasp_orientation_drift, grasp_frame_count = _grasp_drift_errors(
        names, handle, target, achieved
    )

    joint_violations = (q < lower) | (q > upper)
    joint_violation_frames = np.any(joint_violations, axis=1)
    joint_violation_count = int(np.count_nonzero(joint_violations))
    joint_violation_frame_count = int(np.count_nonzero(joint_violation_frames))
    joint_violation_frame_ratio = float(joint_violation_frame_count / frame_count)
    collision_frame_count = int(np.count_nonzero(collision))
    collision_frame_ratio = float(collision_frame_count / frame_count)
    ik_success_count = int(np.count_nonzero(ik_success))
    ik_success_rate = float(ik_success_count / frame_count)

    final_door_angle_deg: float | None
    final_door_angle_error_deg: float | None
    if math.isfinite(float(door[-1])):
        final_door_angle_deg = math.degrees(float(door[-1]))
        final_door_angle_error_deg = abs(final_door_angle_deg - math.degrees(target_door))
    else:
        final_door_angle_deg = None
        final_door_angle_error_deg = None

    median_position_error = _finite_stat(position_errors, "median")
    max_position_error = _finite_stat(position_errors, "max")
    median_orientation_error = _finite_stat(orientation_errors, "median")
    max_orientation_error = _finite_stat(orientation_errors, "max")
    median_grasp_position_drift = _finite_stat(grasp_position_drift, "median")
    max_grasp_position_drift = _finite_stat(grasp_position_drift, "max")
    median_grasp_orientation_drift = _finite_stat(grasp_orientation_drift, "median")
    max_grasp_orientation_drift = _finite_stat(grasp_orientation_drift, "max")

    gates = {
        "ik_waypoint_success_rate": _at_least(
            ik_success_rate, parsed_thresholds.min_ik_success_rate
        ),
        "median_ee_position_error_m": _at_most(
            median_position_error, parsed_thresholds.position_error_m
        ),
        "median_ee_orientation_error_deg": _at_most(
            median_orientation_error, parsed_thresholds.orientation_error_deg
        ),
        "joint_limit_violation_count": _at_most(
            joint_violation_count, parsed_thresholds.max_joint_limit_violations
        ),
        "joint_limit_violation_frame_ratio": _at_most(
            joint_violation_frame_ratio,
            parsed_thresholds.max_joint_limit_violation_frame_ratio,
        ),
        "collision_frame_ratio": _at_most(
            collision_frame_ratio, parsed_thresholds.max_collision_frame_ratio
        ),
        "final_door_angle_error_deg": _at_most(
            final_door_angle_error_deg, parsed_thresholds.final_door_angle_deg
        ),
        "max_handle_gripper_position_drift_m": _at_most(
            max_grasp_position_drift, parsed_thresholds.grasp_position_drift_m
        ),
        "max_handle_gripper_orientation_drift_deg": _at_most(
            max_grasp_orientation_drift, parsed_thresholds.grasp_orientation_drift_deg
        ),
        "nan_free": {
            "value": not has_nan,
            "operator": "required_true" if parsed_thresholds.require_nan_free else "not_required",
            "threshold": parsed_thresholds.require_nan_free,
            "passed": bool((not parsed_thresholds.require_nan_free) or (not has_nan)),
        },
    }

    metrics: dict[str, Any] = {
        "frame_count": int(frame_count),
        "phase_frame_counts": {
            phase: int(np.count_nonzero(names == phase)) for phase in np.unique(names)
        },
        "ik_waypoint_success_count": ik_success_count,
        "ik_waypoint_count": int(frame_count),
        "ik_waypoint_success_rate": ik_success_rate,
        "median_ee_position_error_m": median_position_error,
        "max_ee_position_error_m": max_position_error,
        "median_ee_orientation_error_deg": median_orientation_error,
        "max_ee_orientation_error_deg": max_orientation_error,
        "ee_error_valid_frame_count": int(np.count_nonzero(np.isfinite(position_errors))),
        "joint_limit_violation_count": joint_violation_count,
        "joint_limit_violation_frame_count": joint_violation_frame_count,
        "joint_limit_violation_frame_ratio": joint_violation_frame_ratio,
        "collision_frame_count": collision_frame_count,
        "collision_frame_ratio": collision_frame_ratio,
        "final_door_angle_deg": final_door_angle_deg,
        "target_door_angle_deg": math.degrees(target_door),
        "final_door_angle_error_deg": final_door_angle_error_deg,
        "has_nan": has_nan,
        "has_infinite": has_infinite,
        "grasp_drift_phase_frame_count": grasp_frame_count,
        "grasp_drift_valid_frame_count": int(
            np.count_nonzero(np.isfinite(grasp_position_drift))
        ),
        "median_handle_gripper_position_drift_m": median_grasp_position_drift,
        "max_handle_gripper_position_drift_m": max_grasp_position_drift,
        "median_handle_gripper_orientation_drift_deg": median_grasp_orientation_drift,
        "max_handle_gripper_orientation_drift_deg": max_grasp_orientation_drift,
        "thresholds": parsed_thresholds.to_dict(),
        "gates": gates,
    }
    metrics["success"] = all(bool(gate["passed"]) for gate in gates.values())
    return metrics
