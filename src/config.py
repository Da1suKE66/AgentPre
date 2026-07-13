"""Strict JSON configuration loading without hidden control defaults."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import FailureCode, PipelineError


PHASE_ORDER = ("pregrasp", "approach", "close", "actuate", "retreat")


def _lookup(data: dict[str, Any], dotted: str) -> Any:
    value: Any = data
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise PipelineError(
                FailureCode.CONFIG_INVALID,
                f"missing required config field: {dotted}",
                stage="config",
                details={"field": dotted},
            )
        value = value[key]
    return value


def _finite_vector(value: Any, length: int, dotted: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            f"{dotted} must contain exactly {length} values",
            stage="config",
            details={"field": dotted, "value": value},
        )
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            f"{dotted} contains a non-finite value",
            stage="config",
            details={"field": dotted, "value": value},
        )
    return result


def _validate_pose(data: dict[str, Any], dotted: str) -> None:
    pose = _lookup(data, dotted)
    if not isinstance(pose, dict):
        raise PipelineError(FailureCode.CONFIG_INVALID, f"{dotted} must be an object", stage="config")
    _finite_vector(_lookup(data, f"{dotted}.position"), 3, f"{dotted}.position")
    quat = _finite_vector(_lookup(data, f"{dotted}.orientation_wxyz"), 4, f"{dotted}.orientation_wxyz")
    norm = math.sqrt(sum(component * component for component in quat))
    if norm < 1.0e-12 or abs(norm - 1.0) > 1.0e-4:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            f"{dotted}.orientation_wxyz must be a normalized quaternion",
            stage="config",
            details={"field": f"{dotted}.orientation_wxyz", "norm": norm},
        )


def _positive_number(data: dict[str, Any], dotted: str, *, allow_zero: bool = False) -> float:
    raw = _lookup(data, dotted)
    if isinstance(raw, bool):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            f"{dotted} must be numeric, not boolean",
            stage="config",
            details={"field": dotted, "value": raw},
        )
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            f"{dotted} must be numeric",
            stage="config",
            details={"field": dotted, "value": raw},
        ) from exc
    valid = math.isfinite(value) and (value >= 0.0 if allow_zero else value > 0.0)
    if not valid:
        relation = "non-negative" if allow_zero else "positive"
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            f"{dotted} must be finite and {relation}",
            stage="config",
            details={"field": dotted, "value": raw},
        )
    return value


def _positive_integer(data: dict[str, Any], dotted: str, *, allow_zero: bool = False) -> int:
    raw = _lookup(data, dotted)
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            f"{dotted} must be an integer",
            stage="config",
            details={"field": dotted, "value": raw},
        )
    if raw < 0 if allow_zero else raw <= 0:
        relation = "non-negative" if allow_zero else "positive"
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            f"{dotted} must be {relation}",
            stage="config",
            details={"field": dotted, "value": raw},
        )
    return raw


@dataclass(frozen=True)
class ProjectConfig:
    """Validated configuration plus deterministic path resolution."""

    path: Path
    project_root: Path
    data: dict[str, Any]

    def get(self, dotted: str) -> Any:
        return _lookup(self.data, dotted)

    def resolve_path(self, value: str | os.PathLike[str]) -> Path:
        cache_default = self.project_root / ".agentpre-cache"
        expanded = str(value).replace(
            "${AGENTPRE_CACHE_ROOT}",
            os.environ.get("AGENTPRE_CACHE_ROOT", str(cache_default)),
        )
        expanded = os.path.expanduser(os.path.expandvars(expanded))
        path = Path(expanded)
        return path if path.is_absolute() else (self.project_root / path).resolve()

    def asset_path(self, kind: str) -> Path:
        return self.resolve_path(str(self.get(f"assets.{kind}.urdf")))


def validate_config(data: dict[str, Any]) -> None:
    """Validate every field required by the deterministic first stage."""

    if not isinstance(data, dict):
        raise PipelineError(FailureCode.CONFIG_INVALID, "config root must be an object", stage="config")

    required_strings = (
        "assets.object.name",
        "assets.object.urdf",
        "assets.object.door_joint",
        "assets.object.door_link",
        "assets.object.handle_link",
        "assets.object.affordances",
        "assets.object.handle_frame",
        "assets.robot.name",
        "assets.robot.urdf",
        "assets.robot.end_effector_link",
        "runtime.device",
        "output.root",
    )
    for dotted in required_strings:
        value = _lookup(data, dotted)
        if not isinstance(value, str) or not value.strip():
            raise PipelineError(
                FailureCode.CONFIG_INVALID,
                f"{dotted} must be a non-empty string",
                stage="config",
                details={"field": dotted},
            )

    _validate_pose(data, "assets.object.world_pose")
    _validate_pose(data, "assets.robot.world_pose")
    _validate_pose(data, "assets.robot.end_effector_offset")
    _validate_pose(data, "task.grasp_offset")

    seed = _lookup(data, "seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise PipelineError(FailureCode.CONFIG_INVALID, "seed must be a non-negative integer", stage="config")

    arm_joint_names = _lookup(data, "assets.robot.arm_joint_names")
    default_q = _lookup(data, "assets.robot.default_joint_positions")
    if (
        not isinstance(arm_joint_names, list)
        or not arm_joint_names
        or not all(isinstance(name, str) and name for name in arm_joint_names)
    ):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "assets.robot.arm_joint_names must be a non-empty list of names",
            stage="config",
        )
    _finite_vector(default_q, len(arm_joint_names), "assets.robot.default_joint_positions")
    if len(set(arm_joint_names)) != len(arm_joint_names):
        raise PipelineError(FailureCode.CONFIG_INVALID, "robot arm joint names must be unique", stage="config")

    for phase in PHASE_ORDER:
        count = _lookup(data, f"task.phases.{phase}.samples")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise PipelineError(
                FailureCode.CONFIG_INVALID,
                f"task.phases.{phase}.samples must be a positive integer",
                stage="config",
            )

    _positive_number(data, "task.goal_angle_deg")
    _positive_number(data, "task.pregrasp_distance_m", allow_zero=True)
    _positive_number(data, "task.retreat_distance_m", allow_zero=True)
    _positive_number(data, "simulation.dt")
    _positive_integer(data, "simulation.physics_substeps")
    _positive_integer(data, "simulation.solver_iterations")
    _positive_integer(data, "ik.iterations")
    _positive_integer(data, "ik.max_recovery_seeds")
    _positive_integer(data, "runtime.threads")
    _positive_number(data, "ik.position_weight")
    _positive_number(data, "ik.rotation_weight")
    _positive_number(data, "ik.joint_limit_weight")
    _positive_number(data, "ik.nominal_posture_weight", allow_zero=True)
    _positive_number(data, "ik.step_size")
    _positive_number(data, "ik.lambda_initial")
    _positive_number(data, "ik.recovery_noise_std", allow_zero=True)
    _positive_number(data, "ik.joint_limit_tolerance_rad", allow_zero=True)
    _positive_number(data, "affordance_generation.width_margin_m", allow_zero=True)
    _positive_number(data, "affordance_generation.max_gripper_width_m")
    _positive_integer(data, "affordance_generation.max_candidates")
    radial_samples = _positive_integer(data, "affordance_generation.primitive_radial_samples")
    if radial_samples < 4:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "affordance_generation.primitive_radial_samples must be at least 4",
            stage="config",
        )
    _positive_number(data, "assets.robot.open_gripper_width_m")
    _positive_number(data, "assets.robot.closed_gripper_width_m")
    _finite_vector(_lookup(data, "simulation.gravity_m_s2"), 3, "simulation.gravity_m_s2")
    for dotted in (
        "collision.margin_m",
        "collision.candidate_reach_min_m",
        "simulation.robot_pd.arm_stiffness",
        "simulation.robot_pd.arm_damping",
        "simulation.robot_pd.finger_stiffness",
        "simulation.robot_pd.finger_damping",
        "simulation.robot_pd.arm_effort_limit",
        "simulation.robot_pd.finger_effort_limit",
    ):
        _positive_number(data, dotted, allow_zero=dotted == "collision.margin_m")
    reach_max = _positive_number(data, "collision.candidate_reach_max_m")
    reach_min = float(_lookup(data, "collision.candidate_reach_min_m"))
    if reach_min >= reach_max:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "collision candidate reach minimum must be less than maximum",
            stage="config",
        )

    for dotted in (
        "thresholds.position_error_m",
        "thresholds.orientation_error_deg",
        "thresholds.final_door_angle_deg",
        "thresholds.grasp_position_drift_m",
        "thresholds.grasp_orientation_drift_deg",
    ):
        _positive_number(data, dotted)
    _positive_number(data, "thresholds.max_collision_frame_ratio", allow_zero=True)
    _positive_integer(data, "thresholds.max_joint_limit_violations", allow_zero=True)
    _positive_number(data, "thresholds.max_joint_limit_violation_frame_ratio", allow_zero=True)
    if not isinstance(_lookup(data, "thresholds.require_nan_free"), bool):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "thresholds.require_nan_free must be boolean",
            stage="config",
        )
    success_rate = float(_lookup(data, "thresholds.min_ik_success_rate"))
    if not math.isfinite(success_rate) or not 0.0 <= success_rate <= 1.0:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "thresholds.min_ik_success_rate must be in [0, 1]",
            stage="config",
        )
    for dotted in (
        "thresholds.max_collision_frame_ratio",
        "thresholds.max_joint_limit_violation_frame_ratio",
    ):
        ratio = float(_lookup(data, dotted))
        if ratio > 1.0:
            raise PipelineError(
                FailureCode.CONFIG_INVALID,
                f"{dotted} must be in [0, 1]",
                stage="config",
            )

    if str(_lookup(data, "runtime.device")).lower() != "cpu":
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "the checked-in first-stage config must use CPU to avoid the occupied GPU",
            stage="config",
            details={"field": "runtime.device"},
        )


def load_config(path: str | os.PathLike[str]) -> ProjectConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            f"config file does not exist: {config_path}",
            stage="config",
            details={"path": str(config_path)},
        )
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            f"cannot read config: {config_path}",
            stage="config",
            details={"error": repr(exc)},
        ) from exc
    validate_config(data)
    root_setting = data.get("project_root", "..")
    project_root = (config_path.parent / str(root_setting)).resolve()
    return ProjectConfig(path=config_path, project_root=project_root, data=data)
