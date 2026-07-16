"""Strict JSON configuration loading without hidden control defaults."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import FailureCode, PipelineError


PHASE_ORDER = (
    "pregrasp",
    "approach",
    "close",
    "actuate",
    "release",
    "retreat",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_RUNTIME_DEVICE_PATTERN = re.compile(r"(?:cpu|cuda(?::[0-9]+)?)")


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


def _finite_number(data: dict[str, Any], dotted: str) -> float:
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
    if not math.isfinite(value):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            f"{dotted} must be finite",
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
    invalid = raw < 0 if allow_zero else raw <= 0
    if invalid:
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
        # This deployment intentionally keeps environments, downloaded assets,
        # and bulky rollouts off the small /workspace filesystem.  The
        # environment variable remains available for portable test overrides.
        cache_default = Path("/cache/liluchen/agentpre")
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

    if data.get("schema_version") != 1:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "schema_version must be exactly 1",
            stage="config",
            details={"field": "schema_version", "value": data.get("schema_version")},
        )
    project_root = data.get("project_root")
    if not isinstance(project_root, str) or not project_root.strip():
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "project_root must be a non-empty path string",
            stage="config",
            details={"field": "project_root", "value": project_root},
        )

    required_strings = (
        "assets.object.name",
        "assets.object.urdf",
        "assets.object.expected_urdf_sha256",
        "assets.object.door_joint",
        "assets.object.door_link",
        "assets.object.handle_link",
        "assets.object.affordances",
        "assets.object.handle_frame",
        "assets.robot.name",
        "assets.robot.urdf",
        "assets.robot.expected_urdf_sha256",
        "assets.robot.end_effector_link",
        "collision.broad_phase",
        "collision.scope",
        "ik.optimizer",
        "ik.jacobian",
        "simulation.solver",
        "simulation.robot_control.backend",
        "simulation.robot_control.implementation",
        "simulation.robot_control.target_velocity_mode",
        "simulation.door_control.backend",
        "simulation.fixed_grasp_constraint.activate_after_phase",
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

    for dotted in (
        "assets.object.expected_urdf_sha256",
        "assets.robot.expected_urdf_sha256",
    ):
        value = _lookup(data, dotted)
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise PipelineError(
                FailureCode.CONFIG_INVALID,
                f"{dotted} must be a lowercase 64-character SHA-256 digest",
                stage="config",
                details={"field": dotted, "value": value},
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

    finger_joint_names = _lookup(data, "assets.robot.finger_joint_names")
    if (
        not isinstance(finger_joint_names, list)
        or len(finger_joint_names) != 2
        or not all(isinstance(name, str) and name.strip() for name in finger_joint_names)
        or len(set(finger_joint_names)) != 2
    ):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "assets.robot.finger_joint_names must contain exactly two unique names",
            stage="config",
        )
    if set(arm_joint_names) & set(finger_joint_names):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "arm and finger joint names must not overlap",
            stage="config",
        )

    phases = _lookup(data, "task.phases")
    if not isinstance(phases, dict):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "task.phases must be an object",
            stage="config",
            details={"field": "task.phases", "value": phases},
        )
    missing_phases = [phase for phase in PHASE_ORDER if phase not in phases]
    extra_phases = [phase for phase in phases if phase not in PHASE_ORDER]
    if missing_phases or extra_phases:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "task.phases must contain exactly the six configured phases",
            stage="config",
            details={
                "field": "task.phases",
                "missing": missing_phases,
                "extra": extra_phases,
                "phase_order": list(PHASE_ORDER),
            },
        )
    for phase in PHASE_ORDER:
        _positive_integer(data, f"task.phases.{phase}.samples")

    closed_angle_deg = _finite_number(data, "task.closed_angle_deg")
    goal_angle_deg = _finite_number(data, "task.goal_angle_deg")
    if math.isclose(goal_angle_deg, closed_angle_deg, abs_tol=1.0e-12, rel_tol=0.0):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "task.goal_angle_deg must differ from task.closed_angle_deg",
            stage="config",
            details={"closed_angle_deg": closed_angle_deg, "goal_angle_deg": goal_angle_deg},
        )
    _positive_number(data, "task.pregrasp_distance_m", allow_zero=True)
    _positive_number(data, "task.retreat_distance_m", allow_zero=True)
    _positive_number(data, "simulation.dt")
    _positive_integer(data, "simulation.physics_substeps")
    _positive_integer(data, "simulation.solver_iterations")
    release_blend_frames = _positive_integer(
        data, "simulation.robot_control.grasp_release_blend_frames"
    )
    if release_blend_frames < 2:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "simulation.robot_control.grasp_release_blend_frames must be at least 2",
            stage="config",
            details={
                "field": "simulation.robot_control.grasp_release_blend_frames",
                "value": release_blend_frames,
            },
        )
    for dotted in (
        "simulation.robot_control.arm_joint_tracking_reserve_rad",
        "simulation.robot_control.arm_stiffness",
        "simulation.robot_control.arm_damping",
        "simulation.robot_control.finger_stiffness",
        "simulation.robot_control.finger_damping",
        "simulation.fixed_grasp_constraint.activation_position_tolerance_m",
        "simulation.fixed_grasp_constraint.activation_orientation_tolerance_deg",
        "simulation.fixed_grasp_constraint.activation_linear_velocity_tolerance_m_s",
        "simulation.fixed_grasp_constraint.activation_angular_velocity_tolerance_deg_s",
    ):
        _positive_number(data, dotted)
    door_stiffness = _positive_number(
        data, "simulation.door_control.target_stiffness", allow_zero=True
    )
    if door_stiffness != 0.0:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "simulation.door_control.target_stiffness must be exactly 0",
            stage="config",
            details={"value": door_stiffness},
        )
    _positive_number(data, "simulation.door_control.target_damping")
    door_target_velocity = _finite_number(
        data, "simulation.door_control.target_velocity_rad_s"
    )
    if door_target_velocity != 0.0:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "simulation.door_control.target_velocity_rad_s must be exactly 0",
            stage="config",
            details={"value": door_target_velocity},
        )
    _positive_integer(data, "ik.iterations")
    runtime_threads = _positive_integer(data, "runtime.threads")
    if runtime_threads != 1:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "runtime.threads must be 1 for deterministic host-side preprocessing",
            stage="config",
            details={"field": "runtime.threads", "value": runtime_threads},
        )
    _positive_number(data, "ik.position_weight")
    _positive_number(data, "ik.rotation_weight")
    _positive_number(data, "ik.joint_limit_weight")
    _positive_number(data, "ik.nominal_posture_weight", allow_zero=True)
    _positive_number(data, "ik.continuity_weight")
    _positive_number(data, "ik.control_limit_margin_rad")
    _positive_number(data, "ik.step_size")
    _positive_number(data, "ik.lambda_initial")
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
    open_width = _positive_number(data, "assets.robot.open_gripper_width_m")
    closed_width = _positive_number(data, "assets.robot.closed_gripper_width_m")
    if closed_width > open_width:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "closed gripper width must not exceed open gripper width",
            stage="config",
            details={"closed_width_m": closed_width, "open_width_m": open_width},
        )
    candidate_max_width = float(_lookup(data, "affordance_generation.max_gripper_width_m"))
    if candidate_max_width > open_width:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "affordance max gripper width must not exceed the robot open width",
            stage="config",
            details={
                "candidate_max_width_m": candidate_max_width,
                "robot_open_width_m": open_width,
            },
        )
    _finite_vector(_lookup(data, "simulation.gravity_m_s2"), 3, "simulation.gravity_m_s2")
    for dotted in (
        "collision.margin_m",
        "collision.candidate_reach_min_m",
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
        "thresholds.max_joint_velocity_limit_ratio",
        "thresholds.max_joint_acceleration_rad_s2",
        "thresholds.max_joint_jerk_rad_s3",
        "thresholds.max_finger_acceleration_m_s2",
        "thresholds.max_finger_jerk_m_s3",
    ):
        _positive_number(data, dotted)
    _positive_number(data, "thresholds.max_collision_frame_ratio", allow_zero=True)
    _positive_integer(data, "thresholds.max_joint_limit_violations", allow_zero=True)
    _positive_number(data, "thresholds.max_joint_limit_violation_frame_ratio", allow_zero=True)
    boolean_fields = (
        "collision.enabled",
        "collision.deterministic",
        "simulation.fixed_grasp_constraint.enabled",
        "thresholds.require_nan_free",
        "output.write_rollout_jsonl",
        "output.write_resolved_config",
    )
    for dotted in boolean_fields:
        if not isinstance(_lookup(data, dotted), bool):
            raise PipelineError(
                FailureCode.CONFIG_INVALID,
                f"{dotted} must be boolean",
                stage="config",
                details={"field": dotted},
            )

    if str(_lookup(data, "ik.optimizer")).lower() != "lm":
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "ik.optimizer must be 'lm' for the Newton first-stage adapter",
            stage="config",
        )
    if str(_lookup(data, "ik.jacobian")).lower() != "analytic":
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "ik.jacobian must be 'analytic'",
            stage="config",
        )
    if str(_lookup(data, "simulation.solver")).lower() != "xpbd":
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "simulation.solver must be 'xpbd'",
            stage="config",
        )
    if str(_lookup(data, "simulation.robot_control.backend")).lower() != "joint_pd":
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "simulation.robot_control.backend must be 'joint_pd'",
            stage="config",
            details={
                "field": "simulation.robot_control.backend",
                "value": _lookup(data, "simulation.robot_control.backend"),
            },
        )
    if (
        str(_lookup(data, "simulation.robot_control.implementation")).lower()
        != "newton_xpbd_joint_targets"
    ):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "simulation.robot_control.implementation must be 'newton_xpbd_joint_targets'",
            stage="config",
        )
    if (
        str(_lookup(data, "simulation.robot_control.target_velocity_mode")).lower()
        != "finite_difference"
    ):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "simulation.robot_control.target_velocity_mode must be 'finite_difference'",
            stage="config",
        )
    if (
        str(_lookup(data, "simulation.door_control.backend")).lower()
        != "passive_velocity_damping"
    ):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "simulation.door_control.backend must be 'passive_velocity_damping'",
            stage="config",
        )
    if str(_lookup(data, "collision.broad_phase")).lower() != "sap":
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "collision.broad_phase must be 'sap'",
            stage="config",
        )
    if str(_lookup(data, "collision.scope")).lower() != "cross_asset_robot_object":
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "collision.scope must be 'cross_asset_robot_object' for the current audited backend",
            stage="config",
            details={
                "field": "collision.scope",
                "value": _lookup(data, "collision.scope"),
            },
        )
    if _lookup(data, "simulation.fixed_grasp_constraint.activate_after_phase") != "close":
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "fixed grasp activation must occur after the close phase",
            stage="config",
        )

    allowed_links = _lookup(data, "collision.allowed_contact_links")
    if (
        not isinstance(allowed_links, list)
        or not allowed_links
        or not all(isinstance(name, str) and name.strip() for name in allowed_links)
        or len(set(allowed_links)) != len(allowed_links)
    ):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "collision.allowed_contact_links must contain unique non-empty names",
            stage="config",
        )

    success_rate = _finite_number(data, "thresholds.min_ik_success_rate")
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

    runtime_device = str(_lookup(data, "runtime.device")).lower()
    if _RUNTIME_DEVICE_PATTERN.fullmatch(runtime_device) is None:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "runtime.device must be 'cpu', 'cuda', or 'cuda:<index>'",
            stage="config",
            details={"field": "runtime.device", "value": runtime_device},
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
