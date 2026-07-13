"""Command-line entry point for deterministic articulated-manipulation runs.

The checked-in first stage is intentionally explicit: asset names, poses,
solver settings, collision settings, thresholds, phase lengths, and output
locations are read from :class:`src.config.ProjectConfig`.  Project-facing
quaternions use ``wxyz`` throughout; Newton conversion remains isolated in
``src.newton_backend``.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from .affordances import (
    AffordanceError,
    CandidateGenerationConfig,
    CheckResult,
    GraspCandidate,
    resolve_handle_candidates,
    select_grasp_candidate,
)
from .asset_inspector import inspect_asset
from .config import PHASE_ORDER, ProjectConfig, load_config
from .door_kinematics import DoorKinematics
from .errors import FailureCode, PipelineError
from .metrics import MetricThresholds, MetricsInputError, compute_metrics
from .newton_backend import NewtonFrankaIKBackend, NewtonIKParameters
from .output import write_json, write_jsonl, write_trajectory
from .trajectory import TaskPlan, generate_task_plan
from .transforms import compose_transforms, decompose_pose, pose_matrix
from .urdf_model import URDFModelError, load_urdf


BackendFactory = Callable[[NewtonIKParameters], Any]


class CollisionEvaluator(Protocol):
    """Minimal runner-facing collision API.

    The concrete implementation may use Newton shapes, but it must resolve
    bodies and joints by configured names.  Tests inject a deterministic fake
    so CLI orchestration does not require the optional Newton installation.
    """

    def candidate_is_collision_free(
        self,
        candidate: GraspCandidate,
        gripper_world: np.ndarray,
        arm_joint_q: np.ndarray,
    ) -> bool | CheckResult | tuple[bool, str]:
        ...

    def trajectory_collision_flags(
        self,
        plan: TaskPlan,
        arm_joint_q: np.ndarray,
    ) -> Sequence[bool] | np.ndarray:
        ...


CollisionFactory = Callable[
    [ProjectConfig, DoorKinematics, Any], CollisionEvaluator
]


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Completed run plus the CLI exit decision."""

    output_dir: Path
    metrics: Mapping[str, Any]
    exit_code: int


class _RunLog:
    """Deterministic structured events written to ``run.log`` as JSONL."""

    def __init__(self, *, mode: str, seed: int, path: Path | None = None) -> None:
        self._rows: list[dict[str, Any]] = []
        self._path = path
        self.add("run_started", mode=mode, seed=seed)

    @property
    def rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._rows)

    def add(self, event: str, **details: Any) -> None:
        self._rows.append(
            {
                "event_index": len(self._rows),
                "event": event,
                "details": details,
            }
        )
        if self._path is not None:
            write_jsonl(self._path, self._rows)


def _distribution_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _source_state(project_root: Path) -> dict[str, Any]:
    """Return auditable Git state without making Git a runtime requirement."""

    result: dict[str, Any] = {"git_commit": None, "git_dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError):
        return result
    result["git_commit"] = commit.stdout.strip() or None
    result["git_dirty"] = bool(status.stdout.strip())
    return result


def _resolved_config_payload(
    config: ProjectConfig, *, output_dir: Path
) -> dict[str, Any]:
    """Resolve filesystem placeholders and record the actual runtime state."""

    payload = copy.deepcopy(config.data)
    payload["project_root"] = str(config.project_root)
    payload["assets"]["object"]["urdf"] = str(config.asset_path("object"))
    payload["assets"]["object"]["affordances"] = str(
        config.resolve_path(str(config.get("assets.object.affordances")))
    )
    payload["assets"]["robot"]["urdf"] = str(config.asset_path("robot"))
    payload["assets"]["robot"]["bootstrap_source"] = str(
        config.resolve_path(str(config.get("assets.robot.bootstrap_source")))
    )
    payload["output"]["root"] = str(
        config.resolve_path(str(config.get("output.root")))
    )
    payload["resolved_runtime"] = {
        "output_dir": str(output_dir),
        "python": sys.version.split()[0],
        "newton": _distribution_version("newton"),
        "warp_lang": _distribution_version("warp-lang"),
        "numpy": _distribution_version("numpy"),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "AGENTPRE_CACHE_ROOT",
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "XDG_CACHE_HOME",
                "WARP_CACHE_PATH",
                "NEWTON_CACHE_PATH",
                "TMPDIR",
            )
        },
        **_source_state(config.project_root),
    }
    return payload


def _configure_deterministic_runtime(config: ProjectConfig) -> None:
    """Apply only runtime controls that are explicitly present in config."""

    threads = str(int(config.get("runtime.threads")))
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = threads
    if str(config.get("runtime.device")).lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    seed = int(config.get("seed"))
    random.seed(seed)
    np.random.seed(seed)


def _configured_pose(config: ProjectConfig, dotted: str) -> np.ndarray:
    value = config.get(dotted)
    return pose_matrix(value["position"], value["orientation_wxyz"])


def _output_directory(
    config: ProjectConfig,
    override: str | Path | None,
    *,
    mode: str = "kinematic",
) -> Path:
    """Resolve an explicit directory or reserve a non-overwriting run folder."""

    if override is not None:
        return Path(override).expanduser().resolve()
    root = config.resolve_path(str(config.get("output.root")))
    root.mkdir(parents=True, exist_ok=True)
    normalized_mode = str(mode).strip().replace("/", "_")
    if not normalized_mode:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "run mode cannot be empty when allocating an output directory",
            stage="output",
        )
    prefix = f"{normalized_mode}_seed_{int(config.get('seed'))}"
    for run_number in range(1, 1_000_000):
        candidate = root / f"{prefix}_{run_number:04d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        except OSError as exc:
            raise PipelineError(
                FailureCode.OUTPUT_FAILURE,
                f"failed to reserve output directory below {root}",
                stage="output",
                details={"error": repr(exc), "root": str(root)},
            ) from exc
        return candidate
    raise PipelineError(
        FailureCode.OUTPUT_FAILURE,
        f"could not allocate a unique output directory below {root}",
        stage="output",
        details={"root": str(root)},
    )


def _bool_config(config: ProjectConfig, dotted: str) -> bool:
    value = config.get(dotted)
    if not isinstance(value, bool):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            f"{dotted} must be boolean",
            stage="config",
            details={"field": dotted, "value": value},
        )
    return value


def _newton_parameters(config: ProjectConfig) -> NewtonIKParameters:
    try:
        return NewtonIKParameters.from_project_config(
            config,
            joint_limit_tolerance=float(config.get("ik.joint_limit_tolerance_rad")),
        )
    except PipelineError:
        raise
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "Newton IK parameters are invalid",
            stage="ik_configuration",
            details={"exception_type": type(exc).__name__, "error": str(exc)},
        ) from exc


def _initial_tcp_transform(backend: Any) -> np.ndarray:
    """Read the nominal TCP through the narrowest backend capability.

    ``initial_tcp_transform`` is the preferred public extension point.  The
    Newton v1.3 adapter predates that method, so its FK helper is used only as
    a compatibility path; neither route relies on body or joint indices.
    """

    public = getattr(backend, "initial_tcp_transform", None)
    if callable(public):
        transform = np.asarray(public(), dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise PipelineError(
                FailureCode.NUMERICAL_INSTABILITY,
                "backend returned an invalid nominal TCP transform",
                stage="initial_fk",
                details={"shape": list(transform.shape)},
            )
        return transform

    forward = getattr(backend, "_forward_kinematics_tcp", None)
    nominal = getattr(backend, "nominal_joint_q", None)
    if not callable(forward) or nominal is None:
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "IK backend does not expose nominal TCP forward kinematics",
            stage="initial_fk",
        )
    position, orientation_wxyz = forward(np.asarray(nominal, dtype=float))
    try:
        return pose_matrix(position, orientation_wxyz)
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "nominal TCP forward kinematics is non-finite or malformed",
            stage="initial_fk",
            details={"error": str(exc)},
        ) from exc


def _arm_joint_limits(config: ProjectConfig) -> tuple[np.ndarray, np.ndarray]:
    """Resolve configured arm-joint limits by URDF name, never model index."""

    robot_path = config.asset_path("robot")
    if not robot_path.is_file():
        raise PipelineError(
            FailureCode.ASSET_MISSING,
            f"robot URDF does not exist: {robot_path}",
            stage="robot_asset",
            details={"path": str(robot_path)},
        )
    try:
        model = load_urdf(robot_path)
    except URDFModelError as exc:
        raise PipelineError(
            FailureCode.ASSET_INVALID,
            "robot URDF cannot be parsed",
            stage="robot_asset",
            details=exc.to_dict(),
        ) from exc

    lower: list[float] = []
    upper: list[float] = []
    for joint_name in config.get("assets.robot.arm_joint_names"):
        try:
            joint = model.require_joint(str(joint_name))
        except URDFModelError as exc:
            raise PipelineError(
                FailureCode.ASSET_INVALID,
                "configured arm joint is absent from the robot URDF",
                stage="robot_asset",
                details=exc.to_dict(),
            ) from exc
        if joint.joint_type == "continuous":
            lower.append(-math.inf)
            upper.append(math.inf)
            continue
        if joint.limit is None or joint.limit.lower is None or joint.limit.upper is None:
            raise PipelineError(
                FailureCode.ASSET_INVALID,
                "configured arm joint has no lower/upper URDF limit",
                stage="robot_asset",
                details={"joint": joint.name, "joint_type": joint.joint_type},
            )
        lower.append(float(joint.limit.lower))
        upper.append(float(joint.limit.upper))
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _candidate_target(
    kinematics: DoorKinematics,
    candidate: GraspCandidate,
    closed_angle_rad: float,
    handle_to_gripper: np.ndarray,
) -> np.ndarray:
    handle_world = kinematics.handle_frame_transform(
        closed_angle_rad, candidate.transform
    )
    return compose_transforms(handle_world, handle_to_gripper)


def _single_target_is_reachable(
    *,
    config: ProjectConfig,
    backend: Any,
    gripper_world: np.ndarray,
) -> tuple[CheckResult, np.ndarray | None]:
    robot_position = np.asarray(
        config.get("assets.robot.world_pose")["position"], dtype=float
    )
    distance = float(np.linalg.norm(gripper_world[:3, 3] - robot_position))
    reach_min = float(config.get("collision.candidate_reach_min_m"))
    reach_max = float(config.get("collision.candidate_reach_max_m"))
    if not reach_min <= distance <= reach_max:
        return (
            CheckResult(
                False,
                "candidate is outside the configured radial reach interval",
                {"distance_m": distance, "minimum_m": reach_min, "maximum_m": reach_max},
            ),
            None,
        )

    position, orientation = decompose_pose(gripper_world)
    try:
        result = backend.solve_waypoints([position], [orientation])
    except PipelineError:
        raise
    except Exception as exc:
        return (
            CheckResult(
                False,
                "candidate IK reachability solve failed",
                {"exception_type": type(exc).__name__, "error": str(exc)},
            ),
            None,
        )
    waypoints = tuple(getattr(result, "waypoints", ()))
    if len(waypoints) != 1:
        return (
            CheckResult(
                False,
                "candidate IK reachability solve returned an invalid result count",
                {"waypoint_count": len(waypoints)},
            ),
            None,
        )
    validation = getattr(waypoints[0], "validation", None)
    success = bool(getattr(validation, "success", False))
    arm_joint_q = np.asarray(
        getattr(waypoints[0], "arm_joint_positions", ()), dtype=float
    )
    if success and (arm_joint_q.ndim != 1 or not np.isfinite(arm_joint_q).all()):
        return (
            CheckResult(
                False,
                "candidate IK returned malformed arm joint positions",
                {"shape": list(arm_joint_q.shape)},
            ),
            None,
        )
    return (
        CheckResult(
            success,
            None if success else "candidate failed post-FK IK validation",
            {
                "distance_m": distance,
                "failed_checks": list(getattr(validation, "failed_checks", ())),
            },
        ),
        arm_joint_q if success else None,
    )


class _NamedCollisionEvaluator:
    """Adapter from runner arrays to the name-based collision backend."""

    def __init__(self, config: ProjectConfig, kinematics: DoorKinematics) -> None:
        from .collision import NamedAABBCollisionChecker

        self.config = config
        self.kinematics = kinematics
        self.robot_urdf = config.asset_path("robot")
        self.robot_model = load_urdf(self.robot_urdf)
        self.robot_world = _configured_pose(config, "assets.robot.world_pose")
        self.arm_joint_names = tuple(
            str(name) for name in config.get("assets.robot.arm_joint_names")
        )
        finger_names = config.get("assets.robot.finger_joint_names")
        if (
            not isinstance(finger_names, list)
            or any(not isinstance(name, str) or not name for name in finger_names)
            or len(set(finger_names)) != len(finger_names)
        ):
            raise PipelineError(
                FailureCode.CONFIG_INVALID,
                "assets.robot.finger_joint_names must be a list of unique names",
                stage="collision_setup",
            )
        self.finger_joint_names = tuple(finger_names)
        allowed_links = config.get("collision.allowed_contact_links")
        if not isinstance(allowed_links, list):
            raise PipelineError(
                FailureCode.CONFIG_INVALID,
                "collision.allowed_contact_links must be a list",
                stage="collision_setup",
            )
        self.checker = NamedAABBCollisionChecker(
            self.robot_urdf,
            config.asset_path("object"),
            broad_phase=str(config.get("collision.broad_phase")),
            margin_m=float(config.get("collision.margin_m")),
            allowed_contact_links=allowed_links,
        )
        self.candidate_reports: list[dict[str, Any]] = []
        self.last_report: Any | None = None

    def _robot_transforms(
        self, arm_joint_q: np.ndarray, gripper_width_m: float
    ) -> Mapping[str, np.ndarray]:
        from .collision import named_link_fk

        q = np.asarray(arm_joint_q, dtype=float)
        if q.shape != (len(self.arm_joint_names),) or not np.isfinite(q).all():
            raise PipelineError(
                FailureCode.NUMERICAL_INSTABILITY,
                "collision FK received malformed arm joint positions",
                stage="collision_evaluation",
                details={
                    "expected": [len(self.arm_joint_names)],
                    "actual": list(q.shape),
                },
            )
        joint_positions = {
            name: float(value)
            for name, value in zip(self.arm_joint_names, q, strict=True)
        }
        if self.finger_joint_names:
            # Configured gripper width is the total opening; distribute it over
            # every configured symmetric finger joint without assuming names.
            coordinate = float(gripper_width_m) / len(self.finger_joint_names)
            joint_positions.update(
                {name: coordinate for name in self.finger_joint_names}
            )
        return named_link_fk(
            self.robot_model,
            self.robot_world,
            joint_positions,
        )

    def candidate_is_collision_free(
        self,
        candidate: GraspCandidate,
        gripper_world: np.ndarray,
        arm_joint_q: np.ndarray,
    ) -> CheckResult:
        del gripper_world
        robot_transforms = self._robot_transforms(
            arm_joint_q, candidate.gripper_width_m
        )
        object_transforms = self.kinematics.link_transforms(
            math.radians(float(self.config.get("task.closed_angle_deg")))
        )
        result = self.checker.check_candidate(robot_transforms, object_transforms)
        evidence = {"candidate_id": candidate.candidate_id, **result.to_dict()}
        self.candidate_reports.append(evidence)
        return CheckResult(
            result.collision_free,
            None if result.collision_free else "candidate has a conservative collision",
            evidence,
        )

    def trajectory_collision_flags(
        self, plan: TaskPlan, arm_joint_q: np.ndarray
    ) -> Sequence[bool] | np.ndarray:
        robot_frames = [
            self._robot_transforms(arm_joint_q[index], plan.gripper_width_m[index])
            for index in range(len(plan.phase_names))
        ]
        object_frames = [
            self.kinematics.link_transforms(float(angle))
            for angle in plan.door_angle_rad
        ]
        self.last_report = self.checker.check_trajectory(robot_frames, object_frames)
        return self.last_report.flags


def _default_collision_factory(
    config: ProjectConfig,
    kinematics: DoorKinematics,
    backend: Any,
) -> CollisionEvaluator:
    """Load the optional Newton collision checker without hiding its absence."""

    del backend
    try:
        return _NamedCollisionEvaluator(config, kinematics)
    except PipelineError:
        raise
    except (ImportError, AttributeError) as exc:
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "collision.enabled=true but the Newton collision evaluator is unavailable",
            stage="collision_setup",
            details={"exception_type": type(exc).__name__, "error": str(exc)},
        ) from exc
    except Exception as exc:
        details = (
            exc.to_dict()
            if callable(getattr(exc, "to_dict", None))
            else {"exception_type": type(exc).__name__, "error": str(exc)}
        )
        raise PipelineError(
            FailureCode.ASSET_INVALID,
            "name-based collision evaluator could not load configured URDF geometry",
            stage="collision_setup",
            details=details,
        ) from exc


class _DisabledCollisionEvaluator:
    """Explicit collision-disabled path; never used when collision is enabled."""

    def candidate_is_collision_free(
        self,
        candidate: GraspCandidate,
        gripper_world: np.ndarray,
        arm_joint_q: np.ndarray,
    ) -> CheckResult:
        del candidate, gripper_world, arm_joint_q
        return CheckResult(True, details={"collision_check": "disabled_by_config"})

    def trajectory_collision_flags(
        self, plan: TaskPlan, arm_joint_q: np.ndarray
    ) -> np.ndarray:
        del arm_joint_q
        return np.zeros(len(plan.phase_names), dtype=bool)


def _collision_evaluator(
    *,
    config: ProjectConfig,
    kinematics: DoorKinematics,
    backend: Any,
    factory: CollisionFactory | None,
) -> CollisionEvaluator:
    if not _bool_config(config, "collision.enabled"):
        return _DisabledCollisionEvaluator()
    builder = factory or _default_collision_factory
    try:
        return builder(config, kinematics, backend)
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "failed to construct collision evaluator",
            stage="collision_setup",
            details={"exception_type": type(exc).__name__, "error": str(exc)},
        ) from exc


def _phase_samples(config: ProjectConfig) -> dict[str, int]:
    return {
        phase: int(config.get(f"task.phases.{phase}.samples"))
        for phase in PHASE_ORDER
    }


def _actual_pose(waypoint: Any) -> np.ndarray:
    position = np.asarray(getattr(waypoint, "actual_position"), dtype=float)
    orientation = np.asarray(
        getattr(waypoint, "actual_orientation_wxyz"), dtype=float
    )
    if position.shape != (3,) or orientation.shape != (4,):
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "IK waypoint has malformed realized TCP pose",
            stage="ik_result",
            details={
                "position_shape": list(position.shape),
                "orientation_shape": list(orientation.shape),
            },
        )
    if not np.isfinite(position).all() or not np.isfinite(orientation).all():
        return np.full((4, 4), math.nan, dtype=float)
    try:
        return pose_matrix(position, orientation)
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "IK waypoint has an invalid realized TCP rotation",
            stage="ik_result",
            details={"error": str(exc)},
        ) from exc


def _failure_code_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _joint_violation_row(item: Any) -> dict[str, Any]:
    return {
        "dof_index": int(getattr(item, "dof_index")),
        "coord_index": int(getattr(item, "coord_index")),
        "value": _finite_or_none(getattr(item, "value")),
        "lower": _finite_or_none(getattr(item, "lower")),
        "upper": _finite_or_none(getattr(item, "upper")),
        "magnitude": _finite_or_none(getattr(item, "magnitude")),
    }


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _numeric_json(value: Any) -> Any:
    """Replace non-finite diagnostics with JSON ``null`` recursively."""

    array = np.asarray(value)
    if array.ndim == 0:
        return _finite_or_none(array.item())
    if np.issubdtype(array.dtype, np.number):
        converted = array.astype(float).tolist()

        def clean(item: Any) -> Any:
            if isinstance(item, list):
                return [clean(child) for child in item]
            return _finite_or_none(item)

        return clean(converted)
    return array.tolist()


def _rollout_rows(
    *,
    plan: TaskPlan,
    waypoints: Sequence[Any],
    arm_joint_names: Sequence[str],
    arm_joint_q: np.ndarray,
    achieved_gripper_world: np.ndarray,
    collision_flags: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_index, waypoint in enumerate(waypoints):
        validation = getattr(waypoint, "validation")
        pose_error = getattr(validation, "pose_error")
        joint_violations = getattr(validation, "joint_limit_violations", ())
        rows.append(
            {
                "schema_version": 1,
                "frame_index": frame_index,
                "time_s": float(plan.time_s[frame_index]),
                "phase": str(plan.phase_names[frame_index]),
                "phase_index": int(plan.phase_indices[frame_index]),
                "door_angle_rad": float(plan.door_angle_rad[frame_index]),
                "door_angle_deg": math.degrees(float(plan.door_angle_rad[frame_index])),
                "gripper_width_m": float(plan.gripper_width_m[frame_index]),
                "handle_world": _numeric_json(plan.handle_world[frame_index]),
                "target_gripper_world": _numeric_json(
                    plan.target_gripper_world[frame_index]
                ),
                "achieved_gripper_world": _numeric_json(
                    achieved_gripper_world[frame_index]
                ),
                "arm_joint_positions": {
                    name: _finite_or_none(value)
                    for name, value in zip(
                        arm_joint_names, arm_joint_q[frame_index], strict=True
                    )
                },
                "collision": bool(collision_flags[frame_index]),
                "ik": {
                    "success": bool(getattr(validation, "success")),
                    "objective_cost": _finite_or_none(
                        getattr(waypoint, "objective_cost")
                    ),
                    "position_error_m": _finite_or_none(
                        getattr(pose_error, "position_m")
                    ),
                    "orientation_error_deg": _finite_or_none(
                        getattr(pose_error, "orientation_deg")
                    ),
                    "has_nonfinite": bool(getattr(validation, "has_nonfinite")),
                    "failed_checks": list(getattr(validation, "failed_checks", ())),
                    "failure_codes": [
                        _failure_code_value(code)
                        for code in getattr(validation, "failure_codes", ())
                    ],
                    "joint_limit_violations": [
                        _joint_violation_row(item) for item in joint_violations
                    ],
                },
            }
        )
    return rows


def _validate_collision_flags(values: Any, frame_count: int) -> np.ndarray:
    flags = np.asarray(values)
    if flags.shape != (frame_count,):
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "collision evaluator returned an invalid flag count",
            stage="collision_evaluation",
            details={"expected": frame_count, "actual_shape": list(flags.shape)},
        )
    if np.issubdtype(flags.dtype, np.bool_):
        return flags.astype(bool, copy=False)
    try:
        numeric = flags.astype(float)
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "collision flags must be boolean or numeric 0/1",
            stage="collision_evaluation",
        ) from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "collision flags must contain only boolean or numeric 0/1 values",
            stage="collision_evaluation",
        )
    return numeric.astype(bool)


def run_kinematic(
    config: ProjectConfig,
    *,
    output_dir: str | Path | None = None,
    backend_factory: BackendFactory | None = None,
    collision_factory: CollisionFactory | None = None,
) -> RunOutcome:
    """Execute and persist one deterministic five-phase kinematic rollout."""

    _configure_deterministic_runtime(config)
    destination = _output_directory(config, output_dir)
    log = _RunLog(
        mode="kinematic",
        seed=int(config.get("seed")),
        path=destination / "run.log",
    )
    if _bool_config(config, "output.write_resolved_config"):
        write_json(
            destination / "resolved_config.json",
            _resolved_config_payload(config, output_dir=destination),
        )

    object_urdf = config.asset_path("object")
    inspection = inspect_asset(
        object_urdf,
        door_joint_name=str(config.get("assets.object.door_joint")),
        door_link_name=str(config.get("assets.object.door_link")),
        handle_link_name=str(config.get("assets.object.handle_link")),
        require_mesh_files=True,
    )
    write_json(destination / "asset_inspection.json", inspection.to_dict())
    log.add(
        "asset_inspected",
        ok=inspection.ok,
        error_count=len(inspection.errors),
        warning_count=len(inspection.warnings),
    )
    if not inspection.ok:
        raise PipelineError(
            FailureCode.ASSET_INVALID,
            "microwave asset inspection failed",
            stage="asset_inspection",
            details=inspection.to_dict(),
        )

    try:
        object_model = load_urdf(object_urdf)
        kinematics = DoorKinematics(
            model=object_model,
            root_world_transform=_configured_pose(config, "assets.object.world_pose"),
            door_joint_name=str(config.get("assets.object.door_joint")),
            door_link_name=str(config.get("assets.object.door_link")),
            handle_link_name=str(config.get("assets.object.handle_link")),
        )
    except PipelineError:
        raise
    except URDFModelError as exc:
        raise PipelineError(
            FailureCode.ASSET_INVALID,
            "microwave URDF cannot be loaded for forward kinematics",
            stage="object_fk",
            details=exc.to_dict(),
        ) from exc

    backend_type = backend_factory or NewtonFrankaIKBackend
    backend = backend_type(_newton_parameters(config))
    initial_gripper_world = _initial_tcp_transform(backend)
    log.add("ik_backend_ready", backend=type(backend).__name__)

    affordance_settings = CandidateGenerationConfig(
        width_margin_m=float(config.get("affordance_generation.width_margin_m")),
        max_gripper_width_m=float(
            config.get("affordance_generation.max_gripper_width_m")
        ),
        max_candidates=int(config.get("affordance_generation.max_candidates")),
        primitive_radial_samples=int(
            config.get("affordance_generation.primitive_radial_samples")
        ),
    )
    affordance_path = config.resolve_path(str(config.get("assets.object.affordances")))
    try:
        resolution = resolve_handle_candidates(
            affordance_path,
            str(config.get("assets.object.handle_frame")),
            object_urdf,
            str(config.get("assets.object.handle_link")),
            config=affordance_settings,
        )
    except AffordanceError as exc:
        raise PipelineError(
            FailureCode.FRAME_MISSING
            if exc.code in {"FRAME_NOT_FOUND", "FRAME_NAME_INVALID"}
            else FailureCode.ASSET_INVALID,
            "handle affordance resolution failed",
            stage="affordance_resolution",
            details=exc.to_dict(),
        ) from exc

    collision_evaluator = _collision_evaluator(
        config=config,
        kinematics=kinematics,
        backend=backend,
        factory=collision_factory,
    )
    closed_angle_rad = math.radians(float(config.get("task.closed_angle_deg")))
    goal_angle_rad = math.radians(float(config.get("task.goal_angle_deg")))
    handle_to_gripper = _configured_pose(config, "task.grasp_offset")
    candidate_targets: dict[str, np.ndarray] = {}

    def target_for(candidate: GraspCandidate) -> np.ndarray:
        if candidate.candidate_id not in candidate_targets:
            candidate_targets[candidate.candidate_id] = _candidate_target(
                kinematics,
                candidate,
                closed_angle_rad,
                handle_to_gripper,
            )
        return candidate_targets[candidate.candidate_id]

    candidate_joint_q: dict[str, np.ndarray] = {}

    def reachability_check(candidate: GraspCandidate) -> CheckResult:
        check, joint_q = _single_target_is_reachable(
            config=config,
            backend=backend,
            gripper_world=target_for(candidate),
        )
        if joint_q is not None:
            candidate_joint_q[candidate.candidate_id] = joint_q
        return check

    def candidate_collision_check(candidate: GraspCandidate) -> CheckResult | bool | tuple[bool, str]:
        joint_q = candidate_joint_q.get(candidate.candidate_id)
        if joint_q is None:
            return CheckResult(
                False,
                "candidate collision check has no validated IK solution",
                {"candidate_id": candidate.candidate_id},
            )
        return collision_evaluator.candidate_is_collision_free(
            candidate,
            target_for(candidate),
            joint_q,
        )

    selection = select_grasp_candidate(
        resolution.candidates,
        reachability_check=reachability_check,
        collision_free_check=candidate_collision_check,
    )
    candidate_collision_evidence: Any = getattr(
        collision_evaluator, "candidate_reports", None
    )
    if candidate_collision_evidence is None:
        results = getattr(collision_evaluator, "candidate_results", {})
        candidate_collision_evidence = {
            key: value.to_dict() if callable(getattr(value, "to_dict", None)) else value
            for key, value in results.items()
        }
    write_json(
        destination / "affordance_candidates.json",
        {
            "resolution": resolution.to_dict(),
            "selection": selection.to_dict(),
            "collision_evidence": candidate_collision_evidence,
        },
    )
    log.add(
        "grasp_candidates_evaluated",
        candidate_count=len(resolution.candidates),
        selected=(selection.selected.candidate_id if selection.selected else None),
        geometry_fallback=resolution.used_geometry_fallback,
    )
    if selection.selected is None:
        failure_codes = {
            reason.code for reason in selection.failure_reasons
        }
        code = (
            FailureCode.COLLISION
            if failure_codes and failure_codes <= {"COLLISION", "COLLISION_CHECK_ERROR"}
            else FailureCode.IK_UNREACHABLE
        )
        raise PipelineError(
            code,
            "no handle grasp candidate passed reachability and collision checks",
            stage="candidate_selection",
            details=selection.to_dict(),
        )

    selected = selection.selected
    plan = generate_task_plan(
        kinematics=kinematics,
        link_to_handle_frame=selected.transform,
        handle_approach_axis=np.asarray(selected.approach_axis, dtype=float),
        handle_to_gripper=handle_to_gripper,
        initial_gripper_world=initial_gripper_world,
        closed_angle_rad=closed_angle_rad,
        goal_angle_rad=goal_angle_rad,
        phase_samples=_phase_samples(config),
        pregrasp_distance_m=float(config.get("task.pregrasp_distance_m")),
        retreat_distance_m=float(config.get("task.retreat_distance_m")),
        open_gripper_width_m=float(config.get("assets.robot.open_gripper_width_m")),
        closed_gripper_width_m=float(
            config.get("assets.robot.closed_gripper_width_m")
        ),
        dt=float(config.get("simulation.dt")),
    )
    log.add(
        "task_plan_generated",
        frame_count=len(plan.phase_names),
        phase_counts=_phase_samples(config),
    )

    target_positions: list[np.ndarray] = []
    target_orientations: list[np.ndarray] = []
    for transform in plan.target_gripper_world:
        position, orientation = decompose_pose(transform)
        target_positions.append(position)
        target_orientations.append(orientation)
    try:
        result = backend.solve_waypoints(target_positions, target_orientations)
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "Newton IK trajectory solve failed",
            stage="ik_solve",
            details={"exception_type": type(exc).__name__, "error": str(exc)},
        ) from exc
    waypoints = tuple(getattr(result, "waypoints", ()))
    if len(waypoints) != len(plan.phase_names):
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "IK backend returned a different number of waypoints than requested",
            stage="ik_result",
            details={"expected": len(plan.phase_names), "actual": len(waypoints)},
        )

    arm_joint_names = tuple(str(name) for name in config.get("assets.robot.arm_joint_names"))
    joint_rows = [
        tuple(float(value) for value in getattr(waypoint, "arm_joint_positions"))
        for waypoint in waypoints
    ]
    arm_joint_q = np.asarray(joint_rows, dtype=float)
    expected_joint_shape = (len(waypoints), len(arm_joint_names))
    if arm_joint_q.shape != expected_joint_shape:
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "IK backend arm joint result shape does not match configured joint names",
            stage="ik_result",
            details={
                "expected": list(expected_joint_shape),
                "actual": list(arm_joint_q.shape),
                "joint_names": list(arm_joint_names),
            },
        )
    achieved = np.stack([_actual_pose(waypoint) for waypoint in waypoints])
    ik_success = np.asarray(
        [bool(getattr(waypoint.validation, "success")) for waypoint in waypoints],
        dtype=bool,
    )
    try:
        raw_collision_flags = collision_evaluator.trajectory_collision_flags(
            plan, arm_joint_q
        )
    except PipelineError:
        raise
    except Exception as exc:
        details = (
            exc.to_dict()
            if callable(getattr(exc, "to_dict", None))
            else {"exception_type": type(exc).__name__, "error": str(exc)}
        )
        raise PipelineError(
            FailureCode.COLLISION,
            "trajectory collision evaluation failed",
            stage="collision_evaluation",
            details=details,
        ) from exc
    collision_flags = _validate_collision_flags(
        raw_collision_flags, len(plan.phase_names)
    )
    collision_report = getattr(collision_evaluator, "last_report", None)
    if collision_report is None:
        collision_report = getattr(
            collision_evaluator, "last_trajectory_report", None
        )
    collision_report_payload = (
        collision_report.to_dict()
        if callable(getattr(collision_report, "to_dict", None))
        else {
            "backend": type(collision_evaluator).__name__,
            "frame_count": len(plan.phase_names),
            "flags": collision_flags.tolist(),
            "evidence_scope": "injected_or_disabled_evaluator",
        }
    )
    write_json(
        destination / "collision_report.json",
        {
            "candidate_checks": candidate_collision_evidence,
            "trajectory": collision_report_payload,
        },
    )
    joint_lower, joint_upper = _arm_joint_limits(config)
    try:
        metrics = compute_metrics(
            phase_names=plan.phase_names,
            door_angle_rad=plan.door_angle_rad,
            handle_world=plan.handle_world,
            target_gripper_world=plan.target_gripper_world,
            achieved_gripper_world=achieved,
            joint_q=arm_joint_q,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            collision_flags=collision_flags,
            ik_success_flags=ik_success,
            target_door_angle_rad=goal_angle_rad,
            thresholds=MetricThresholds.from_mapping(config.get("thresholds")),
            joint_limit_tolerance_rad=float(
                config.get("ik.joint_limit_tolerance_rad")
            ),
        )
    except MetricsInputError as exc:
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "rollout metrics could not be computed",
            stage="metrics",
            details=exc.to_dict(),
        ) from exc

    run_status = "success" if bool(metrics["success"]) else "acceptance_failed"
    metrics = {
        **metrics,
        "mode": "kinematic",
        "collision_scope": str(config.get("collision.scope")),
        "run_status": run_status,
        "seed": int(config.get("seed")),
        "selected_grasp_candidate": selected.to_dict(),
    }
    arrays = {
        **plan.as_arrays(),
        "arm_joint_q": arm_joint_q,
        "joint_lower": joint_lower,
        "joint_upper": joint_upper,
        "achieved_gripper_world": achieved,
        "ik_success_flags": ik_success,
        "collision_flags": collision_flags,
        "objective_cost": np.asarray(
            [float(getattr(waypoint, "objective_cost")) for waypoint in waypoints],
            dtype=float,
        ),
    }
    write_trajectory(destination / "trajectory.npz", arrays)
    if _bool_config(config, "output.write_rollout_jsonl"):
        write_jsonl(
            destination / "rollout.jsonl",
            _rollout_rows(
                plan=plan,
                waypoints=waypoints,
                arm_joint_names=arm_joint_names,
                arm_joint_q=arm_joint_q,
                achieved_gripper_world=achieved,
                collision_flags=collision_flags,
            ),
        )
    write_json(destination / "metrics.json", metrics)
    log.add(
        "run_completed",
        status=run_status,
        acceptance_passed=bool(metrics["success"]),
        ik_success_rate=float(metrics["ik_waypoint_success_rate"]),
    )
    write_jsonl(destination / "run.log", log.rows)
    return RunOutcome(
        output_dir=destination,
        metrics=metrics,
        exit_code=0 if bool(metrics["success"]) else 3,
    )


def _write_failure(
    destination: Path,
    *,
    mode: str,
    failure: Mapping[str, Any],
) -> None:
    payload = {
        "success": False,
        "mode": mode,
        "run_status": "failed",
        "failure": dict(failure),
    }
    write_json(destination / "metrics.json", payload)
    log_path = destination / "run.log"
    rows: list[dict[str, Any]] = []
    if log_path.is_file():
        try:
            rows = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError):
            rows = []
    rows.append(
        {
            "event_index": len(rows),
            "event": "run_failed",
            "details": {"mode": mode, **dict(failure)},
        }
    )
    write_jsonl(log_path, rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("kinematic", "physics_assisted"),
        default="kinematic",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config: ProjectConfig | None = None
    destination = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else None
    )
    try:
        config = load_config(args.config)
        destination = _output_directory(config, destination, mode=args.mode)
        if args.mode == "kinematic":
            outcome = run_kinematic(config, output_dir=destination)
        else:
            try:
                from .physics_assisted import run_physics_assisted
            except (ImportError, AttributeError) as exc:
                raise PipelineError(
                    FailureCode.PHYSICS_UNAVAILABLE,
                    "physics-assisted runner is unavailable",
                    stage="physics_setup",
                    details={
                        "exception_type": type(exc).__name__,
                        "error": str(exc),
                    },
                ) from exc
            outcome = run_physics_assisted(config, output_dir=destination)
        print(
            json.dumps(
                {
                    "success": bool(outcome.metrics["success"]),
                    "mode": args.mode,
                    "output_dir": str(outcome.output_dir),
                    "exit_code": outcome.exit_code,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return int(outcome.exit_code)
    except PipelineError as exc:
        failure = exc.to_dict()
        if destination is not None:
            try:
                _write_failure(destination, mode=args.mode, failure=failure)
            except PipelineError:
                pass
        print(
            json.dumps(
                {
                    "success": False,
                    "mode": args.mode,
                    "output_dir": str(destination) if destination is not None else None,
                    "exit_code": 2,
                    "failure": failure,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # Defensive CLI boundary; never hide the exception type.
        failure = {
            "code": "unexpected_error",
            "stage": "cli",
            "message": str(exc),
            "details": {"exception_type": type(exc).__name__},
        }
        if destination is not None:
            try:
                _write_failure(destination, mode=args.mode, failure=failure)
            except PipelineError:
                pass
        print(
            json.dumps(
                {
                    "success": False,
                    "mode": args.mode,
                    "output_dir": str(destination) if destination is not None else None,
                    "exit_code": 2,
                    "failure": failure,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
