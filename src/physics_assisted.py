"""Newton 1.3 CPU physics-assisted execution for the microwave task.

The robot remains fully dynamic and tracks the name-resolved IK trajectory with
Newton XPBD joint position/velocity PD targets.  Runtime target writes are
scattered only to configured robot coordinate/DoF indices.  The microwave door
is never assigned a runtime coordinate, velocity, target, or generalized force.

Grasp assistance is a pre-authored, initially-disabled Newton fixed loop joint.
Its anchors encode the *planned* handle-frame-to-TCP relationship.  Before the
joint can be enabled, that planned relationship must already agree with the
current measured handle/TCP state, preventing a remote-latch constraint.

Project-facing poses use ``[x, y, z, qw, qx, qy, qz]``.  Newton-facing poses
use Warp's ``[x, y, z, qx, qy, qz, qw]`` convention only inside this module.
The validation and coordinate-mapping helpers remain importable on machines
without Newton/Warp so unit tests do not require the remote simulation
environment.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .config import PHASE_ORDER, ProjectConfig, load_config
from .door_kinematics import forward_kinematics
from .errors import FailureCode, PipelineError
from .metrics import MetricThresholds, MetricsInputError, compute_metrics
from .newton_backend import (
    quaternion_wxyz_to_xyzw,
    resolve_unique_label,
)
from .output import write_json, write_jsonl, write_trajectory
from .trajectory import TaskPlan
from .transforms import (
    compose_transforms,
    decompose_pose,
    invert_transform,
    normalize_quaternion,
    pose_matrix,
    transform_point,
)
from .urdf_model import URDFModel, URDFModelError, load_urdf


try:  # Newton is installed only in the pinned remote environment.
    import newton
    import warp as wp
except (ImportError, OSError) as exc:
    newton = None  # type: ignore[assignment]
    wp = None  # type: ignore[assignment]
    _NEWTON_IMPORT_ERROR: BaseException | None = exc
else:
    _NEWTON_IMPORT_ERROR = None


if wp is not None:

    @wp.kernel
    def _scatter_indexed_joint_pd_targets(
        coord_indices: wp.array(dtype=wp.int32),
        dof_indices: wp.array(dtype=wp.int32),
        position_targets: wp.array(dtype=wp.float32),
        velocity_targets: wp.array(dtype=wp.float32),
        destination_position_targets: wp.array(dtype=wp.float32),
        destination_velocity_targets: wp.array(dtype=wp.float32),
    ):
        controlled_joint = wp.tid()
        destination_position_targets[coord_indices[controlled_joint]] = position_targets[
            controlled_joint
        ]
        destination_velocity_targets[dof_indices[controlled_joint]] = velocity_targets[
            controlled_joint
        ]

else:
    _scatter_indexed_joint_pd_targets = None


_SUPPORTED_NEWTON_VERSION_PREFIX = "1.3."
_GRASP_JOINT_LABEL = "agentpre_fixed_grasp"
_STATE_SAMPLE_TIMING = "post_step_end_of_frame"


def _finite_vector(values: Iterable[float], size: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {size} numeric values") from exc
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    result = float(value)
    valid = math.isfinite(result) and (result >= 0.0 if allow_zero else result > 0.0)
    if not valid:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {relation}")
    return result


def _positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalize_newton_broad_phase(value: Any) -> str:
    """Return the canonical Newton broad-phase name or fail closed."""

    if not isinstance(value, str):
        raise ValueError("collision_broad_phase must be a string")
    normalized = value.strip().lower()
    if normalized not in {"sap", "nxn", "explicit"}:
        raise ValueError(
            "unsupported Newton collision broad phase; expected sap, nxn, or explicit"
        )
    return normalized


def _post_step_state_sample_times(frame_count: int, dt: float) -> np.ndarray:
    """Timestamp post-step state samples at each frame's right endpoint."""

    count = _positive_int(frame_count, "frame_count")
    step = _positive(dt, "dt")
    return (np.arange(count, dtype=float) + 1.0) * step


def _physics_safety_audit_metadata() -> dict[str, Any]:
    """Describe state sampling and the grasp contact-filtering policy."""

    return {
        "state_sample_timing": _STATE_SAMPLE_TIMING,
        "grasp_parent_child_collision_filtered": False,
    }


@dataclass(frozen=True, slots=True)
class ScalarJointRef:
    """Name-resolved scalar Newton joint coordinate/DoF mapping."""

    configured_name: str
    label: str
    joint_index: int
    coord_index: int
    dof_index: int


def plan_massless_fixed_joint_collapse(
    joint_labels: Sequence[str],
    joint_types: Sequence[int],
    joint_parents: Sequence[int],
    joint_children: Sequence[int],
    body_masses: Sequence[float],
    *,
    fixed_joint_type: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Select only non-root fixed joints whose child body is massless.

    Newton XPBD treats a zero-inverse-mass body as static.  Retaining such a
    body behind a fixed joint can therefore pin the downstream articulation.
    All other fixed joints are explicitly kept so named frames such as
    ``panda_hand`` survive the selective collapse.
    """

    count = len(joint_labels)
    if not (
        len(joint_types) == count
        and len(joint_parents) == count
        and len(joint_children) == count
    ):
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "Newton joint arrays disagree while planning fixed-joint collapse",
            stage="physics_model",
        )
    masses = np.asarray(body_masses, dtype=float)
    if masses.ndim != 1 or not np.isfinite(masses).all():
        raise PipelineError(
            FailureCode.ASSET_INVALID,
            "Newton robot body masses are malformed or non-finite",
            stage="physics_model",
        )
    collapse: list[str] = []
    keep: list[str] = []
    for label, joint_type, parent, child in zip(
        joint_labels,
        joint_types,
        joint_parents,
        joint_children,
        strict=True,
    ):
        if int(joint_type) != int(fixed_joint_type):
            continue
        child_index = int(child)
        if not 0 <= child_index < len(masses):
            raise PipelineError(
                FailureCode.PHYSICS_UNAVAILABLE,
                "Newton fixed joint references an invalid child body",
                stage="physics_model",
                details={"joint": str(label), "child_index": child_index},
            )
        if int(parent) >= 0 and float(masses[child_index]) <= 0.0:
            collapse.append(str(label))
        else:
            keep.append(str(label))
    return tuple(collapse), tuple(keep)


def resolve_named_scalar_joints(
    labels: Sequence[str],
    joint_q_start: Sequence[int] | np.ndarray,
    joint_qd_start: Sequence[int] | np.ndarray,
    joint_dof_dim: Sequence[Sequence[int]] | np.ndarray,
    configured_names: Sequence[str],
    *,
    joint_coord_count: int | None = None,
    joint_dof_count: int | None = None,
    stage: str = "physics_model",
) -> tuple[ScalarJointRef, ...]:
    """Resolve named joints and reject non-scalar or aliased mappings.

    Newton URDF import may prefix labels, so exact or unique ``/...`` suffix
    matching is used.  No caller relies on import order or hard-coded indices.
    """

    q_start = np.asarray(joint_q_start, dtype=np.int64)
    qd_start = np.asarray(joint_qd_start, dtype=np.int64)
    dof_dim = np.asarray(joint_dof_dim, dtype=np.int64)
    joint_count = len(labels)
    if q_start.shape == (joint_count,) and joint_coord_count is not None:
        q_start = np.concatenate((q_start, np.asarray([joint_coord_count], dtype=np.int64)))
    if qd_start.shape == (joint_count,) and joint_dof_count is not None:
        qd_start = np.concatenate((qd_start, np.asarray([joint_dof_count], dtype=np.int64)))
    if q_start.shape != (joint_count + 1,) or qd_start.shape != (joint_count + 1,):
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "Newton joint start arrays do not match joint labels",
            stage=stage,
            details={
                "joint_count": joint_count,
                "joint_q_start_shape": list(q_start.shape),
                "joint_qd_start_shape": list(qd_start.shape),
            },
        )
    if dof_dim.shape != (joint_count, 2):
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "Newton joint DoF dimensions do not match joint labels",
            stage=stage,
            details={"joint_count": joint_count, "joint_dof_dim_shape": list(dof_dim.shape)},
        )

    refs: list[ScalarJointRef] = []
    used_joint_indices: set[int] = set()
    used_coord_indices: set[int] = set()
    used_dof_indices: set[int] = set()
    for configured_name in configured_names:
        resolved = resolve_unique_label(labels, configured_name, kind="joint")
        joint_index = resolved.index
        coordinate_count = int(q_start[joint_index + 1] - q_start[joint_index])
        dof_count = int(qd_start[joint_index + 1] - qd_start[joint_index])
        dimension_count = int(dof_dim[joint_index, 0] + dof_dim[joint_index, 1])
        if coordinate_count != 1 or dof_count != 1 or dimension_count != 1:
            raise PipelineError(
                FailureCode.CONFIG_INVALID,
                f"physics control joint is not scalar: {configured_name}",
                stage=stage,
                details={
                    "configured_name": configured_name,
                    "resolved_label": resolved.label,
                    "coordinate_count": coordinate_count,
                    "dof_count": dof_count,
                    "dimension_count": dimension_count,
                },
            )
        coord_index = int(q_start[joint_index])
        dof_index = int(qd_start[joint_index])
        if (
            joint_index in used_joint_indices
            or coord_index in used_coord_indices
            or dof_index in used_dof_indices
        ):
            raise PipelineError(
                FailureCode.NAME_NOT_UNIQUE,
                "configured physics joint names alias one Newton joint/coordinate",
                stage=stage,
                details={"configured_name": configured_name, "resolved_label": resolved.label},
            )
        used_joint_indices.add(joint_index)
        used_coord_indices.add(coord_index)
        used_dof_indices.add(dof_index)
        refs.append(
            ScalarJointRef(
                configured_name=str(configured_name),
                label=resolved.label,
                joint_index=joint_index,
                coord_index=coord_index,
                dof_index=dof_index,
            )
        )
    return tuple(refs)


@dataclass(frozen=True, slots=True)
class GraspActivationWindow:
    """Half-open frame interval during which the fixed grasp is enabled."""

    activation_frame: int
    release_frame: int

    def is_active(self, frame_index: int) -> bool:
        return self.activation_frame <= int(frame_index) < self.release_frame


def _anchor_pose_error(first_world: np.ndarray, second_world: np.ndarray) -> tuple[float, float]:
    first_position, _ = decompose_pose(first_world)
    second_position, _ = decompose_pose(second_world)
    position_error = float(np.linalg.norm(first_position - second_position))
    relative_rotation = first_world[:3, :3].T @ second_world[:3, :3]
    cosine = float(
        np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
    )
    return position_error, math.degrees(math.acos(cosine))


@dataclass(frozen=True, slots=True)
class PlannedFixedGraspAnchors:
    """Loop-joint anchors derived only from the planned handle/TCP relation."""

    parent_xform: np.ndarray
    child_xform: np.ndarray

    def __post_init__(self) -> None:
        for name in ("parent_xform", "child_xform"):
            value = np.asarray(getattr(self, name), dtype=float)
            decompose_pose(value)
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class FixedGraspActivationGate:
    """Measured agreement with the planned fixed-grasp anchors at activation."""

    frame_index: int
    position_error_m: float
    orientation_error_deg: float
    position_limit_m: float
    orientation_limit_deg: float

    def __post_init__(self) -> None:
        if not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        for name in (
            "position_error_m",
            "orientation_error_deg",
            "position_limit_m",
            "orientation_limit_deg",
        ):
            _positive(
                getattr(self, name),
                name,
                allow_zero=name in {"position_error_m", "orientation_error_deg"},
            )

    @property
    def passed(self) -> bool:
        return (
            self.position_error_m <= self.position_limit_m
            and self.orientation_error_deg <= self.orientation_limit_deg
        )

    def audit_metadata(self) -> dict[str, Any]:
        return {
            "grasp_anchor_source": "planned_handle_frame_to_tcp_relation",
            "grasp_anchor_runtime_model_write_count": 0,
            "grasp_activation_gate_frame": self.frame_index,
            "grasp_activation_gate_position_error_m": self.position_error_m,
            "grasp_activation_gate_orientation_error_deg": self.orientation_error_deg,
            "grasp_activation_gate_position_limit_m": self.position_limit_m,
            "grasp_activation_gate_orientation_limit_deg": self.orientation_limit_deg,
            "grasp_activation_gate_passed": self.passed,
            "remote_latch_allowed": False,
        }


def planned_fixed_grasp_anchors(
    hand_to_tcp: np.ndarray,
    handle_link_to_handle_frame: np.ndarray,
    expected_handle_frame_to_tcp: np.ndarray,
) -> PlannedFixedGraspAnchors:
    """Build coincident anchors for the *planned* handle-frame/TCP relation.

    The fixed joint joins the hand body (parent) to the handle-link body
    (child), while the authored relationship is between the TCP and a named
    handle frame.  The parent anchor therefore maps ``hand -> TCP -> handle``;
    the child anchor is exactly the configured ``handle_link -> handle`` frame.
    No measured pose participates in this construction.
    """

    try:
        hand_tcp = np.asarray(hand_to_tcp, dtype=float)
        link_handle = np.asarray(handle_link_to_handle_frame, dtype=float)
        handle_tcp = np.asarray(expected_handle_frame_to_tcp, dtype=float)
        decompose_pose(hand_tcp)
        decompose_pose(link_handle)
        decompose_pose(handle_tcp)
        return PlannedFixedGraspAnchors(
            parent_xform=compose_transforms(hand_tcp, invert_transform(handle_tcp)),
            child_xform=link_handle,
        )
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "planned fixed-grasp transforms are malformed",
            stage="physics_constraint",
            details={"error": str(exc)},
        ) from exc


def evaluate_fixed_grasp_activation_gate(
    parent_world: np.ndarray,
    child_world: np.ndarray,
    anchors: PlannedFixedGraspAnchors,
    *,
    frame_index: int,
    position_limit_m: float,
    orientation_limit_deg: float,
) -> FixedGraspActivationGate:
    """Compare current measured anchors to the planned relationship."""

    try:
        parent_anchor_world = compose_transforms(parent_world, anchors.parent_xform)
        child_anchor_world = compose_transforms(child_world, anchors.child_xform)
        position_error, orientation_error = _anchor_pose_error(
            parent_anchor_world, child_anchor_world
        )
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "cannot evaluate the fixed-grasp activation gate",
            stage="physics_constraint",
            details={"frame_index": int(frame_index), "error": str(exc)},
        ) from exc
    return FixedGraspActivationGate(
        frame_index=int(frame_index),
        position_error_m=position_error,
        orientation_error_deg=orientation_error,
        position_limit_m=float(position_limit_m),
        orientation_limit_deg=float(orientation_limit_deg),
    )


def fixed_grasp_activation_window(
    phase_names: Sequence[str] | np.ndarray,
    activate_after_phase: str,
) -> GraspActivationWindow:
    """Return activation-after-phase and release-at-retreat frame indices."""

    phases = tuple(str(value) for value in np.asarray(phase_names).tolist())
    if not phases:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "physics command trajectory is empty",
            stage="physics_trajectory",
        )
    if activate_after_phase not in PHASE_ORDER:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "fixed grasp activation phase is not a known task phase",
            stage="physics_trajectory",
            details={"activate_after_phase": activate_after_phase, "phase_order": list(PHASE_ORDER)},
        )
    matching = [index for index, phase in enumerate(phases) if phase == activate_after_phase]
    if not matching:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "fixed grasp activation phase is absent from the command trajectory",
            stage="physics_trajectory",
            details={"activate_after_phase": activate_after_phase, "phases": list(phases)},
        )
    activation = max(matching) + 1
    if activation >= len(phases):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "fixed grasp cannot activate after the final trajectory frame",
            stage="physics_trajectory",
            details={"activate_after_phase": activate_after_phase, "frame_count": len(phases)},
        )
    retreat_indices = [
        index for index, phase in enumerate(phases) if index >= activation and phase == "retreat"
    ]
    release = retreat_indices[0] if retreat_indices else len(phases)
    return GraspActivationWindow(activation_frame=activation, release_frame=release)


@dataclass(frozen=True, slots=True)
class PhysicsCommandTrajectory:
    """Robot commands consumed by physics; door references are diagnostic only."""

    phase_names: np.ndarray
    arm_joint_targets: np.ndarray
    gripper_width_m: np.ndarray
    door_reference_rad: np.ndarray | None = None
    handle_frame_in_link: np.ndarray | None = None
    expected_handle_to_tcp: np.ndarray | None = None

    def __post_init__(self) -> None:
        phases = np.asarray(self.phase_names, dtype="U16")
        arm = np.asarray(self.arm_joint_targets, dtype=float)
        width = np.asarray(self.gripper_width_m, dtype=float)
        if phases.ndim != 1 or len(phases) == 0:
            raise PipelineError(
                FailureCode.CONFIG_INVALID,
                "phase_names must be a non-empty one-dimensional array",
                stage="physics_trajectory",
                details={"shape": list(phases.shape)},
            )
        if arm.ndim != 2 or arm.shape[0] != len(phases) or arm.shape[1] < 1:
            raise PipelineError(
                FailureCode.CONFIG_INVALID,
                "arm_joint_targets must have shape (frames, arm_joints)",
                stage="physics_trajectory",
                details={"phase_count": len(phases), "shape": list(arm.shape)},
            )
        if width.shape != (len(phases),):
            raise PipelineError(
                FailureCode.CONFIG_INVALID,
                "gripper_width_m must have one value per frame",
                stage="physics_trajectory",
                details={"phase_count": len(phases), "shape": list(width.shape)},
            )
        unknown = sorted(set(phases.tolist()).difference(PHASE_ORDER))
        if unknown:
            raise PipelineError(
                FailureCode.CONFIG_INVALID,
                "physics trajectory contains unknown task phases",
                stage="physics_trajectory",
                details={"unknown_phases": unknown, "phase_order": list(PHASE_ORDER)},
            )
        if not np.isfinite(arm).all() or not np.isfinite(width).all() or np.any(width < 0.0):
            raise PipelineError(
                FailureCode.NUMERICAL_INSTABILITY,
                "physics robot commands must be finite and gripper widths non-negative",
                stage="physics_trajectory",
            )

        door: np.ndarray | None = None
        if self.door_reference_rad is not None:
            door = np.asarray(self.door_reference_rad, dtype=float)
            if door.shape != (len(phases),) or not np.isfinite(door).all():
                raise PipelineError(
                    FailureCode.CONFIG_INVALID,
                    "door_reference_rad must contain one finite diagnostic value per frame",
                    stage="physics_trajectory",
                    details={"phase_count": len(phases), "shape": list(door.shape)},
                )

        handle_frame: np.ndarray | None = None
        expected_relation: np.ndarray | None = None
        if (self.handle_frame_in_link is None) != (
            self.expected_handle_to_tcp is None
        ):
            raise PipelineError(
                FailureCode.CONFIG_INVALID,
                "handle_frame_in_link and expected_handle_to_tcp must be supplied together",
                stage="physics_trajectory",
            )
        if self.handle_frame_in_link is not None:
            try:
                handle_frame = np.asarray(self.handle_frame_in_link, dtype=float)
                expected_relation = np.asarray(
                    self.expected_handle_to_tcp, dtype=float
                )
                decompose_pose(handle_frame)
                decompose_pose(expected_relation)
            except (TypeError, ValueError) as exc:
                raise PipelineError(
                    FailureCode.CONFIG_INVALID,
                    "fixed-grasp command transforms must be finite rigid transforms",
                    stage="physics_trajectory",
                    details={"error": str(exc)},
                ) from exc

        phases = phases.copy()
        arm = arm.copy()
        width = width.copy()
        phases.setflags(write=False)
        arm.setflags(write=False)
        width.setflags(write=False)
        if door is not None:
            door = door.copy()
            door.setflags(write=False)
        if handle_frame is not None and expected_relation is not None:
            handle_frame = handle_frame.copy()
            expected_relation = expected_relation.copy()
            handle_frame.setflags(write=False)
            expected_relation.setflags(write=False)
        object.__setattr__(self, "phase_names", phases)
        object.__setattr__(self, "arm_joint_targets", arm)
        object.__setattr__(self, "gripper_width_m", width)
        object.__setattr__(self, "door_reference_rad", door)
        object.__setattr__(self, "handle_frame_in_link", handle_frame)
        object.__setattr__(self, "expected_handle_to_tcp", expected_relation)

    @property
    def frame_count(self) -> int:
        return int(len(self.phase_names))


def build_robot_position_targets(
    base_joint_q: Sequence[float] | np.ndarray,
    arm_refs: Sequence[ScalarJointRef],
    finger_refs: Sequence[ScalarJointRef],
    arm_targets: Sequence[float] | np.ndarray,
    gripper_width_m: float,
    *,
    protected_coord_indices: Sequence[int] = (),
) -> np.ndarray:
    """Build a full coordinate vector while changing only named robot joints."""

    base = np.asarray(base_joint_q, dtype=float)
    desired_arm = np.asarray(arm_targets, dtype=float)
    if base.ndim != 1:
        raise ValueError("base_joint_q must be one-dimensional")
    if desired_arm.shape != (len(arm_refs),) or not np.isfinite(desired_arm).all():
        raise ValueError("arm_targets must be finite and match arm_refs")
    width = _positive(gripper_width_m, "gripper_width_m", allow_zero=True)
    target = base.copy()
    protected_before = {int(index): float(base[int(index)]) for index in protected_coord_indices}
    for ref, value in zip(arm_refs, desired_arm, strict=True):
        target[ref.coord_index] = float(value)
    finger_target = 0.5 * width
    for ref in finger_refs:
        target[ref.coord_index] = finger_target
    for coord_index, value in protected_before.items():
        if target[coord_index] != value:
            raise PipelineError(
                FailureCode.CONFIG_INVALID,
                "robot target mapping overlaps a protected object coordinate",
                stage="physics_control",
                details={"protected_coord_index": coord_index},
            )
    return target


@dataclass(frozen=True, slots=True)
class PhysicsParameters:
    """All model, joint-PD, collision, and stepping inputs from config."""

    robot_urdf: Path
    object_urdf: Path
    robot_world_position: tuple[float, float, float]
    robot_world_orientation_wxyz: tuple[float, float, float, float]
    object_world_position: tuple[float, float, float]
    object_world_orientation_wxyz: tuple[float, float, float, float]
    arm_joint_names: tuple[str, ...]
    finger_joint_names: tuple[str, ...]
    nominal_arm_joint_positions: tuple[float, ...]
    open_gripper_width_m: float
    end_effector_link: str
    end_effector_offset_position: tuple[float, float, float]
    end_effector_offset_orientation_wxyz: tuple[float, float, float, float]
    door_joint: str
    handle_link: str
    closed_angle_rad: float
    dt: float
    substeps: int
    gravity_m_s2: tuple[float, float, float]
    solver: str
    solver_iterations: int
    robot_control_backend: str
    robot_control_implementation: str
    target_velocity_mode: str
    arm_stiffness: float
    arm_damping: float
    finger_stiffness: float
    finger_damping: float
    door_control_backend: str
    door_target_stiffness: float
    door_target_damping: float
    door_target_velocity_rad_s: float
    collision_enabled: bool
    collision_broad_phase: str
    collision_deterministic: bool
    collision_margin_m: float
    allowed_contact_links: tuple[str, ...]
    fixed_grasp_enabled: bool
    fixed_grasp_activate_after_phase: str
    grasp_activation_position_tolerance_m: float
    grasp_activation_orientation_tolerance_deg: float
    device: str

    def __post_init__(self) -> None:
        _finite_vector(self.robot_world_position, 3, "robot_world_position")
        _finite_vector(self.object_world_position, 3, "object_world_position")
        normalize_quaternion(self.robot_world_orientation_wxyz)
        normalize_quaternion(self.object_world_orientation_wxyz)
        _finite_vector(self.end_effector_offset_position, 3, "end_effector_offset_position")
        normalize_quaternion(self.end_effector_offset_orientation_wxyz)
        _finite_vector(self.gravity_m_s2, 3, "gravity_m_s2")
        if not self.arm_joint_names or len(self.arm_joint_names) != len(self.nominal_arm_joint_positions):
            raise ValueError("arm joint names and nominal positions must be non-empty and equal length")
        all_names = (*self.arm_joint_names, *self.finger_joint_names)
        if any(not name.strip() for name in all_names) or len(set(all_names)) != len(all_names):
            raise ValueError("controlled robot joint names must be non-empty and unique")
        if not np.isfinite(np.asarray(self.nominal_arm_joint_positions, dtype=float)).all():
            raise ValueError("nominal arm joint positions must be finite")
        for name in ("end_effector_link", "door_joint", "handle_link"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        _positive(self.open_gripper_width_m, "open_gripper_width_m")
        _positive(self.dt, "dt")
        _positive_int(self.substeps, "substeps")
        _positive_int(self.solver_iterations, "solver_iterations")
        for name in (
            "arm_stiffness",
            "arm_damping",
            "finger_stiffness",
            "finger_damping",
            "grasp_activation_position_tolerance_m",
            "grasp_activation_orientation_tolerance_deg",
        ):
            _positive(getattr(self, name), name)
        door_stiffness = _positive(
            self.door_target_stiffness,
            "door_target_stiffness",
            allow_zero=True,
        )
        if door_stiffness != 0.0:
            raise ValueError("door_target_stiffness must be zero")
        _positive(self.door_target_damping, "door_target_damping")
        if not math.isfinite(self.door_target_velocity_rad_s) or (
            self.door_target_velocity_rad_s != 0.0
        ):
            raise ValueError("door_target_velocity_rad_s must be finite and zero")
        _positive(self.collision_margin_m, "collision_margin_m", allow_zero=True)
        if self.device.lower() != "cpu":
            raise ValueError("physics-assisted first stage requires explicit CPU")
        if self.solver.lower() != "xpbd":
            raise ValueError("physics-assisted backend currently requires solver='xpbd'")
        if self.robot_control_backend.lower() != "joint_pd":
            raise ValueError(
                "physics-assisted backend requires robot_control_backend='joint_pd'"
            )
        if self.robot_control_implementation.lower() != "newton_xpbd_joint_targets":
            raise ValueError(
                "physics-assisted backend requires newton_xpbd_joint_targets"
            )
        if self.target_velocity_mode.lower() != "finite_difference":
            raise ValueError("physics-assisted joint PD requires finite-difference targets")
        if self.door_control_backend.lower() != "passive_velocity_damping":
            raise ValueError("physics-assisted door requires passive velocity damping")
        object.__setattr__(
            self,
            "collision_broad_phase",
            _normalize_newton_broad_phase(self.collision_broad_phase),
        )
        if not isinstance(self.collision_enabled, bool) or not isinstance(self.collision_deterministic, bool):
            raise ValueError("collision flags must be booleans")
        if not isinstance(self.fixed_grasp_enabled, bool):
            raise ValueError("fixed_grasp_enabled must be boolean")
        if (
            not self.allowed_contact_links
            or any(not name.strip() for name in self.allowed_contact_links)
            or len(set(self.allowed_contact_links)) != len(self.allowed_contact_links)
        ):
            raise ValueError("allowed_contact_links must contain unique non-empty names")
        if self.fixed_grasp_activate_after_phase not in PHASE_ORDER:
            raise ValueError("fixed grasp activation phase is unknown")

    @classmethod
    def from_project_config(cls, config: Any) -> "PhysicsParameters":
        robot_pose = config.get("assets.robot.world_pose")
        object_pose = config.get("assets.object.world_pose")
        ee_offset = config.get("assets.robot.end_effector_offset")
        fixed = config.get("simulation.fixed_grasp_constraint")
        control = config.get("simulation.robot_control")
        door_control = config.get("simulation.door_control")
        return cls(
            robot_urdf=config.asset_path("robot"),
            object_urdf=config.asset_path("object"),
            robot_world_position=tuple(float(v) for v in robot_pose["position"]),
            robot_world_orientation_wxyz=tuple(float(v) for v in robot_pose["orientation_wxyz"]),
            object_world_position=tuple(float(v) for v in object_pose["position"]),
            object_world_orientation_wxyz=tuple(float(v) for v in object_pose["orientation_wxyz"]),
            arm_joint_names=tuple(str(v) for v in config.get("assets.robot.arm_joint_names")),
            finger_joint_names=tuple(str(v) for v in config.get("assets.robot.finger_joint_names")),
            nominal_arm_joint_positions=tuple(
                float(v) for v in config.get("assets.robot.default_joint_positions")
            ),
            open_gripper_width_m=float(config.get("assets.robot.open_gripper_width_m")),
            end_effector_link=str(config.get("assets.robot.end_effector_link")),
            end_effector_offset_position=tuple(float(v) for v in ee_offset["position"]),
            end_effector_offset_orientation_wxyz=tuple(
                float(v) for v in ee_offset["orientation_wxyz"]
            ),
            door_joint=str(config.get("assets.object.door_joint")),
            handle_link=str(config.get("assets.object.handle_link")),
            closed_angle_rad=math.radians(float(config.get("task.closed_angle_deg"))),
            dt=float(config.get("simulation.dt")),
            substeps=int(config.get("simulation.physics_substeps")),
            gravity_m_s2=tuple(float(v) for v in config.get("simulation.gravity_m_s2")),
            solver=str(config.get("simulation.solver")),
            solver_iterations=int(config.get("simulation.solver_iterations")),
            robot_control_backend=str(
                control["backend"]
            ),
            robot_control_implementation=str(control["implementation"]),
            target_velocity_mode=str(control["target_velocity_mode"]),
            arm_stiffness=float(control["arm_stiffness"]),
            arm_damping=float(control["arm_damping"]),
            finger_stiffness=float(control["finger_stiffness"]),
            finger_damping=float(control["finger_damping"]),
            door_control_backend=str(door_control["backend"]),
            door_target_stiffness=float(door_control["target_stiffness"]),
            door_target_damping=float(door_control["target_damping"]),
            door_target_velocity_rad_s=float(door_control["target_velocity_rad_s"]),
            collision_enabled=config.get("collision.enabled"),
            collision_broad_phase=config.get("collision.broad_phase"),
            collision_deterministic=config.get("collision.deterministic"),
            collision_margin_m=float(config.get("collision.margin_m")),
            allowed_contact_links=tuple(str(v) for v in config.get("collision.allowed_contact_links")),
            fixed_grasp_enabled=fixed["enabled"],
            fixed_grasp_activate_after_phase=str(fixed["activate_after_phase"]),
            grasp_activation_position_tolerance_m=float(
                fixed["activation_position_tolerance_m"]
            ),
            grasp_activation_orientation_tolerance_deg=float(
                fixed["activation_orientation_tolerance_deg"]
            ),
            device=str(config.get("runtime.device")),
        )


@dataclass(frozen=True, slots=True)
class PhysicsRollout:
    """Measured, audit-ready physics rollout."""

    phase_names: np.ndarray
    time_s: np.ndarray
    command_joint_q: np.ndarray
    measured_joint_q: np.ndarray
    measured_joint_qd: np.ndarray
    measured_arm_joint_q: np.ndarray
    door_angle_rad: np.ndarray
    ee_pose_wxyz: np.ndarray
    handle_link_pose_wxyz: np.ndarray
    body_pose_wxyz: np.ndarray
    collision_flags: np.ndarray
    grasp_constraint_active: np.ndarray
    external_robot_joint_force_command: np.ndarray
    forbidden_contact_pairs: tuple[tuple[tuple[str, str], ...], ...]
    forbidden_contact_signed_clearance_m: tuple[tuple[float, ...], ...]
    body_labels: tuple[str, ...]
    joint_labels: tuple[str, ...]
    metadata: Mapping[str, Any]

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "phase_names": self.phase_names,
            "time_s": self.time_s,
            "command_joint_q": self.command_joint_q,
            "measured_joint_q": self.measured_joint_q,
            "measured_joint_qd": self.measured_joint_qd,
            "measured_arm_joint_q": self.measured_arm_joint_q,
            "door_angle_rad": self.door_angle_rad,
            "ee_pose_wxyz": self.ee_pose_wxyz,
            "handle_link_pose_wxyz": self.handle_link_pose_wxyz,
            "body_pose_wxyz": self.body_pose_wxyz,
            "collision_flags": self.collision_flags,
            "grasp_constraint_active": self.grasp_constraint_active,
            "external_robot_joint_force_command": self.external_robot_joint_force_command,
        }

@dataclass(slots=True)
class _PhysicsRuntime:
    model: Any
    solver: Any
    collision_pipeline: Any | None
    contacts: Any | None
    state_in: Any
    state_out: Any
    control: Any
    measured_q: Any
    measured_qd: Any
    arm_refs: tuple[ScalarJointRef, ...]
    finger_refs: tuple[ScalarJointRef, ...]
    door_ref: ScalarJointRef
    end_effector_body_index: int
    handle_body_index: int
    allowed_contact_body_indices: frozenset[int]
    robot_body_indices: frozenset[int]
    object_body_indices: frozenset[int]
    grasp_joint_index: int | None
    initial_joint_q: np.ndarray
    collapsed_robot_fixed_joints: tuple[str, ...]
    controlled_coord_indices: tuple[int, ...]
    controlled_dof_indices: tuple[int, ...]
    controlled_coord_indices_device: Any
    controlled_dof_indices_device: Any
    initial_door_target_q: float
    initial_door_target_qd: float
    authored_door_target_ke: float
    authored_door_target_kd: float
    robot_body_inverse_mass_positive_verified: bool
    robot_body_flags_dynamic_verified: bool
    planned_grasp_anchors: PlannedFixedGraspAnchors | None
    collision_margin_m: float


def _newton_version() -> str | None:
    if newton is None:
        return None
    version = getattr(newton, "__version__", None)
    return str(version) if version is not None else None


def require_newton_v13() -> str:
    """Require the pinned Newton API; never pretend an unverified backend ran."""

    if _NEWTON_IMPORT_ERROR is not None or newton is None or wp is None:
        error = _NEWTON_IMPORT_ERROR
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "physics-assisted mode requires newton==1.3.0 and warp-lang==1.14.0",
            stage="physics_import",
            details={
                "exception_type": type(error).__name__ if error is not None else None,
                "exception": str(error) if error is not None else None,
            },
        ) from error
    version = _newton_version()
    if version is None or not version.startswith(_SUPPORTED_NEWTON_VERSION_PREFIX):
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "physics-assisted mode is validated only against Newton 1.3.x",
            stage="physics_import",
            details={"detected_version": version, "required_prefix": _SUPPORTED_NEWTON_VERSION_PREFIX},
        )
    return version


def _warp_transform(position: Iterable[float], orientation_wxyz: Iterable[float]) -> Any:
    position_np = _finite_vector(position, 3, "position")
    quaternion_xyzw = quaternion_wxyz_to_xyzw(normalize_quaternion(orientation_wxyz))
    return wp.transform(
        wp.vec3(*position_np.tolist()),
        wp.quat(*quaternion_xyzw.tolist()),
    )


def _matrix_from_newton_pose(row_xyzw: Sequence[float] | np.ndarray) -> np.ndarray:
    row = np.asarray(row_xyzw, dtype=float)
    if row.shape != (7,) or not np.isfinite(row).all():
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "Newton produced an invalid body transform",
            stage="physics_measurement",
            details={"pose": row.tolist()},
        )
    quaternion_wxyz = np.asarray([row[6], row[3], row[4], row[5]], dtype=float)
    return pose_matrix(row[:3], quaternion_wxyz)


def _newton_pose_row_from_matrix(matrix: np.ndarray) -> np.ndarray:
    position, quaternion_wxyz = decompose_pose(matrix)
    quaternion_xyzw = quaternion_wxyz_to_xyzw(quaternion_wxyz)
    return np.concatenate((position, quaternion_xyzw))


def _project_pose_from_matrix(matrix: np.ndarray) -> np.ndarray:
    position, quaternion_wxyz = decompose_pose(matrix)
    return np.concatenate((position, quaternion_wxyz))


def _project_body_poses(body_q_xyzw: np.ndarray) -> np.ndarray:
    result = np.empty((len(body_q_xyzw), 7), dtype=float)
    result[:, :3] = body_q_xyzw[:, :3]
    result[:, 3] = body_q_xyzw[:, 6]
    result[:, 4:] = body_q_xyzw[:, 3:6]
    return result


def _resolve_robot_body_link_names(
    body_labels: Sequence[str],
    body_indices: Sequence[int],
    robot_model: URDFModel,
) -> tuple[str, ...]:
    """Resolve finalized Newton robot bodies to unique URDF link names."""

    link_names = tuple(robot_model.link_names)
    resolved: list[str] = []
    for body_index in body_indices:
        label = str(body_labels[int(body_index)])
        matches = [
            name for name in link_names if label == name or label.endswith(f"/{name}")
        ]
        if len(matches) != 1:
            raise PipelineError(
                FailureCode.NAME_NOT_UNIQUE if matches else FailureCode.FRAME_MISSING,
                "Newton robot body cannot be mapped uniquely to a URDF link",
                stage="physics_model",
                details={
                    "body_index": int(body_index),
                    "body_label": label,
                    "matches": matches,
                },
            )
        resolved.append(matches[0])
    if len(set(resolved)) != len(resolved):
        raise PipelineError(
            FailureCode.NAME_NOT_UNIQUE,
            "multiple Newton robot bodies map to one URDF link",
            stage="physics_model",
            details={"resolved_links": resolved},
        )
    return tuple(resolved)


def named_robot_body_poses(
    robot_model: URDFModel,
    root_world_transform: np.ndarray,
    body_link_names: Sequence[str],
    arm_joint_names: Sequence[str],
    arm_joint_positions: Sequence[float] | np.ndarray,
    finger_joint_names: Sequence[str],
    gripper_width_m: float,
) -> np.ndarray:
    """Return ordered robot body poses from project name-based URDF FK."""

    arm = _finite_vector(
        arm_joint_positions, len(arm_joint_names), "arm_joint_positions"
    )
    width = _positive(gripper_width_m, "gripper_width_m", allow_zero=True)
    positions = {
        str(name): float(value)
        for name, value in zip(arm_joint_names, arm, strict=True)
    }
    positions.update({str(name): 0.5 * width for name in finger_joint_names})
    transforms = forward_kinematics(robot_model, root_world_transform, positions)
    try:
        ordered = np.stack(
            [np.asarray(transforms[name], dtype=float) for name in body_link_names]
        )
        for transform in ordered:
            decompose_pose(transform)
    except KeyError as exc:
        raise PipelineError(
            FailureCode.FRAME_MISSING,
            "name-based robot FK did not produce a required Newton body link",
            stage="physics_control",
            details={"link": str(exc.args[0])},
        ) from exc
    return ordered


def _rotation_vector_world(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    relative = current @ previous.T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1.0e-10:
        return np.asarray(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ],
            dtype=float,
        ) * 0.5
    sine = math.sin(angle)
    if abs(sine) < 1.0e-8:
        # The deterministic trajectories never intentionally jump by pi, but
        # fail closed rather than inventing an arbitrary angular velocity.
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "kinematic body driver encountered an ambiguous pi rotation",
            stage="physics_control",
            details={"angle_rad": angle},
        )
    axis = np.asarray(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ],
        dtype=float,
    ) / (2.0 * sine)
    return axis * angle


def kinematic_body_twists(
    previous_poses: np.ndarray,
    current_poses: np.ndarray,
    body_com: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Finite-difference Newton body twists ``(v_com_world, omega_world)``."""

    previous = np.asarray(previous_poses, dtype=float)
    current = np.asarray(current_poses, dtype=float)
    com = np.asarray(body_com, dtype=float)
    step = _positive(dt, "kinematic body driver dt")
    if previous.shape != current.shape or previous.ndim != 3 or previous.shape[1:] != (4, 4):
        raise ValueError("previous/current poses must have equal shape (bodies, 4, 4)")
    if com.shape != (len(previous), 3) or not np.isfinite(com).all():
        raise ValueError("body_com must have shape (bodies, 3) and be finite")
    twists = np.empty((len(previous), 6), dtype=float)
    for index, (before, after) in enumerate(zip(previous, current, strict=True)):
        decompose_pose(before)
        decompose_pose(after)
        before_com = before[:3, 3] + before[:3, :3] @ com[index]
        after_com = after[:3, 3] + after[:3, :3] @ com[index]
        twists[index, :3] = (after_com - before_com) / step
        twists[index, 3:] = (
            _rotation_vector_world(before[:3, :3], after[:3, :3]) / step
        )
    if not np.isfinite(twists).all():
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "kinematic body driver produced a non-finite twist",
            stage="physics_control",
        )
    return twists


class NewtonPhysicsAssistedSimulator:
    """Build and execute one deterministic CPU Newton physics rollout."""

    def __init__(self, parameters: PhysicsParameters) -> None:
        self.parameters = parameters

    def _joint_refs_from_builder(
        self, builder: Any, names: Sequence[str]
    ) -> tuple[ScalarJointRef, ...]:
        return resolve_named_scalar_joints(
            builder.joint_label,
            builder.joint_q_start,
            builder.joint_qd_start,
            builder.joint_dof_dim,
            names,
            joint_coord_count=builder.joint_coord_count,
            joint_dof_count=builder.joint_dof_count,
        )

    def _set_builder_initial_coordinates(
        self,
        builder: Any,
        arm_refs: Sequence[ScalarJointRef],
        finger_refs: Sequence[ScalarJointRef],
        door_ref: ScalarJointRef,
        initial_arm_joint_positions: Sequence[float] | np.ndarray,
        initial_gripper_width_m: float,
    ) -> None:
        initial_arm = _finite_vector(
            initial_arm_joint_positions,
            len(arm_refs),
            "initial_arm_joint_positions",
        )
        initial_width = _positive(
            initial_gripper_width_m,
            "initial_gripper_width_m",
            allow_zero=True,
        )
        for ref, value in zip(arm_refs, initial_arm, strict=True):
            builder.joint_q[ref.coord_index] = float(value)
        finger_half_width = 0.5 * initial_width
        for ref in finger_refs:
            builder.joint_q[ref.coord_index] = finger_half_width
        # The door coordinate is never assigned by this backend, including at
        # scene construction.  Its authored URDF default must match the
        # configured closed angle; otherwise the scene is rejected.
        authored_door = float(builder.joint_q[door_ref.coord_index])
        if not math.isfinite(authored_door) or not math.isclose(
            authored_door,
            self.parameters.closed_angle_rad,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise PipelineError(
                FailureCode.CONFIG_INVALID,
                "configured closed angle differs from the untouched URDF door default",
                stage="physics_model",
                details={
                    "door_joint": door_ref.configured_name,
                    "urdf_default_rad": authored_door,
                    "configured_closed_rad": self.parameters.closed_angle_rad,
                },
            )

    @staticmethod
    def _set_builder_robot_pd(
        builder: Any,
        arm_refs: Sequence[ScalarJointRef],
        finger_refs: Sequence[ScalarJointRef],
        *,
        arm_stiffness: float,
        arm_damping: float,
        finger_stiffness: float,
        finger_damping: float,
    ) -> None:
        """Configure Newton XPBD position/velocity PD for robot joints only.

        The caller passes only name-resolved Franka references.  This method
        deliberately has no door reference and therefore cannot mutate the
        door target or its gains.
        """

        for refs, stiffness, damping in (
            (arm_refs, arm_stiffness, arm_damping),
            (finger_refs, finger_stiffness, finger_damping),
        ):
            for ref in refs:
                builder.joint_target_ke[ref.dof_index] = float(stiffness)
                builder.joint_target_kd[ref.dof_index] = float(damping)
                builder.joint_target_q[ref.coord_index] = float(
                    builder.joint_q[ref.coord_index]
                )
                builder.joint_target_qd[ref.dof_index] = 0.0

    @staticmethod
    def _add_disabled_fixed_grasp_joint(
        builder: Any,
        *,
        parent: int,
        child: int,
        parent_xform: Any,
        child_xform: Any,
    ) -> int:
        """Author the inactive grasp loop without hiding hand-handle contact."""

        return builder.add_joint_fixed(
            parent=parent,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            label=_GRASP_JOINT_LABEL,
            collision_filter_parent=False,
            enabled=False,
        )

    def _build_runtime(
        self,
        commands: PhysicsCommandTrajectory,
        grasp_window: GraspActivationWindow | None,
    ) -> _PhysicsRuntime:
        require_newton_v13()
        params = self.parameters
        for kind, path in (("robot", params.robot_urdf), ("object", params.object_urdf)):
            if not Path(path).expanduser().is_file():
                raise PipelineError(
                    FailureCode.ASSET_MISSING,
                    f"physics {kind} URDF does not exist: {path}",
                    stage="physics_model",
                    details={"kind": kind, "path": str(path)},
                )
        if commands.arm_joint_targets.shape[1] != len(params.arm_joint_names):
            raise PipelineError(
                FailureCode.CONFIG_INVALID,
                "physics arm command width does not match configured arm joints",
                stage="physics_trajectory",
                details={
                    "command_joint_count": int(commands.arm_joint_targets.shape[1]),
                    "configured_joint_count": len(params.arm_joint_names),
                },
            )
        if params.fixed_grasp_enabled and (
            commands.handle_frame_in_link is None
            or commands.expected_handle_to_tcp is None
        ):
            raise PipelineError(
                FailureCode.CONFIG_INVALID,
                "fixed grasp requires planned handle-frame and handle-to-TCP transforms",
                stage="physics_trajectory",
            )

        try:
            hand_to_tcp = pose_matrix(
                params.end_effector_offset_position,
                params.end_effector_offset_orientation_wxyz,
            )
            planned_anchors = (
                planned_fixed_grasp_anchors(
                    hand_to_tcp,
                    commands.handle_frame_in_link,
                    commands.expected_handle_to_tcp,
                )
                if params.fixed_grasp_enabled
                else None
            )
            newton.use_coord_layout_targets = True
            with wp.ScopedDevice(params.device):
                builder = newton.ModelBuilder()
                robot_body_start = builder.body_count
                builder.add_urdf(
                    str(Path(params.robot_urdf).expanduser().resolve()),
                    xform=_warp_transform(
                        params.robot_world_position,
                        params.robot_world_orientation_wxyz,
                    ),
                    floating=False,
                    hide_visuals=True,
                    parse_visuals_as_colliders=False,
                    # The configured first-stage acceptance scope is
                    # cross-asset robot-object contact only.
                    enable_self_collisions=False,
                    collapse_fixed_joints=False,
                    collapse_massless_fixed_root=False,
                    override_root_xform=True,
                )
                collapsed_robot_fixed_joints, fixed_joints_to_keep = (
                    plan_massless_fixed_joint_collapse(
                        builder.joint_label,
                        builder.joint_type,
                        builder.joint_parent,
                        builder.joint_child,
                        builder.body_mass,
                        fixed_joint_type=int(newton.JointType.FIXED),
                    )
                )
                if collapsed_robot_fixed_joints:
                    builder.collapse_fixed_joints(
                        joints_to_keep=fixed_joints_to_keep
                    )
                remaining_massless_fixed, _ = plan_massless_fixed_joint_collapse(
                    builder.joint_label,
                    builder.joint_type,
                    builder.joint_parent,
                    builder.joint_child,
                    builder.body_mass,
                    fixed_joint_type=int(newton.JointType.FIXED),
                )
                if remaining_massless_fixed:
                    raise PipelineError(
                        FailureCode.ASSET_INVALID,
                        "robot retains a massless fixed child that XPBD would treat as static",
                        stage="physics_model",
                        details={"joints": list(remaining_massless_fixed)},
                    )
                robot_body_labels = tuple(
                    str(label) for label in builder.body_label[robot_body_start:]
                )
                object_body_start = builder.body_count
                builder.add_urdf(
                    str(Path(params.object_urdf).expanduser().resolve()),
                    xform=_warp_transform(
                        params.object_world_position,
                        params.object_world_orientation_wxyz,
                    ),
                    floating=False,
                    hide_visuals=True,
                    parse_visuals_as_colliders=False,
                    enable_self_collisions=False,
                    collapse_fixed_joints=False,
                    collapse_massless_fixed_root=False,
                    override_root_xform=True,
                )
                object_body_labels = tuple(
                    str(label) for label in builder.body_label[object_body_start:]
                )

                arm_refs = self._joint_refs_from_builder(builder, params.arm_joint_names)
                finger_refs = self._joint_refs_from_builder(builder, params.finger_joint_names)
                door_ref = self._joint_refs_from_builder(builder, (params.door_joint,))[0]
                controlled_dofs = {ref.dof_index for ref in (*arm_refs, *finger_refs)}
                if door_ref.dof_index in controlled_dofs:
                    raise PipelineError(
                        FailureCode.CONFIG_INVALID,
                        "configured door joint overlaps a controlled robot joint",
                        stage="physics_model",
                    )
                self._set_builder_initial_coordinates(
                    builder,
                    arm_refs,
                    finger_refs,
                    door_ref,
                    commands.arm_joint_targets[0],
                    float(commands.gripper_width_m[0]),
                )
                authored_door_target_ke = float(
                    builder.joint_target_ke[door_ref.dof_index]
                )
                authored_door_target_kd = float(
                    builder.joint_target_kd[door_ref.dof_index]
                )
                # Disable only the imported door actuator gains.  The door's
                # coordinate, velocity, position/velocity targets, and force
                # slot remain untouched throughout construction and rollout.
                builder.joint_target_ke[door_ref.dof_index] = (
                    params.door_target_stiffness
                )
                builder.joint_target_kd[door_ref.dof_index] = (
                    params.door_target_damping
                )
                self._set_builder_robot_pd(
                    builder,
                    arm_refs,
                    finger_refs,
                    arm_stiffness=params.arm_stiffness,
                    arm_damping=params.arm_damping,
                    finger_stiffness=params.finger_stiffness,
                    finger_damping=params.finger_damping,
                )

                ee_body = resolve_unique_label(
                    builder.body_label, params.end_effector_link, kind="body"
                ).index
                handle_body = resolve_unique_label(
                    builder.body_label, params.handle_link, kind="body"
                ).index

                grasp_joint_index: int | None = None
                if params.fixed_grasp_enabled:
                    if grasp_window is None:
                        raise PipelineError(
                            FailureCode.CONFIG_INVALID,
                            "fixed grasp is enabled but no activation window was computed",
                            stage="physics_model",
                        )
                    grasp_joint_index = self._add_disabled_fixed_grasp_joint(
                        builder,
                        parent=ee_body,
                        child=handle_body,
                        # These anchors encode the planned handle-frame/TCP
                        # relation.  Runtime capture is deliberately forbidden.
                        parent_xform=_warp_transform(
                            *decompose_pose(planned_anchors.parent_xform)
                        ),
                        child_xform=_warp_transform(
                            *decompose_pose(planned_anchors.child_xform)
                        ),
                    )
                    # A loop-closing joint is a constraint, not part of either
                    # articulation's FK tree (official Newton loop-joint pattern).
                    builder.joint_articulation[grasp_joint_index] = -1

                model = builder.finalize(device=params.device, requires_grad=False)
                model.set_gravity(params.gravity_m_s2)
                if not model.use_coord_layout_targets:
                    raise PipelineError(
                        FailureCode.PHYSICS_UNAVAILABLE,
                        "Newton model did not retain coordinate-layout targets",
                        stage="physics_model",
                    )
                if params.collision_margin_m > 0.0 and model.shape_gap is not None:
                    gaps = model.shape_gap.numpy()
                    gaps[:] = np.maximum(gaps, params.collision_margin_m)
                    model.shape_gap.assign(
                        wp.array(gaps, dtype=wp.float32, device=params.device)
                    )

                # Re-resolve finalized labels/layout.  Builder indices are not
                # assumed to survive finalization even though they normally do.
                arm_refs = resolve_named_scalar_joints(
                    model.joint_label,
                    model.joint_q_start.numpy(),
                    model.joint_qd_start.numpy(),
                    model.joint_dof_dim.numpy(),
                    params.arm_joint_names,
                )
                finger_refs = resolve_named_scalar_joints(
                    model.joint_label,
                    model.joint_q_start.numpy(),
                    model.joint_qd_start.numpy(),
                    model.joint_dof_dim.numpy(),
                    params.finger_joint_names,
                )
                door_ref = resolve_named_scalar_joints(
                    model.joint_label,
                    model.joint_q_start.numpy(),
                    model.joint_qd_start.numpy(),
                    model.joint_dof_dim.numpy(),
                    (params.door_joint,),
                )[0]
                ee_body = resolve_unique_label(
                    model.body_label, params.end_effector_link, kind="body"
                ).index
                handle_body = resolve_unique_label(
                    model.body_label, params.handle_link, kind="body"
                ).index
                if grasp_joint_index is not None:
                    grasp_joint_index = resolve_unique_label(
                        model.joint_label, _GRASP_JOINT_LABEL, kind="joint"
                    ).index

                allowed_body_indices: set[int] = set()
                finalized_label_to_index = {
                    str(label): index for index, label in enumerate(model.body_label)
                }
                if len(finalized_label_to_index) != len(model.body_label):
                    raise PipelineError(
                        FailureCode.NAME_NOT_UNIQUE,
                        "Newton finalized duplicate body labels",
                        stage="physics_model",
                    )
                try:
                    robot_body_indices = frozenset(
                        finalized_label_to_index[label] for label in robot_body_labels
                    )
                    object_body_indices = frozenset(
                        finalized_label_to_index[label] for label in object_body_labels
                    )
                except KeyError as exc:
                    raise PipelineError(
                        FailureCode.FRAME_MISSING,
                        "Newton finalization dropped a named imported body",
                        stage="physics_model",
                        details={"missing_label": str(exc.args[0])},
                    ) from exc
                if robot_body_indices.intersection(object_body_indices):
                    raise PipelineError(
                        FailureCode.NAME_NOT_UNIQUE,
                        "robot and object body index sets overlap after finalization",
                        stage="physics_model",
                    )
                robot_body_indices_ordered = tuple(sorted(robot_body_indices))
                body_flags = model.body_flags.numpy().astype(np.int64, copy=False)
                inv_mass = model.body_inv_mass.numpy().astype(float, copy=False)
                robot_flags = body_flags[list(robot_body_indices_ordered)]
                robot_inv_mass = inv_mass[list(robot_body_indices_ordered)]
                kinematic_flag = int(newton.BodyFlags.KINEMATIC)
                robot_body_flags_dynamic_verified = bool(
                    np.all(np.bitwise_and(robot_flags, kinematic_flag) == 0)
                )
                robot_body_inverse_mass_positive_verified = bool(
                    np.isfinite(robot_inv_mass).all()
                    and np.all(robot_inv_mass > 0.0)
                )
                if not robot_body_flags_dynamic_verified:
                    raise PipelineError(
                        FailureCode.PHYSICS_UNAVAILABLE,
                        "Franka contains a kinematic body; joint PD requires dynamic bodies",
                        stage="physics_model",
                    )
                if not robot_body_inverse_mass_positive_verified:
                    raise PipelineError(
                        FailureCode.ASSET_INVALID,
                        "Franka contains a non-dynamic or non-finite body mass",
                        stage="physics_model",
                        details={"robot_inverse_mass": robot_inv_mass.tolist()},
                    )
                if params.collision_enabled:
                    for body_name in params.allowed_contact_links:
                        allowed_body_indices.add(
                            resolve_unique_label(
                                model.body_label, body_name, kind="body"
                            ).index
                        )
                    collision_pipeline = newton.CollisionPipeline(
                        model,
                        broad_phase=params.collision_broad_phase,
                        deterministic=params.collision_deterministic,
                        reduce_contacts=True,
                    )
                    contacts = collision_pipeline.contacts()
                else:
                    collision_pipeline = None
                    contacts = None

                solver = newton.solvers.SolverXPBD(
                    model, iterations=params.solver_iterations
                )
                state_in = model.state()
                state_out = model.state()
                control = model.control()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
                newton.eval_fk(model, model.joint_q, model.joint_qd, state_out)
                measured_q = wp.clone(model.joint_q)
                measured_qd = wp.clone(model.joint_qd)
                newton.eval_ik(model, state_in, measured_q, measured_qd)
                final_controlled_dofs = tuple(
                    ref.dof_index for ref in (*arm_refs, *finger_refs)
                )
                if door_ref.dof_index in final_controlled_dofs:
                    raise PipelineError(
                        FailureCode.CONFIG_INVALID,
                        "finalized robot control mapping overlaps the door DoF",
                        stage="physics_model",
                    )
                final_controlled_coords = tuple(
                    ref.coord_index for ref in (*arm_refs, *finger_refs)
                )
                if door_ref.coord_index in final_controlled_coords:
                    raise PipelineError(
                        FailureCode.CONFIG_INVALID,
                        "finalized robot target mapping overlaps the door coordinate",
                        stage="physics_model",
                    )
                if len(set(final_controlled_coords)) != len(final_controlled_coords) or len(
                    set(final_controlled_dofs)
                ) != len(final_controlled_dofs):
                    raise PipelineError(
                        FailureCode.NAME_NOT_UNIQUE,
                        "finalized robot PD mappings contain duplicate coordinates or DoFs",
                        stage="physics_model",
                    )

                target_ke = model.joint_target_ke.numpy().astype(float, copy=False)
                target_kd = model.joint_target_kd.numpy().astype(float, copy=False)
                expected_ke = np.asarray(
                    [params.arm_stiffness] * len(arm_refs)
                    + [params.finger_stiffness] * len(finger_refs),
                    dtype=float,
                )
                expected_kd = np.asarray(
                    [params.arm_damping] * len(arm_refs)
                    + [params.finger_damping] * len(finger_refs),
                    dtype=float,
                )
                if not (
                    np.allclose(
                        target_ke[list(final_controlled_dofs)], expected_ke, rtol=0.0, atol=1.0e-6
                    )
                    and np.allclose(
                        target_kd[list(final_controlled_dofs)], expected_kd, rtol=0.0, atol=1.0e-6
                    )
                ):
                    raise PipelineError(
                        FailureCode.PHYSICS_UNAVAILABLE,
                        "Newton finalized different robot joint-PD gains than configured",
                        stage="physics_model",
                    )
                if not (
                    math.isclose(
                        float(target_ke[door_ref.dof_index]),
                        params.door_target_stiffness,
                        abs_tol=1.0e-6,
                    )
                    and math.isclose(
                        float(target_kd[door_ref.dof_index]),
                        params.door_target_damping,
                        abs_tol=1.0e-6,
                    )
                ):
                    raise PipelineError(
                        FailureCode.PHYSICS_UNAVAILABLE,
                        "Newton did not retain the configured passive door gains",
                        stage="physics_model",
                        details={
                            "door_target_ke": float(target_ke[door_ref.dof_index]),
                            "door_target_kd": float(target_kd[door_ref.dof_index]),
                        },
                    )
                if control.joint_target_q is None or control.joint_target_qd is None:
                    raise PipelineError(
                        FailureCode.PHYSICS_UNAVAILABLE,
                        "Newton control did not expose joint position/velocity targets",
                        stage="physics_model",
                    )
                initial_door_target_q = float(
                    control.joint_target_q.numpy()[door_ref.coord_index]
                )
                initial_door_target_qd = float(
                    control.joint_target_qd.numpy()[door_ref.dof_index]
                )
                if not math.isclose(
                    initial_door_target_qd,
                    params.door_target_velocity_rad_s,
                    rel_tol=0.0,
                    abs_tol=0.0,
                ):
                    raise PipelineError(
                        FailureCode.ASSET_INVALID,
                        "authored door velocity target differs from configuration",
                        stage="physics_model",
                        details={
                            "authored_target_velocity_rad_s": initial_door_target_qd,
                            "configured_target_velocity_rad_s": (
                                params.door_target_velocity_rad_s
                            ),
                        },
                    )
                if not np.all(control.joint_f.numpy() == 0.0):
                    raise PipelineError(
                        FailureCode.ASSET_INVALID,
                        "authored model contains a nonzero generalized-force command",
                        stage="physics_model",
                    )
                controlled_coord_indices_device = wp.array(
                    np.asarray(final_controlled_coords, dtype=np.int32),
                    dtype=wp.int32,
                    device=model.device,
                )
                controlled_dof_indices_device = wp.array(
                    np.asarray(final_controlled_dofs, dtype=np.int32),
                    dtype=wp.int32,
                    device=model.device,
                )

                return _PhysicsRuntime(
                    model=model,
                    solver=solver,
                    collision_pipeline=collision_pipeline,
                    contacts=contacts,
                    state_in=state_in,
                    state_out=state_out,
                    control=control,
                    measured_q=measured_q,
                    measured_qd=measured_qd,
                    arm_refs=arm_refs,
                    finger_refs=finger_refs,
                    door_ref=door_ref,
                    end_effector_body_index=ee_body,
                    handle_body_index=handle_body,
                    allowed_contact_body_indices=frozenset(allowed_body_indices),
                    robot_body_indices=robot_body_indices,
                    object_body_indices=object_body_indices,
                    grasp_joint_index=grasp_joint_index,
                    initial_joint_q=model.joint_q.numpy().astype(float, copy=True),
                    collapsed_robot_fixed_joints=collapsed_robot_fixed_joints,
                    controlled_coord_indices=final_controlled_coords,
                    controlled_dof_indices=final_controlled_dofs,
                    controlled_coord_indices_device=controlled_coord_indices_device,
                    controlled_dof_indices_device=controlled_dof_indices_device,
                    initial_door_target_q=initial_door_target_q,
                    initial_door_target_qd=initial_door_target_qd,
                    authored_door_target_ke=authored_door_target_ke,
                    authored_door_target_kd=authored_door_target_kd,
                    robot_body_inverse_mass_positive_verified=(
                        robot_body_inverse_mass_positive_verified
                    ),
                    robot_body_flags_dynamic_verified=(
                        robot_body_flags_dynamic_verified
                    ),
                    planned_grasp_anchors=planned_anchors,
                    collision_margin_m=params.collision_margin_m,
                )
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(
                FailureCode.PHYSICS_UNAVAILABLE,
                "Newton could not build the combined robot/microwave physics model",
                stage="physics_model",
                details={"exception_type": type(exc).__name__, "exception": str(exc)},
            ) from exc

    @staticmethod
    def _set_joint_enabled(runtime: _PhysicsRuntime, joint_index: int, enabled: bool) -> None:
        flags = runtime.model.joint_enabled.numpy()
        flags[int(joint_index)] = bool(enabled)
        runtime.model.joint_enabled.assign(
            wp.array(flags, dtype=bool, device=runtime.model.device)
        )

    @staticmethod
    def _forbidden_contact_evidence(
        runtime: _PhysicsRuntime,
    ) -> tuple[tuple[tuple[str, str], float], ...]:
        """Return forbidden cross-asset pairs and signed surface clearance.

        Newton's collision pipeline emits speculative contact candidates as
        well as active contacts.  XPBD itself ignores a candidate when the
        signed effective-surface separation is compared to the configured
        acceptance margin.  Every reported pair therefore carries the exact
        signed clearance used by the gate.
        """

        if runtime.contacts is None:
            return ()
        count = int(runtime.contacts.rigid_contact_count.numpy()[0])
        if count <= 0:
            return ()
        shape0 = runtime.contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = runtime.contacts.rigid_contact_shape1.numpy()[:count]
        point0 = runtime.contacts.rigid_contact_point0.numpy()[:count]
        point1 = runtime.contacts.rigid_contact_point1.numpy()[:count]
        normals = runtime.contacts.rigid_contact_normal.numpy()[:count]
        margin0 = runtime.contacts.rigid_contact_margin0.numpy()[:count]
        margin1 = runtime.contacts.rigid_contact_margin1.numpy()[:count]
        shape_body = runtime.model.shape_body.numpy()
        body_q = runtime.state_in.body_q.numpy()
        labels = runtime.model.body_label
        allowed = runtime.allowed_contact_body_indices
        pair_clearance: dict[tuple[str, str], float] = {}
        for contact_index, (first_shape, second_shape) in enumerate(
            zip(shape0, shape1, strict=True)
        ):
            if int(first_shape) < 0 or int(second_shape) < 0:
                continue
            first_body = int(shape_body[int(first_shape)])
            second_body = int(shape_body[int(second_shape)])
            if first_body == second_body:
                continue
            cross_asset = (
                first_body in runtime.robot_body_indices
                and second_body in runtime.object_body_indices
            ) or (
                second_body in runtime.robot_body_indices
                and first_body in runtime.object_body_indices
            )
            if not cross_asset:
                continue
            if first_body in allowed and second_body in allowed:
                continue

            try:
                first_world = _matrix_from_newton_pose(body_q[first_body])
                second_world = _matrix_from_newton_pose(body_q[second_body])
                first_point_world = transform_point(
                    first_world, point0[contact_index]
                )
                second_point_world = transform_point(
                    second_world, point1[contact_index]
                )
                normal = _finite_vector(
                    normals[contact_index], 3, "contact_normal"
                )
                first_margin = float(margin0[contact_index])
                second_margin = float(margin1[contact_index])
                if not math.isfinite(first_margin) or not math.isfinite(
                    second_margin
                ):
                    raise ValueError("contact margins must be finite")
                separation = float(
                    np.dot(normal, second_point_world - first_point_world)
                    - (first_margin + second_margin)
                )
            except (IndexError, TypeError, ValueError) as exc:
                raise PipelineError(
                    FailureCode.NUMERICAL_INSTABILITY,
                    "Newton produced malformed rigid-contact evidence",
                    stage="physics_collision",
                    details={
                        "contact_index": contact_index,
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                    },
                ) from exc
            if not math.isfinite(separation):
                raise PipelineError(
                    FailureCode.NUMERICAL_INSTABILITY,
                    "Newton produced a non-finite contact separation",
                    stage="physics_collision",
                    details={"contact_index": contact_index},
                )
            if separation > runtime.collision_margin_m:
                continue
            first_label = "<world>" if first_body < 0 else str(labels[first_body])
            second_label = "<world>" if second_body < 0 else str(labels[second_body])
            pair = tuple(sorted((first_label, second_label)))
            pair_clearance[pair] = min(
                separation, pair_clearance.get(pair, math.inf)
            )
        return tuple(
            (pair, float(pair_clearance[pair])) for pair in sorted(pair_clearance)
        )

    @staticmethod
    def _forbidden_contact_pairs(
        runtime: _PhysicsRuntime,
    ) -> tuple[tuple[str, str], ...]:
        """Compatibility projection of signed contact evidence to pair names."""

        return tuple(
            pair
            for pair, _ in NewtonPhysicsAssistedSimulator._forbidden_contact_evidence(
                runtime
            )
        )

    def run(self, commands: PhysicsCommandTrajectory) -> PhysicsRollout:
        params = self.parameters
        grasp_window = (
            fixed_grasp_activation_window(
                commands.phase_names, params.fixed_grasp_activate_after_phase
            )
            if params.fixed_grasp_enabled
            else None
        )
        runtime = self._build_runtime(commands, grasp_window)
        frame_count = commands.frame_count
        coord_count = int(runtime.model.joint_coord_count)
        dof_count = int(runtime.model.joint_dof_count)
        body_count = int(runtime.model.body_count)

        command_q = np.empty((frame_count, coord_count), dtype=float)
        measured_q = np.empty((frame_count, coord_count), dtype=float)
        measured_qd = np.empty((frame_count, dof_count), dtype=float)
        arm_q = np.empty((frame_count, len(runtime.arm_refs)), dtype=float)
        door_angle = np.empty(frame_count, dtype=float)
        ee_pose = np.empty((frame_count, 7), dtype=float)
        handle_pose = np.empty((frame_count, 7), dtype=float)
        body_pose = np.empty((frame_count, body_count, 7), dtype=float)
        collision_flags = np.zeros(frame_count, dtype=bool)
        constraint_active = np.zeros(frame_count, dtype=bool)
        external_joint_force = np.zeros((frame_count, dof_count), dtype=float)
        contact_pairs: list[tuple[tuple[str, str], ...]] = []
        contact_clearances: list[tuple[float, ...]] = []
        fixed_enabled = False
        activation_gate: FixedGraspActivationGate | None = None
        substep_dt = params.dt / params.substeps

        try:
            for frame_index in range(frame_count):
                should_enable = bool(
                    grasp_window is not None and grasp_window.is_active(frame_index)
                )
                if runtime.grasp_joint_index is not None and should_enable != fixed_enabled:
                    if should_enable:
                        if runtime.planned_grasp_anchors is None:
                            raise PipelineError(
                                FailureCode.CONFIG_INVALID,
                                "fixed grasp has no planned anchors",
                                stage="physics_constraint",
                            )
                        body_q_before_enable = (
                            runtime.state_in.body_q.numpy().astype(float, copy=True)
                        )
                        activation_gate = evaluate_fixed_grasp_activation_gate(
                            _matrix_from_newton_pose(
                                body_q_before_enable[
                                    runtime.end_effector_body_index
                                ]
                            ),
                            _matrix_from_newton_pose(
                                body_q_before_enable[runtime.handle_body_index]
                            ),
                            runtime.planned_grasp_anchors,
                            frame_index=frame_index,
                            position_limit_m=(
                                params.grasp_activation_position_tolerance_m
                            ),
                            orientation_limit_deg=(
                                params.grasp_activation_orientation_tolerance_deg
                            ),
                        )
                        if not activation_gate.passed:
                            raise PipelineError(
                                FailureCode.IK_UNREACHABLE,
                                "fixed grasp activation rejected: measured relation does not match the plan",
                                stage="physics_constraint",
                                details=activation_gate.audit_metadata(),
                            )
                    self._set_joint_enabled(
                        runtime, runtime.grasp_joint_index, should_enable
                    )
                    fixed_enabled = should_enable
                constraint_active[frame_index] = fixed_enabled

                desired_q = build_robot_position_targets(
                    runtime.initial_joint_q,
                    runtime.arm_refs,
                    runtime.finger_refs,
                    commands.arm_joint_targets[frame_index],
                    float(commands.gripper_width_m[frame_index]),
                    protected_coord_indices=(runtime.door_ref.coord_index,),
                )
                command_q[frame_index] = desired_q

                previous_frame = max(frame_index - 1, 0)
                previous_arm = commands.arm_joint_targets[previous_frame]
                current_arm = commands.arm_joint_targets[frame_index]
                previous_width = float(commands.gripper_width_m[previous_frame])
                current_width = float(commands.gripper_width_m[frame_index])
                if frame_index == 0:
                    arm_velocity_target = np.zeros_like(current_arm)
                    finger_velocity_target = 0.0
                else:
                    arm_velocity_target = (current_arm - previous_arm) / params.dt
                    finger_velocity_target = (
                        0.5 * (current_width - previous_width) / params.dt
                    )
                controlled_velocity_targets = np.concatenate(
                    (
                        np.asarray(arm_velocity_target, dtype=np.float32),
                        np.full(
                            len(runtime.finger_refs),
                            finger_velocity_target,
                            dtype=np.float32,
                        ),
                    )
                )
                for substep_index in range(params.substeps):
                    amount = float(substep_index + 1) / float(params.substeps)
                    interpolated_arm = (
                        (1.0 - amount) * previous_arm + amount * current_arm
                    )
                    interpolated_width = (
                        (1.0 - amount) * previous_width + amount * current_width
                    )
                    controlled_position_targets = np.concatenate(
                        (
                            np.asarray(interpolated_arm, dtype=np.float32),
                            np.full(
                                len(runtime.finger_refs),
                                0.5 * interpolated_width,
                                dtype=np.float32,
                            ),
                        )
                    )
                    if _scatter_indexed_joint_pd_targets is None:
                        raise PipelineError(
                            FailureCode.PHYSICS_UNAVAILABLE,
                            "indexed joint-PD target scatter kernel is unavailable",
                            stage="physics_control",
                        )
                    wp.launch(
                        kernel=_scatter_indexed_joint_pd_targets,
                        dim=len(runtime.controlled_coord_indices),
                        inputs=[
                            runtime.controlled_coord_indices_device,
                            runtime.controlled_dof_indices_device,
                            wp.array(
                                controlled_position_targets,
                                dtype=wp.float32,
                                device=params.device,
                            ),
                            wp.array(
                                controlled_velocity_targets,
                                dtype=wp.float32,
                                device=params.device,
                            ),
                        ],
                        outputs=[
                            runtime.control.joint_target_q,
                            runtime.control.joint_target_qd,
                        ],
                        device=params.device,
                    )
                    runtime.state_in.clear_forces()
                    runtime.state_out.clear_forces()
                    if runtime.collision_pipeline is not None:
                        runtime.collision_pipeline.collide(
                            runtime.state_in, runtime.contacts
                        )
                    runtime.solver.step(
                        runtime.state_in,
                        runtime.state_out,
                        runtime.control,
                        runtime.contacts,
                        substep_dt,
                    )
                    runtime.state_in, runtime.state_out = (
                        runtime.state_out,
                        runtime.state_in,
                    )

                # XPBD evolves dynamic robot and object body state.  Generalized
                # coordinates are measured through IK into dedicated buffers;
                # no measured coordinate is copied back into the door state.
                newton.eval_ik(
                    runtime.model,
                    runtime.state_in,
                    runtime.measured_q,
                    runtime.measured_qd,
                )

                if runtime.collision_pipeline is not None:
                    runtime.collision_pipeline.collide(
                        runtime.state_in, runtime.contacts
                    )
                evidence = self._forbidden_contact_evidence(runtime)
                pairs = tuple(pair for pair, _ in evidence)
                clearances = tuple(clearance for _, clearance in evidence)
                contact_pairs.append(pairs)
                contact_clearances.append(clearances)
                collision_flags[frame_index] = bool(pairs)

                q_np = runtime.measured_q.numpy().astype(float, copy=True)
                qd_np = runtime.measured_qd.numpy().astype(float, copy=True)
                body_q_np = runtime.state_in.body_q.numpy().astype(float, copy=True)
                if not (
                    np.isfinite(q_np).all()
                    and np.isfinite(qd_np).all()
                    and np.isfinite(body_q_np).all()
                    and np.isfinite(external_joint_force[frame_index]).all()
                ):
                    raise PipelineError(
                        FailureCode.NUMERICAL_INSTABILITY,
                        "Newton physics produced a non-finite state",
                        stage="physics_step",
                        details={"frame_index": frame_index},
                    )

                measured_q[frame_index] = q_np
                measured_qd[frame_index] = qd_np
                arm_q[frame_index] = q_np[
                    [ref.coord_index for ref in runtime.arm_refs]
                ]
                door_angle[frame_index] = q_np[runtime.door_ref.coord_index]
                body_pose[frame_index] = _project_body_poses(body_q_np)
                hand_world = _matrix_from_newton_pose(
                    body_q_np[runtime.end_effector_body_index]
                )
                tcp_offset = pose_matrix(
                    params.end_effector_offset_position,
                    params.end_effector_offset_orientation_wxyz,
                )
                ee_pose[frame_index] = _project_pose_from_matrix(
                    compose_transforms(hand_world, tcp_offset)
                )
                handle_pose[frame_index] = _project_body_poses(
                    body_q_np[runtime.handle_body_index : runtime.handle_body_index + 1]
                )[0]
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(
                FailureCode.PHYSICS_UNAVAILABLE,
                "Newton physics-assisted rollout failed",
                stage="physics_step",
                details={"exception_type": type(exc).__name__, "exception": str(exc)},
            ) from exc

        final_door_target_q = float(
            runtime.control.joint_target_q.numpy()[runtime.door_ref.coord_index]
        )
        final_door_target_qd = float(
            runtime.control.joint_target_qd.numpy()[runtime.door_ref.dof_index]
        )
        if not (
            math.isclose(
                final_door_target_q,
                runtime.initial_door_target_q,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and math.isclose(
                final_door_target_qd,
                runtime.initial_door_target_qd,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise PipelineError(
                FailureCode.NUMERICAL_INSTABILITY,
                "indexed robot target writer modified the door target",
                stage="physics_audit",
                details={
                    "initial_target_q": runtime.initial_door_target_q,
                    "final_target_q": final_door_target_q,
                    "initial_target_qd": runtime.initial_door_target_qd,
                    "final_target_qd": final_door_target_qd,
                },
            )
        if not np.all(runtime.control.joint_f.numpy() == 0.0):
            raise PipelineError(
                FailureCode.NUMERICAL_INSTABILITY,
                "joint-PD rollout unexpectedly used generalized-force commands",
                stage="physics_audit",
            )

        version = require_newton_v13()
        metadata: dict[str, Any] = {
            **_physics_safety_audit_metadata(),
            "status": "completed",
            "backend": "newton",
            "newton_version": version,
            "device": params.device,
            "solver": params.solver,
            "solver_iterations": params.solver_iterations,
            "physics_substeps": params.substeps,
            "control_backend": "joint_pd",
            "robot_control_implementation": params.robot_control_implementation,
            "robot_control_semantics": "dynamic_robot_newton_xpbd_joint_position_velocity_pd",
            "robot_joint_coordinate_semantics": "newton_eval_ik_measured_dynamic_state",
            "door_coordinate_measurement_semantics": "newton_eval_ik_from_dynamic_body_state",
            "robot_body_state_write_backend": "none",
            "robot_body_state_runtime_write_count": 0,
            "robot_target_write_backend": "indexed_scatter_controlled_robot_coordinates_and_dofs_only",
            "joint_force_write_backend": "none",
            "external_robot_joint_force_command_semantics": "all_zero_no_joint_f_writes",
            "joint_pd_controller_used": True,
            "torque_pd_controller_used": False,
            "target_velocity_mode": params.target_velocity_mode,
            "arm_stiffness": params.arm_stiffness,
            "arm_damping": params.arm_damping,
            "finger_stiffness": params.finger_stiffness,
            "finger_damping": params.finger_damping,
            "robot_scene_initialization": "first_commanded_waypoint",
            "door_scene_initialization_position_write_count": 0,
            "door_initialization": "untouched_urdf_default_verified_against_config",
            "massless_fixed_joint_policy": "selective_collapse_nonroot_massless_child",
            "collapsed_robot_fixed_joints": list(
                runtime.collapsed_robot_fixed_joints
            ),
            "door_actuation": "passive_velocity_damping_only",
            "door_position_actuation": "none",
            "door_velocity_control_semantics": "zero_velocity_passive_damping_no_runtime_target_writes",
            "door_control_backend": params.door_control_backend,
            "door_model_gain_write_count": 2,
            "door_model_gain_write_semantics": "configure_zero_position_stiffness_and_passive_velocity_damping",
            "door_authored_target_ke": runtime.authored_door_target_ke,
            "door_authored_target_kd": runtime.authored_door_target_kd,
            "door_runtime_position_write_count": 0,
            "door_runtime_velocity_write_count": 0,
            "door_runtime_target_write_count": 0,
            "door_runtime_generalized_force_write_count": 0,
            "door_zero_write_evidence": "static_indexed_control_path_guarantee",
            "door_target_values_unchanged_verified": True,
            "door_initial_target_q": runtime.initial_door_target_q,
            "door_final_target_q": final_door_target_q,
            "door_initial_target_qd": runtime.initial_door_target_qd,
            "door_final_target_qd": final_door_target_qd,
            "door_target_ke": params.door_target_stiffness,
            "door_target_kd": params.door_target_damping,
            "door_target_velocity_rad_s": params.door_target_velocity_rad_s,
            "robot_body_indices_written": [],
            "object_body_indices_untouched_by_robot_control": sorted(
                runtime.object_body_indices
            ),
            "robot_object_body_index_sets_disjoint": runtime.robot_body_indices.isdisjoint(
                runtime.object_body_indices
            ),
            "controlled_robot_coord_indices": list(
                runtime.controlled_coord_indices
            ),
            "controlled_robot_dof_indices": list(runtime.controlled_dof_indices),
            "door_coord_index": runtime.door_ref.coord_index,
            "door_dof_index": runtime.door_ref.dof_index,
            "door_coord_excluded_from_driver": True,
            "door_dof_excluded_from_driver": True,
            "door_coord_excluded_from_target_writer": True,
            "door_dof_excluded_from_target_writer": True,
            "robot_body_inverse_mass_positive_verified": (
                runtime.robot_body_inverse_mass_positive_verified
            ),
            "robot_body_flags_dynamic_verified": (
                runtime.robot_body_flags_dynamic_verified
            ),
            "constraint_backend": (
                "newton_fixed_loop_joint_planned_anchors_with_pre_enable_gate"
                if params.fixed_grasp_enabled
                else "disabled"
            ),
            "grasp_activation_frame": (
                grasp_window.activation_frame if grasp_window is not None else None
            ),
            "grasp_release_frame": (
                grasp_window.release_frame if grasp_window is not None else None
            ),
            "pose_layout": "xyz_wxyz",
            "handle_pose_semantics": "configured_handle_link_body_frame",
            "collision_semantics": "cross-asset signed effective-surface clearance at_or_below_configured_margin",
            "collision_margin_m": params.collision_margin_m,
            "collision_clearance_reported": True,
            "collision_evidence_scope": "cross_asset_robot_object",
        }
        if runtime.planned_grasp_anchors is not None:
            metadata.update(
                {
                    "grasp_anchor_source": "planned_handle_frame_to_tcp_relation",
                    "grasp_anchor_runtime_model_write_count": 0,
                    "grasp_anchor_parent_xform_xyz_wxyz": _project_pose_from_matrix(
                        runtime.planned_grasp_anchors.parent_xform
                    ).tolist(),
                    "grasp_anchor_child_xform_xyz_wxyz": _project_pose_from_matrix(
                        runtime.planned_grasp_anchors.child_xform
                    ).tolist(),
                    "remote_latch_allowed": False,
                }
            )
        if activation_gate is not None:
            metadata.update(activation_gate.audit_metadata())
        else:
            metadata.update(
                {
                    "grasp_activation_gate_frame": None,
                    "grasp_activation_gate_position_error_m": None,
                    "grasp_activation_gate_orientation_error_deg": None,
                    "grasp_activation_gate_position_limit_m": (
                        params.grasp_activation_position_tolerance_m
                    ),
                    "grasp_activation_gate_orientation_limit_deg": (
                        params.grasp_activation_orientation_tolerance_deg
                    ),
                    "grasp_activation_gate_passed": (
                        None if params.fixed_grasp_enabled else False
                    ),
                }
            )
        if commands.door_reference_rad is not None:
            metadata["door_reference_semantics"] = "diagnostic_only_never_applied"

        return PhysicsRollout(
            phase_names=commands.phase_names.copy(),
            time_s=_post_step_state_sample_times(frame_count, params.dt),
            command_joint_q=command_q,
            measured_joint_q=measured_q,
            measured_joint_qd=measured_qd,
            measured_arm_joint_q=arm_q,
            door_angle_rad=door_angle,
            ee_pose_wxyz=ee_pose,
            handle_link_pose_wxyz=handle_pose,
            body_pose_wxyz=body_pose,
            collision_flags=collision_flags,
            grasp_constraint_active=constraint_active,
            external_robot_joint_force_command=external_joint_force,
            forbidden_contact_pairs=tuple(contact_pairs),
            forbidden_contact_signed_clearance_m=tuple(contact_clearances),
            body_labels=tuple(str(label) for label in runtime.model.body_label),
            joint_labels=tuple(str(label) for label in runtime.model.joint_label),
            metadata=metadata,
        )


def simulate_physics_assisted(
    config: Any,
    commands: PhysicsCommandTrajectory,
) -> PhysicsRollout:
    """Convenience entrypoint used by the CLI orchestration layer."""

    try:
        parameters = PhysicsParameters.from_project_config(config)
    except PipelineError:
        raise
    except (TypeError, ValueError, KeyError) as exc:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "cannot construct physics-assisted parameters from config",
            stage="physics_config",
            details={"exception_type": type(exc).__name__, "exception": str(exc)},
        ) from exc
    return NewtonPhysicsAssistedSimulator(parameters).run(commands)


def _json_safe(value: Any) -> Any:
    """Return a recursively JSON-safe diagnostic without NaN/Infinity."""

    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _read_json_mapping(path: Path, *, stage: str) -> dict[str, Any]:
    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant: {token}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_nonfinite,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise PipelineError(
            FailureCode.OUTPUT_FAILURE,
            f"cannot read reference artifact: {path}",
            stage=stage,
            details={"path": str(path), "error": repr(exc)},
        ) from exc
    if not isinstance(value, dict):
        raise PipelineError(
            FailureCode.OUTPUT_FAILURE,
            f"reference artifact is not a JSON object: {path}",
            stage=stage,
            details={"path": str(path), "actual_type": type(value).__name__},
        )
    return value


def _flag_vector(name: str, values: Any, frame_count: int) -> np.ndarray:
    flags = np.asarray(values)
    if flags.shape != (frame_count,):
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            f"{name} must contain one value per physics frame",
            stage="physics_result",
            details={"expected": [frame_count], "actual": list(flags.shape)},
        )
    if np.issubdtype(flags.dtype, np.bool_):
        return flags.astype(bool, copy=False)
    try:
        numeric = flags.astype(float)
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            f"{name} must contain only booleans or numeric 0/1",
            stage="physics_result",
        ) from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            f"{name} must contain only booleans or numeric 0/1",
            stage="physics_result",
        )
    return numeric.astype(bool)


def _pose_rows_to_matrices(rows: Any, *, name: str) -> np.ndarray:
    poses = np.asarray(rows, dtype=float)
    if poses.ndim != 2 or poses.shape[1:] != (7,):
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            f"{name} must have shape (frames, 7) in xyz_wxyz order",
            stage="physics_result",
            details={"actual": list(poses.shape)},
        )
    if not np.isfinite(poses).all():
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            f"{name} contains a non-finite physics measurement",
            stage="physics_result",
        )
    result = np.empty((len(poses), 4, 4), dtype=float)
    try:
        for index, row in enumerate(poses):
            result[index] = pose_matrix(row[:3], row[3:])
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            f"{name} contains an invalid quaternion",
            stage="physics_result",
            details={"error": str(exc)},
        ) from exc
    return result


def handle_frame_world_from_link_poses(
    handle_link_pose_wxyz: Any,
    link_to_handle_frame: Any,
) -> np.ndarray:
    """Compose measured world-to-link poses with the configured handle frame.

    Pose rows use project order ``[x, y, z, qw, qx, qy, qz]``.  The local
    frame offset is a full rigid transform, so both its rotation and its
    translation are transformed by the measured link pose.
    """

    link_world = _pose_rows_to_matrices(
        handle_link_pose_wxyz, name="handle_link_pose_wxyz"
    )
    try:
        local = np.asarray(link_to_handle_frame, dtype=float)
        # decompose_pose performs strict finite/rigid validation.
        decompose_pose(local)
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "selected handle frame transform is malformed",
            stage="physics_reference",
            details={"error": str(exc)},
        ) from exc
    return np.stack(
        [compose_transforms(world, local) for world in link_world], axis=0
    )


def _validate_rigid_stack(name: str, values: Any, frame_count: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (frame_count, 4, 4):
        raise PipelineError(
            FailureCode.OUTPUT_FAILURE,
            f"reference {name} has an invalid shape",
            stage="physics_reference",
            details={"expected": [frame_count, 4, 4], "actual": list(array.shape)},
        )
    try:
        for transform in array:
            decompose_pose(transform)
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            f"reference {name} is not a finite rigid transform",
            stage="physics_reference",
            details={"error": str(exc)},
        ) from exc
    return array


def _load_reference_artifacts(destination: Path) -> dict[str, Any]:
    trajectory_path = destination / "trajectory.npz"
    required = {
        "phase_names",
        "phase_indices",
        "time_s",
        "door_angle_rad",
        "handle_world",
        "target_gripper_world",
        "gripper_width_m",
        "arm_joint_q",
        "achieved_gripper_world",
        "ik_success_flags",
        "collision_flags",
        "objective_cost",
    }
    try:
        with np.load(trajectory_path, allow_pickle=False) as archive:
            missing = sorted(required.difference(archive.files))
            if missing:
                raise PipelineError(
                    FailureCode.OUTPUT_FAILURE,
                    "kinematic reference trajectory is incomplete",
                    stage="physics_reference",
                    details={"path": str(trajectory_path), "missing": missing},
                )
            arrays = {name: np.asarray(archive[name]).copy() for name in required}
    except PipelineError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise PipelineError(
            FailureCode.OUTPUT_FAILURE,
            "kinematic reference trajectory cannot be loaded",
            stage="physics_reference",
            details={"path": str(trajectory_path), "error": repr(exc)},
        ) from exc

    phases = np.asarray(arrays["phase_names"], dtype="U16")
    if phases.ndim != 1 or len(phases) == 0:
        raise PipelineError(
            FailureCode.OUTPUT_FAILURE,
            "kinematic reference has invalid phase names",
            stage="physics_reference",
            details={"shape": list(phases.shape)},
        )
    frame_count = len(phases)
    unknown = sorted(set(phases.tolist()).difference(PHASE_ORDER))
    if unknown:
        raise PipelineError(
            FailureCode.OUTPUT_FAILURE,
            "kinematic reference contains unknown phases",
            stage="physics_reference",
            details={"unknown": unknown},
        )
    expected_indices = np.asarray(
        [PHASE_ORDER.index(name) for name in phases], dtype=np.int16
    )
    phase_indices = np.asarray(arrays["phase_indices"])
    if phase_indices.shape != (frame_count,) or not np.array_equal(
        phase_indices, expected_indices
    ):
        raise PipelineError(
            FailureCode.OUTPUT_FAILURE,
            "kinematic phase indices do not match phase names",
            stage="physics_reference",
        )

    vector_names = ("time_s", "door_angle_rad", "gripper_width_m")
    for name in vector_names:
        value = np.asarray(arrays[name], dtype=float)
        if value.shape != (frame_count,) or not np.isfinite(value).all():
            raise PipelineError(
                FailureCode.NUMERICAL_INSTABILITY,
                f"kinematic reference {name} is malformed or non-finite",
                stage="physics_reference",
                details={"shape": list(value.shape)},
            )
        arrays[name] = value
    arrays["handle_world"] = _validate_rigid_stack(
        "handle_world", arrays["handle_world"], frame_count
    )
    arrays["target_gripper_world"] = _validate_rigid_stack(
        "target_gripper_world", arrays["target_gripper_world"], frame_count
    )
    achieved = np.asarray(arrays["achieved_gripper_world"], dtype=float)
    if achieved.shape != (frame_count, 4, 4):
        raise PipelineError(
            FailureCode.OUTPUT_FAILURE,
            "kinematic achieved gripper poses have an invalid shape",
            stage="physics_reference",
        )
    arrays["achieved_gripper_world"] = achieved
    arm_q = np.asarray(arrays["arm_joint_q"], dtype=float)
    if arm_q.ndim != 2 or arm_q.shape[0] != frame_count or arm_q.shape[1] < 1:
        raise PipelineError(
            FailureCode.OUTPUT_FAILURE,
            "kinematic arm commands have an invalid shape",
            stage="physics_reference",
            details={"shape": list(arm_q.shape)},
        )
    arrays["arm_joint_q"] = arm_q
    arrays["ik_success_flags"] = _flag_vector(
        "reference ik_success_flags", arrays["ik_success_flags"], frame_count
    )
    arrays["collision_flags"] = _flag_vector(
        "reference collision_flags", arrays["collision_flags"], frame_count
    )
    objective = np.asarray(arrays["objective_cost"], dtype=float)
    if objective.shape != (frame_count,):
        raise PipelineError(
            FailureCode.OUTPUT_FAILURE,
            "kinematic objective costs have an invalid shape",
            stage="physics_reference",
        )
    arrays["objective_cost"] = objective

    affordance = _read_json_mapping(
        destination / "affordance_candidates.json", stage="physics_reference"
    )
    selection = affordance.get("selection")
    selected = selection.get("selected") if isinstance(selection, dict) else None
    if not isinstance(selected, dict):
        raise PipelineError(
            FailureCode.FRAME_MISSING,
            "kinematic reference has no selected handle candidate",
            stage="physics_reference",
        )
    try:
        selected_transform = pose_matrix(
            selected["position"], selected["quaternion_wxyz"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError(
            FailureCode.FRAME_MISSING,
            "selected handle candidate has an invalid local frame",
            stage="physics_reference",
            details={"error": str(exc)},
        ) from exc

    collision_report = _read_json_mapping(
        destination / "collision_report.json", stage="physics_reference"
    )
    kinematic_metrics = _read_json_mapping(
        destination / "metrics.json", stage="physics_reference"
    )
    reference_rows: tuple[dict[str, Any], ...] = ()
    rollout_path = destination / "rollout.jsonl"
    if rollout_path.is_file():
        def reject_nonfinite(token: str) -> None:
            raise ValueError(f"non-standard JSON numeric constant: {token}")

        try:
            rows = [
                json.loads(line, parse_constant=reject_nonfinite)
                for line in rollout_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise PipelineError(
                FailureCode.OUTPUT_FAILURE,
                "kinematic rollout diagnostics cannot be loaded",
                stage="physics_reference",
                details={"error": repr(exc)},
            ) from exc
        if len(rows) != frame_count or any(not isinstance(row, dict) for row in rows):
            raise PipelineError(
                FailureCode.OUTPUT_FAILURE,
                "kinematic rollout diagnostics do not match the trajectory",
                stage="physics_reference",
                details={"expected": frame_count, "actual": len(rows)},
            )
        reference_rows = tuple(rows)

    plan = TaskPlan(
        phase_names=phases,
        phase_indices=expected_indices,
        time_s=arrays["time_s"],
        door_angle_rad=arrays["door_angle_rad"],
        handle_world=arrays["handle_world"],
        target_gripper_world=arrays["target_gripper_world"],
        gripper_width_m=arrays["gripper_width_m"],
    )
    return {
        "plan": plan,
        "arrays": arrays,
        "selected": selected,
        "selected_transform": selected_transform,
        "candidate_checks": collision_report.get("candidate_checks"),
        "kinematic_collision": collision_report.get("trajectory"),
        "kinematic_metrics": kinematic_metrics,
        "reference_rows": reference_rows,
    }


def _validate_physics_rollout(
    rollout: PhysicsRollout,
    commands: PhysicsCommandTrajectory,
) -> tuple[
    dict[str, np.ndarray],
    tuple[tuple[tuple[str, str], ...], ...],
    tuple[tuple[float, ...], ...],
    dict[str, Any],
]:
    frame_count = commands.frame_count
    phases = np.asarray(getattr(rollout, "phase_names", ()), dtype="U16")
    if phases.shape != (frame_count,) or not np.array_equal(
        phases, commands.phase_names
    ):
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "physics phases differ from the commanded trajectory",
            stage="physics_result",
            details={"expected": commands.phase_names.tolist(), "actual": phases.tolist()},
        )

    names_and_dims = {
        "time_s": 1,
        "command_joint_q": 2,
        "measured_joint_q": 2,
        "measured_joint_qd": 2,
        "measured_arm_joint_q": 2,
        "door_angle_rad": 1,
        "ee_pose_wxyz": 2,
        "handle_link_pose_wxyz": 2,
        "body_pose_wxyz": 3,
        "external_robot_joint_force_command": 2,
    }
    arrays: dict[str, np.ndarray] = {"phase_names": phases}
    for name, ndim in names_and_dims.items():
        try:
            value = np.asarray(getattr(rollout, name), dtype=float)
        except (AttributeError, TypeError, ValueError) as exc:
            raise PipelineError(
                FailureCode.PHYSICS_UNAVAILABLE,
                f"physics result is missing or has a non-numeric {name}",
                stage="physics_result",
            ) from exc
        if value.ndim != ndim or value.shape[0] != frame_count:
            raise PipelineError(
                FailureCode.PHYSICS_UNAVAILABLE,
                f"physics {name} has an invalid shape",
                stage="physics_result",
                details={"frames": frame_count, "actual": list(value.shape)},
            )
        if not np.isfinite(value).all():
            raise PipelineError(
                FailureCode.NUMERICAL_INSTABILITY,
                f"physics {name} contains NaN or Infinity",
                stage="physics_result",
            )
        arrays[name] = value

    expected_arm_shape = commands.arm_joint_targets.shape
    if arrays["measured_arm_joint_q"].shape != expected_arm_shape:
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "measured dynamic arm positions do not match configured arm layout",
            stage="physics_result",
            details={"expected": list(expected_arm_shape), "actual": list(arrays["measured_arm_joint_q"].shape)},
        )
    if arrays["command_joint_q"].shape != arrays["measured_joint_q"].shape:
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "commanded and measured full joint coordinates have different shapes",
            stage="physics_result",
        )
    if arrays["measured_joint_qd"].shape != arrays[
        "external_robot_joint_force_command"
    ].shape:
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "measured joint velocities and external force commands have different shapes",
            stage="physics_result",
        )
    if np.any(arrays["external_robot_joint_force_command"] != 0.0):
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "joint-PD backend reported a non-zero external joint-force command",
            stage="physics_audit",
        )
    if arrays["ee_pose_wxyz"].shape != (frame_count, 7) or arrays[
        "handle_link_pose_wxyz"
    ].shape != (frame_count, 7):
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "measured EE and handle link poses must use xyz_wxyz rows",
            stage="physics_result",
        )
    body = arrays["body_pose_wxyz"]
    if body.shape[2:] != (7,) or body.shape[1] < 1:
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "measured body poses must have shape (frames, bodies, 7)",
            stage="physics_result",
        )
    body_labels = tuple(str(value) for value in getattr(rollout, "body_labels", ()))
    if len(body_labels) != body.shape[1] or any(not value for value in body_labels):
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "physics body labels do not match measured body poses",
            stage="physics_result",
        )
    joint_labels = tuple(str(value) for value in getattr(rollout, "joint_labels", ()))
    if not joint_labels or any(not value for value in joint_labels):
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "physics result has no valid joint labels",
            stage="physics_result",
        )
    arrays["collision_flags"] = _flag_vector(
        "physics collision_flags", getattr(rollout, "collision_flags", ()), frame_count
    )
    arrays["grasp_constraint_active"] = _flag_vector(
        "physics grasp_constraint_active",
        getattr(rollout, "grasp_constraint_active", ()),
        frame_count,
    )

    raw_pairs = tuple(getattr(rollout, "forbidden_contact_pairs", ()))
    if len(raw_pairs) != frame_count:
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "physics contact evidence does not contain one row per frame",
            stage="physics_result",
        )
    pairs: list[tuple[tuple[str, str], ...]] = []
    raw_clearances = tuple(
        getattr(rollout, "forbidden_contact_signed_clearance_m", ())
    )
    if len(raw_clearances) != frame_count:
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "physics signed-clearance evidence does not contain one row per frame",
            stage="physics_result",
        )
    clearances: list[tuple[float, ...]] = []
    for frame_index, frame_pairs in enumerate(raw_pairs):
        normalized: list[tuple[str, str]] = []
        for pair in frame_pairs:
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or any(not isinstance(label, str) or not label for label in pair)
            ):
                raise PipelineError(
                    FailureCode.PHYSICS_UNAVAILABLE,
                    "physics contact pair is malformed",
                    stage="physics_result",
                    details={"frame_index": frame_index, "pair": _json_safe(pair)},
                )
            normalized.append((pair[0], pair[1]))
        normalized_row = tuple(normalized)
        if bool(normalized_row) != bool(arrays["collision_flags"][frame_index]):
            raise PipelineError(
                FailureCode.PHYSICS_UNAVAILABLE,
                "physics collision flag disagrees with contact-pair evidence",
                stage="physics_result",
                details={"frame_index": frame_index},
            )
        pairs.append(normalized_row)
        try:
            clearance_row = tuple(
                float(value) for value in raw_clearances[frame_index]
            )
        except (TypeError, ValueError) as exc:
            raise PipelineError(
                FailureCode.PHYSICS_UNAVAILABLE,
                "physics signed-clearance row is malformed",
                stage="physics_result",
                details={"frame_index": frame_index},
            ) from exc
        if len(clearance_row) != len(normalized_row) or not np.isfinite(
            clearance_row
        ).all():
            raise PipelineError(
                FailureCode.PHYSICS_UNAVAILABLE,
                "physics contact pairs and signed clearances disagree",
                stage="physics_result",
                details={
                    "frame_index": frame_index,
                    "pair_count": len(normalized_row),
                    "clearance_count": len(clearance_row),
                },
            )
        clearances.append(clearance_row)

    metadata_raw = getattr(rollout, "metadata", None)
    if not isinstance(metadata_raw, Mapping):
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "physics result metadata is missing",
            stage="physics_result",
        )
    metadata = dict(metadata_raw)
    if metadata.get("state_sample_timing") != _STATE_SAMPLE_TIMING:
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "physics backend did not report post-step end-of-frame state sampling",
            stage="physics_audit",
            details={"state_sample_timing": metadata.get("state_sample_timing")},
        )
    if metadata.get("grasp_parent_child_collision_filtered") is not False:
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "fixed grasp must preserve parent-child collision detection",
            stage="physics_audit",
            details={
                "grasp_parent_child_collision_filtered": metadata.get(
                    "grasp_parent_child_collision_filtered"
                )
            },
        )
    collision_margin = metadata.get("collision_margin_m")
    if (
        not isinstance(collision_margin, (int, float, np.number))
        or isinstance(collision_margin, (bool, np.bool_))
        or not math.isfinite(float(collision_margin))
        or float(collision_margin) < 0.0
    ):
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "physics metadata has no finite collision margin",
            stage="physics_audit",
        )
    for frame_index, clearance_row in enumerate(clearances):
        if any(value > float(collision_margin) for value in clearance_row):
            raise PipelineError(
                FailureCode.PHYSICS_UNAVAILABLE,
                "reported forbidden contact is above the configured margin",
                stage="physics_audit",
                details={"frame_index": frame_index},
            )
    required_zero_counts = (
        "door_runtime_position_write_count",
        "door_runtime_velocity_write_count",
        "door_runtime_target_write_count",
        "door_runtime_generalized_force_write_count",
    )
    if metadata.get("door_actuation") != "passive_velocity_damping_only" or (
        metadata.get("door_position_actuation") != "none"
    ):
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "physics backend did not prove passive-only door damping",
            stage="physics_audit",
            details={"door_actuation": metadata.get("door_actuation")},
        )
    if metadata.get("door_zero_write_evidence") != "static_indexed_control_path_guarantee":
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "physics backend did not report the static indexed door-write guarantee",
            stage="physics_audit",
        )
    if metadata.get("control_backend") != "joint_pd":
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "physics backend is not the configured joint-PD controller",
            stage="physics_audit",
            details={"control_backend": metadata.get("control_backend")},
        )
    for field in (
        "door_coord_excluded_from_target_writer",
        "door_dof_excluded_from_target_writer",
        "robot_object_body_index_sets_disjoint",
        "door_target_values_unchanged_verified",
        "robot_body_inverse_mass_positive_verified",
        "robot_body_flags_dynamic_verified",
        "joint_pd_controller_used",
    ):
        if metadata.get(field) is not True:
            raise PipelineError(
                FailureCode.NUMERICAL_INSTABILITY,
                "joint-PD isolation or dynamic-body assertion is missing",
                stage="physics_audit",
                details={"field": field, "value": metadata.get(field)},
            )
    if metadata.get("robot_target_write_backend") != (
        "indexed_scatter_controlled_robot_coordinates_and_dofs_only"
    ):
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "joint-PD backend did not prove indexed robot-only target writes",
            stage="physics_audit",
        )
    if metadata.get("robot_body_state_write_backend") != "none" or metadata.get(
        "robot_body_indices_written"
    ) != []:
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "joint-PD backend wrote robot body state directly",
            stage="physics_audit",
        )
    if metadata.get("joint_force_write_backend") != "none":
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "joint-PD backend used an unapproved generalized-force write path",
            stage="physics_audit",
        )
    for field in required_zero_counts:
        value = metadata.get(field)
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or int(value) != 0
        ):
            raise PipelineError(
                FailureCode.NUMERICAL_INSTABILITY,
                "door runtime write audit is missing or non-zero",
                stage="physics_audit",
                details={"field": field, "value": _json_safe(value)},
            )
    if commands.door_reference_rad is not None and metadata.get(
        "door_reference_semantics"
    ) != "diagnostic_only_never_applied":
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "door reference was not proven diagnostic-only",
            stage="physics_audit",
            details={"value": metadata.get("door_reference_semantics")},
        )
    arrays["body_labels"] = np.asarray(body_labels, dtype="U")
    arrays["joint_labels"] = np.asarray(joint_labels, dtype="U")
    return arrays, tuple(pairs), tuple(clearances), metadata


def _physics_rollout_rows(
    *,
    plan: TaskPlan,
    physics: Mapping[str, np.ndarray],
    achieved_gripper_world: np.ndarray,
    handle_world: np.ndarray,
    contact_pairs: tuple[tuple[tuple[str, str], ...], ...],
    contact_signed_clearances: tuple[tuple[float, ...], ...],
    arm_joint_names: Sequence[str],
    reference_arm_q: np.ndarray,
    reference_ik_flags: np.ndarray,
    reference_objective: np.ndarray,
    reference_rows: Sequence[Mapping[str, Any]],
    door_write_audit: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, phase in enumerate(plan.phase_names):
        if reference_rows:
            reference_ik = reference_rows[index].get("ik", {})
        else:
            reference_ik = {
                "success": bool(reference_ik_flags[index]),
                "objective_cost": _json_safe(reference_objective[index]),
            }
        rows.append(
            {
                "schema_version": 1,
                "frame_index": index,
                "time_s": float(physics["time_s"][index]),
                "phase": str(phase),
                "phase_index": int(plan.phase_indices[index]),
                "door_angle_rad": float(physics["door_angle_rad"][index]),
                "door_angle_deg": math.degrees(
                    float(physics["door_angle_rad"][index])
                ),
                "reference_door_angle_rad": float(plan.door_angle_rad[index]),
                "gripper_width_m": float(plan.gripper_width_m[index]),
                "handle_world": _json_safe(handle_world[index]),
                "handle_link_pose_wxyz": _json_safe(
                    physics["handle_link_pose_wxyz"][index]
                ),
                "target_gripper_world": _json_safe(
                    plan.target_gripper_world[index]
                ),
                "achieved_gripper_world": _json_safe(
                    achieved_gripper_world[index]
                ),
                "arm_joint_positions": {
                    name: float(value)
                    for name, value in zip(
                        arm_joint_names,
                        physics["measured_arm_joint_q"][index],
                        strict=True,
                    )
                },
                "command_arm_joint_positions": {
                    name: float(value)
                    for name, value in zip(
                        arm_joint_names, reference_arm_q[index], strict=True
                    )
                },
                "collision": bool(physics["collision_flags"][index]),
                "forbidden_contact_pairs": [
                    {
                        "link_a": first,
                        "link_b": second,
                        "signed_clearance_m": float(clearance),
                    }
                    for (first, second), clearance in zip(
                        contact_pairs[index],
                        contact_signed_clearances[index],
                        strict=True,
                    )
                ],
                "grasp_constraint_active": bool(
                    physics["grasp_constraint_active"][index]
                ),
                "external_robot_joint_force_command": _json_safe(
                    physics["external_robot_joint_force_command"][index]
                ),
                "ik": {
                    "source": "kinematic_command_reference",
                    **(
                        _json_safe(reference_ik)
                        if isinstance(reference_ik, Mapping)
                        else {"diagnostic": _json_safe(reference_ik)}
                    ),
                },
                "physics": {
                    "backend": metadata.get("backend"),
                    "measurement_source": "newton_state_after_step",
                    "door_runtime_write_audit": _json_safe(door_write_audit),
                },
            }
        )
    return rows


def _with_kinematic_reference_acceptance(
    computed: Mapping[str, Any], reference_acceptance_passed: bool
) -> dict[str, Any]:
    """Make reference acceptance a fail-closed physics prerequisite."""

    passed = bool(reference_acceptance_passed)
    gates = dict(computed["gates"])
    gates["kinematic_reference_acceptance"] = {
        "value": passed,
        "operator": "required_true",
        "threshold": True,
        "passed": passed,
    }
    return {
        **computed,
        "gates": gates,
        "success": bool(computed["success"]) and passed,
    }


def run_physics_assisted(
    config_path: ProjectConfig | str | Path,
    output_dir: str | Path | None = None,
) -> Any:
    """Plan with the kinematic pipeline, then evaluate measured Newton state.

    The kinematic rollout supplies only robot commands and reference targets.
    Every physical acceptance input (TCP, handle frame, door coordinate, arm
    coordinates, and contacts) is reconstructed from the measured rollout.
    """

    if isinstance(config_path, ProjectConfig):
        config = config_path
    elif isinstance(config_path, (str, Path)):
        config = load_config(config_path)
    else:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "run_physics_assisted requires a ProjectConfig or config path",
            stage="config",
            details={"actual_type": type(config_path).__name__},
        )

    # Imported lazily to avoid the run.py -> physics_assisted.py CLI cycle.
    from .run import (
        RunOutcome,
        _RunLog,
        _arm_joint_limits,
        _bool_config,
        _output_directory,
        _resolved_config_payload,
        run_kinematic,
    )

    destination = _output_directory(config, output_dir, mode="physics_assisted")
    reference_destination = destination / "kinematic_reference"
    reference_outcome = run_kinematic(
        config, output_dir=reference_destination
    )
    reference = _load_reference_artifacts(reference_destination)
    plan: TaskPlan = reference["plan"]
    reference_arrays: dict[str, np.ndarray] = reference["arrays"]

    log = _RunLog(
        mode="physics_assisted",
        seed=int(config.get("seed")),
        path=destination / "run.log",
    )
    log.add(
        "kinematic_reference_ready",
        frame_count=len(plan.phase_names),
        acceptance_passed=bool(reference_outcome.metrics["success"]),
        exit_code=int(reference_outcome.exit_code),
    )

    reference_grasp_window = fixed_grasp_activation_window(
        plan.phase_names,
        str(
            config.get(
                "simulation.fixed_grasp_constraint.activate_after_phase"
            )
        ),
    )
    planned_relation_frame = reference_grasp_window.activation_frame - 1
    expected_handle_to_tcp = compose_transforms(
        invert_transform(plan.handle_world[planned_relation_frame]),
        plan.target_gripper_world[planned_relation_frame],
    )
    commands = PhysicsCommandTrajectory(
        phase_names=plan.phase_names,
        arm_joint_targets=reference_arrays["arm_joint_q"],
        gripper_width_m=plan.gripper_width_m,
        door_reference_rad=plan.door_angle_rad,
        handle_frame_in_link=reference["selected_transform"],
        expected_handle_to_tcp=expected_handle_to_tcp,
    )
    rollout = simulate_physics_assisted(config, commands)
    physics, contact_pairs, contact_clearances, metadata = (
        _validate_physics_rollout(rollout, commands)
    )

    # Only after a validated physics result exists do the two shared scene
    # diagnostics appear at the physics root.  All kinematic trajectory and
    # rollout artifacts remain namespaced under kinematic_reference/.
    for shared_name in ("asset_inspection.json", "affordance_candidates.json"):
        source = reference_destination / shared_name
        if not source.is_file():
            raise PipelineError(
                FailureCode.OUTPUT_FAILURE,
                "kinematic reference is missing a shared scene diagnostic",
                stage="physics_reference",
                details={"path": str(source)},
            )
        shutil.copy2(source, destination / shared_name)

    selected = reference["selected"]
    configured_handle_link = str(config.get("assets.object.handle_link"))
    if selected.get("link") != configured_handle_link:
        raise PipelineError(
            FailureCode.FRAME_MISSING,
            "selected handle frame is attached to a different link",
            stage="physics_reference",
            details={
                "selected_link": selected.get("link"),
                "configured_link": configured_handle_link,
            },
        )
    achieved_gripper_world = _pose_rows_to_matrices(
        physics["ee_pose_wxyz"], name="ee_pose_wxyz"
    )
    handle_world = handle_frame_world_from_link_poses(
        physics["handle_link_pose_wxyz"], reference["selected_transform"]
    )

    collision_scope = str(config.get("collision.scope"))
    if metadata.get("collision_evidence_scope") != collision_scope:
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "physics collision evidence scope differs from configuration",
            stage="physics_audit",
            details={
                "configured": collision_scope,
                "reported": metadata.get("collision_evidence_scope"),
            },
        )
    door_write_audit = {
        "door_actuation": metadata.get("door_actuation"),
        "door_position_actuation": metadata.get("door_position_actuation"),
        "evidence": "static_indexed_control_path_guarantee",
        "runtime_write_counts": {
            "q": int(metadata["door_runtime_position_write_count"]),
            "qd": int(metadata["door_runtime_velocity_write_count"]),
            "target": int(metadata["door_runtime_target_write_count"]),
            "generalized_force": int(
                metadata["door_runtime_generalized_force_write_count"]
            ),
        },
        "door_reference_semantics": metadata.get("door_reference_semantics"),
        "guaranteed_zero_runtime_writes": True,
        "door_coord_excluded_from_driver": bool(
            metadata.get("door_coord_excluded_from_driver")
        ),
        "door_dof_excluded_from_driver": bool(
            metadata.get("door_dof_excluded_from_driver")
        ),
    }
    joint_lower, joint_upper = _arm_joint_limits(config)
    goal_angle_rad = math.radians(float(config.get("task.goal_angle_deg")))
    try:
        computed = compute_metrics(
            phase_names=plan.phase_names,
            door_angle_rad=physics["door_angle_rad"],
            handle_world=handle_world,
            target_gripper_world=plan.target_gripper_world,
            achieved_gripper_world=achieved_gripper_world,
            joint_q=physics["measured_arm_joint_q"],
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            collision_flags=physics["collision_flags"],
            ik_success_flags=reference_arrays["ik_success_flags"],
            target_door_angle_rad=goal_angle_rad,
            thresholds=MetricThresholds.from_mapping(config.get("thresholds")),
            joint_limit_tolerance_rad=float(
                config.get("ik.joint_limit_tolerance_rad")
            ),
        )
    except MetricsInputError as exc:
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "physics rollout metrics could not be computed",
            stage="metrics",
            details=exc.to_dict(),
        ) from exc

    reference_acceptance_passed = bool(
        reference_outcome.metrics["success"]
    ) and int(reference_outcome.exit_code) == 0
    computed = _with_kinematic_reference_acceptance(
        computed, reference_acceptance_passed
    )
    run_status = "success" if bool(computed["success"]) else "acceptance_failed"
    metrics = {
        **computed,
        "mode": "physics_assisted",
        "run_status": run_status,
        "seed": int(config.get("seed")),
        "collision_scope": collision_scope,
        "selected_grasp_candidate": _json_safe(selected),
        "physics_metadata": _json_safe(metadata),
        "door_runtime_write_audit": door_write_audit,
        "kinematic_reference": {
            "acceptance_passed": reference_acceptance_passed,
            "exit_code": int(reference_outcome.exit_code),
        },
    }

    collision_frames = [
        {
            "frame_index": index,
            "phase": str(plan.phase_names[index]),
            "collision": bool(physics["collision_flags"][index]),
            "forbidden_contact_pairs": [
                {
                    "links": list(pair),
                    "signed_clearance_m": float(clearance),
                }
                for pair, clearance in zip(
                    contact_pairs[index],
                    contact_clearances[index],
                    strict=True,
                )
            ],
        }
        for index in range(len(plan.phase_names))
    ]
    write_json(
        destination / "collision_report.json",
        {
            "candidate_checks": _json_safe(reference["candidate_checks"]),
            "trajectory": {
                "backend": metadata.get("backend"),
                "scope": collision_scope,
                "frame_count": len(plan.phase_names),
                "collision_frame_count": int(
                    np.count_nonzero(physics["collision_flags"])
                ),
                "flags": physics["collision_flags"].tolist(),
                "frames": collision_frames,
                "metadata": _json_safe(metadata),
            },
            "kinematic_reference_trajectory": _json_safe(
                reference["kinematic_collision"]
            ),
        },
    )

    trajectory = {
        "phase_names": plan.phase_names,
        "phase_indices": plan.phase_indices,
        "time_s": physics["time_s"],
        "reference_time_s": plan.time_s,
        "door_angle_rad": physics["door_angle_rad"],
        "reference_door_angle_rad": plan.door_angle_rad,
        "handle_world": handle_world,
        "reference_handle_world": plan.handle_world,
        "target_gripper_world": plan.target_gripper_world,
        "achieved_gripper_world": achieved_gripper_world,
        "reference_achieved_gripper_world": reference_arrays[
            "achieved_gripper_world"
        ],
        "gripper_width_m": plan.gripper_width_m,
        "command_arm_joint_q": reference_arrays["arm_joint_q"],
        "arm_joint_q": physics["measured_arm_joint_q"],
        "joint_lower": joint_lower,
        "joint_upper": joint_upper,
        "ik_success_flags": reference_arrays["ik_success_flags"],
        "reference_collision_flags": reference_arrays["collision_flags"],
        "collision_flags": physics["collision_flags"],
        "objective_cost": reference_arrays["objective_cost"],
        **{
            name: value
            for name, value in physics.items()
            if name
            not in {
                "phase_names",
                "time_s",
                "door_angle_rad",
                "measured_arm_joint_q",
                "collision_flags",
            }
        },
    }
    write_trajectory(destination / "trajectory.npz", trajectory)
    if _bool_config(config, "output.write_rollout_jsonl"):
        write_jsonl(
            destination / "rollout.jsonl",
            _physics_rollout_rows(
                plan=plan,
                physics=physics,
                achieved_gripper_world=achieved_gripper_world,
                handle_world=handle_world,
                contact_pairs=contact_pairs,
                contact_signed_clearances=contact_clearances,
                arm_joint_names=tuple(
                    str(name)
                    for name in config.get("assets.robot.arm_joint_names")
                ),
                reference_arm_q=reference_arrays["arm_joint_q"],
                reference_ik_flags=reference_arrays["ik_success_flags"],
                reference_objective=reference_arrays["objective_cost"],
                reference_rows=reference["reference_rows"],
                door_write_audit=door_write_audit,
                metadata=metadata,
            ),
        )
    write_json(destination / "metrics.json", _json_safe(metrics))
    if _bool_config(config, "output.write_resolved_config"):
        write_json(
            destination / "resolved_config.json",
            _resolved_config_payload(config, output_dir=destination),
        )
    log.add(
        "physics_rollout_completed",
        backend=metadata.get("backend"),
        frame_count=len(plan.phase_names),
        collision_frame_count=int(np.count_nonzero(physics["collision_flags"])),
        door_runtime_write_audit=door_write_audit,
    )
    log.add(
        "run_completed",
        status=run_status,
        acceptance_passed=bool(computed["success"]),
        collision_scope=collision_scope,
    )
    return RunOutcome(
        output_dir=destination,
        metrics=metrics,
        exit_code=0 if bool(computed["success"]) else 3,
    )


__all__ = [
    "FixedGraspActivationGate",
    "GraspActivationWindow",
    "NewtonPhysicsAssistedSimulator",
    "PhysicsCommandTrajectory",
    "PhysicsParameters",
    "PhysicsRollout",
    "ScalarJointRef",
    "build_robot_position_targets",
    "evaluate_fixed_grasp_activation_gate",
    "fixed_grasp_activation_window",
    "handle_frame_world_from_link_poses",
    "kinematic_body_twists",
    "named_robot_body_poses",
    "plan_massless_fixed_joint_collapse",
    "planned_fixed_grasp_anchors",
    "require_newton_v13",
    "resolve_named_scalar_joints",
    "run_physics_assisted",
    "simulate_physics_assisted",
]
