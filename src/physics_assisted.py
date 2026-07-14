"""Newton 1.3 CPU physics-assisted execution for the microwave task.

The robot remains fully dynamic and tracks the name-resolved IK trajectory with
Newton XPBD joint position/velocity PD targets.  Runtime target writes are
scattered only to configured robot coordinate/DoF indices.  The microwave door
is never assigned a runtime coordinate, velocity, target, or generalized force.
The scene starts at the configured nominal arm pose with an open gripper and
zero joint velocity; command frame zero is reached over the first full ``dt``.

Grasp assistance is a pre-authored, initially-disabled Newton fixed loop joint.
Its authored child anchor and planned parent anchor encode the expected
handle-frame-to-TCP relationship used by a fail-closed activation gate.  Only
after that planned relation agrees with the measured state is the parent anchor
captured from the current body poses, written/read back, and enabled.  The gate
prevents a remote latch while the zero-error capture prevents a constraint snap.

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
_MAX_REPORTED_MOTION_VIOLATIONS = 32
_CAPTURE_POSITION_TOLERANCE_M = 1.0e-6
_CAPTURE_ORIENTATION_TOLERANCE_DEG = 1.0e-3
_REMOTE_LATCH_PREVENTION = (
    "planned_handle_to_tcp_pose_and_relative_anchor_twist_gates_must_pass_"
    "before_runtime_measured_anchor_capture"
)


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


def _quintic_blend_weight(index: int, count: int) -> float:
    """Endpoint-exact quintic weight for a finite sample window."""

    if not isinstance(index, int) or not isinstance(count, int) or count < 2:
        raise ValueError("quintic blend windows require at least two samples")
    if index < 0 or index >= count:
        raise ValueError("quintic blend index is outside its sample window")
    if index == 0:
        return 0.0
    if index == count - 1:
        return 1.0
    value = float(index) / float(count - 1)
    return value * value * value * (10.0 + value * (-15.0 + 6.0 * value))


@dataclass(frozen=True, slots=True)
class BumplessGraspReleaseTransfer:
    """Float32 arm targets for unloading and releasing a fixed grasp.

    The fixed constraint stays active during ``release`` while its planned hold
    target is eased to the measured pre-release equilibrium.  The first
    ``retreat`` target is that same equilibrium, so disabling the constraint
    does not also create a PD target/velocity step.  A second quintic window
    then rejoins the original retreat reference.
    """

    applied_arm_joint_targets: np.ndarray
    captured_equilibrium_arm_q: np.ndarray
    equilibrium_capture_frame: int
    release_unload_start_frame: int
    release_unload_end_frame: int
    retreat_rejoin_start_frame: int
    retreat_rejoin_end_frame: int
    retreat_blend_frames: int

    def __post_init__(self) -> None:
        targets = np.asarray(self.applied_arm_joint_targets)
        captured = np.asarray(self.captured_equilibrium_arm_q)
        if targets.ndim != 2 or targets.shape[1] < 1:
            raise ValueError("applied arm targets must be a non-empty matrix")
        if targets.dtype != np.dtype(np.float32):
            raise ValueError("applied arm targets must be realized as float32")
        if captured.shape != (targets.shape[1],) or captured.dtype != np.dtype(
            np.float32
        ):
            raise ValueError("captured arm equilibrium must be a float32 joint row")
        if not np.isfinite(targets).all() or not np.isfinite(captured).all():
            raise ValueError("bumpless release targets must be finite")
        frame_fields = (
            self.equilibrium_capture_frame,
            self.release_unload_start_frame,
            self.release_unload_end_frame,
            self.retreat_rejoin_start_frame,
            self.retreat_rejoin_end_frame,
        )
        if any(not isinstance(value, int) or value < 0 for value in frame_fields):
            raise ValueError("bumpless release frame indices must be non-negative")
        if not (
            self.equilibrium_capture_frame + 1
            == self.release_unload_start_frame
            <= self.release_unload_end_frame
            < self.retreat_rejoin_start_frame
            <= self.retreat_rejoin_end_frame
            < len(targets)
        ):
            raise ValueError("bumpless release windows are not ordered or in range")
        if (
            not isinstance(self.retreat_blend_frames, int)
            or isinstance(self.retreat_blend_frames, bool)
            or self.retreat_blend_frames < 2
            or self.retreat_rejoin_end_frame
            - self.retreat_rejoin_start_frame
            + 1
            != self.retreat_blend_frames
        ):
            raise ValueError("retreat blend frame count must be an integer >= 2")
        targets = targets.copy()
        captured = captured.copy()
        targets.setflags(write=False)
        captured.setflags(write=False)
        object.__setattr__(self, "applied_arm_joint_targets", targets)
        object.__setattr__(self, "captured_equilibrium_arm_q", captured)

    def audit_metadata(self) -> dict[str, Any]:
        return {
            "grasp_release_bumpless_transfer_enabled": True,
            "grasp_release_equilibrium_capture_source": (
                "newton_eval_ik_prior_post_step_measured_arm_q_before_"
                "release_unload_and_constraint_disable"
            ),
            "grasp_release_equilibrium_capture_frame": (
                self.equilibrium_capture_frame
            ),
            "grasp_release_equilibrium_arm_q_float32": (
                self.captured_equilibrium_arm_q.tolist()
            ),
            "grasp_release_unload_start_frame": self.release_unload_start_frame,
            "grasp_release_unload_end_frame": self.release_unload_end_frame,
            "grasp_release_unload_frame_count": (
                self.release_unload_end_frame
                - self.release_unload_start_frame
                + 1
            ),
            "grasp_release_constraint_disable_frame": (
                self.retreat_rejoin_start_frame
            ),
            "grasp_release_retreat_rejoin_start_frame": (
                self.retreat_rejoin_start_frame
            ),
            "grasp_release_retreat_rejoin_end_frame": (
                self.retreat_rejoin_end_frame
            ),
            "grasp_release_blend_frames": self.retreat_blend_frames,
            "grasp_release_blend_profile": "quintic_smoothstep_endpoint_exact",
            "grasp_release_applied_arm_target_dtype": "float32",
            "grasp_release_target_continuity_semantics": (
                "active_release_planned_hold_to_measured_equilibrium_then_"
                "disabled_retreat_equilibrium_to_planned_reference"
            ),
        }


def build_bumpless_grasp_release_transfer(
    phase_names: Sequence[str] | np.ndarray,
    planned_arm_joint_targets: Any,
    captured_equilibrium_arm_q: Any,
    grasp_window: GraspActivationWindow,
    retreat_blend_frames: int,
) -> BumplessGraspReleaseTransfer:
    """Build the exact float32 targets used for fixed-grasp release.

    This routine is deliberately independent of Newton so the controller
    transfer, its active/disabled windows, and its endpoint semantics can be
    tested without a physics installation.
    """

    phases = np.asarray(phase_names).astype(str)
    planned = np.asarray(planned_arm_joint_targets, dtype=float)
    try:
        captured = np.asarray(captured_equilibrium_arm_q, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "measured arm equilibrium cannot be represented as float32",
            stage="physics_control",
            details={"error": str(exc)},
        ) from exc
    if (
        planned.ndim != 2
        or planned.shape[0] < 1
        or phases.shape != (planned.shape[0],)
        or captured.shape != (planned.shape[1],)
    ):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "bumpless grasp-release inputs are not frame/joint aligned",
            stage="physics_trajectory",
            details={
                "phase_shape": list(phases.shape),
                "planned_shape": list(planned.shape),
                "captured_shape": list(captured.shape),
            },
        )
    if not np.isfinite(planned).all() or not np.isfinite(captured).all():
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "bumpless grasp-release inputs contain NaN or Infinity",
            stage="physics_control",
        )
    if (
        not isinstance(retreat_blend_frames, int)
        or isinstance(retreat_blend_frames, bool)
        or retreat_blend_frames < 2
    ):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "grasp release blend frames must be an integer of at least 2",
            stage="physics_trajectory",
            details={"grasp_release_blend_frames": retreat_blend_frames},
        )

    release_indices = np.flatnonzero(phases == "release")
    if (
        release_indices.size < 2
        or not np.array_equal(
            release_indices,
            np.arange(release_indices[0], release_indices[-1] + 1),
        )
        or int(release_indices[-1]) + 1 != grasp_window.release_frame
        or not np.all(phases[release_indices] == "release")
    ):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "fixed-grasp release requires one contiguous multi-frame release phase immediately before retreat",
            stage="physics_trajectory",
            details={
                "release_indices": release_indices.tolist(),
                "constraint_release_frame": grasp_window.release_frame,
            },
        )
    unload_start = int(release_indices[0])
    unload_end = int(release_indices[-1])
    capture_frame = unload_start - 1
    rejoin_start = int(grasp_window.release_frame)
    rejoin_end = rejoin_start + retreat_blend_frames - 1
    if (
        capture_frame < grasp_window.activation_frame
        or rejoin_end >= len(phases)
        or not np.all(phases[rejoin_start : rejoin_end + 1] == "retreat")
    ):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "bumpless release windows do not fit inside active release and disabled retreat phases",
            stage="physics_trajectory",
            details={
                "capture_frame": capture_frame,
                "activation_frame": grasp_window.activation_frame,
                "constraint_release_frame": rejoin_start,
                "retreat_rejoin_end_frame": rejoin_end,
                "frame_count": len(phases),
            },
        )

    with np.errstate(over="ignore", invalid="ignore"):
        planned32 = planned.astype(np.float32)
    if not np.isfinite(planned32).all():
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "planned arm targets are not representable as finite float32",
            stage="physics_control",
        )
    applied = planned32.copy()

    release_count = unload_end - unload_start + 1
    for offset, frame_index in enumerate(range(unload_start, unload_end + 1)):
        weight = _quintic_blend_weight(offset, release_count)
        blended = (
            (1.0 - weight) * planned32[frame_index].astype(float)
            + weight * captured.astype(float)
        )
        applied[frame_index] = np.asarray(blended, dtype=np.float32)
    # Make the controller-transfer boundaries bit-exact rather than relying on
    # floating-point polynomial endpoint evaluation.
    applied[unload_start] = planned32[unload_start]
    applied[unload_end] = captured

    for offset, frame_index in enumerate(range(rejoin_start, rejoin_end + 1)):
        weight = _quintic_blend_weight(offset, retreat_blend_frames)
        blended = (
            (1.0 - weight) * captured.astype(float)
            + weight * planned32[frame_index].astype(float)
        )
        applied[frame_index] = np.asarray(blended, dtype=np.float32)
    applied[rejoin_start] = captured
    applied[rejoin_end] = planned32[rejoin_end]

    return BumplessGraspReleaseTransfer(
        applied_arm_joint_targets=applied,
        captured_equilibrium_arm_q=captured,
        equilibrium_capture_frame=capture_frame,
        release_unload_start_frame=unload_start,
        release_unload_end_frame=unload_end,
        retreat_rejoin_start_frame=rejoin_start,
        retreat_rejoin_end_frame=rejoin_end,
        retreat_blend_frames=retreat_blend_frames,
    )


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
    """Measured pose and relative-anchor-twist agreement at activation."""

    frame_index: int
    parent_body_name: str
    child_body_name: str
    position_error_m: float
    orientation_error_deg: float
    position_limit_m: float
    orientation_limit_deg: float
    relative_linear_velocity_world_m_s: np.ndarray
    relative_angular_velocity_world_deg_s: np.ndarray
    linear_velocity_limit_m_s: float
    angular_velocity_limit_deg_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        for name in ("parent_body_name", "child_body_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in (
            "position_error_m",
            "orientation_error_deg",
            "position_limit_m",
            "orientation_limit_deg",
            "linear_velocity_limit_m_s",
            "angular_velocity_limit_deg_s",
        ):
            _positive(
                getattr(self, name),
                name,
                allow_zero=name in {"position_error_m", "orientation_error_deg"},
            )
        for name in (
            "relative_linear_velocity_world_m_s",
            "relative_angular_velocity_world_deg_s",
        ):
            value = _finite_vector(getattr(self, name), 3, name).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def relative_linear_speed_m_s(self) -> float:
        return float(np.linalg.norm(self.relative_linear_velocity_world_m_s))

    @property
    def relative_angular_speed_deg_s(self) -> float:
        return float(np.linalg.norm(self.relative_angular_velocity_world_deg_s))

    @property
    def pose_passed(self) -> bool:
        return (
            self.position_error_m <= self.position_limit_m
            and self.orientation_error_deg <= self.orientation_limit_deg
        )

    @property
    def twist_passed(self) -> bool:
        return (
            self.relative_linear_speed_m_s <= self.linear_velocity_limit_m_s
            and self.relative_angular_speed_deg_s
            <= self.angular_velocity_limit_deg_s
        )

    @property
    def passed(self) -> bool:
        return self.pose_passed and self.twist_passed

    def audit_metadata(self) -> dict[str, Any]:
        return {
            "grasp_activation_gate_relation_source": (
                "planned_handle_frame_to_tcp_relation"
            ),
            "grasp_activation_twist_source": (
                "newton_state_body_qd_v_com_world_omega_world_at_"
                "runtime_capture_anchor"
            ),
            "grasp_activation_gate_frame": self.frame_index,
            "grasp_activation_parent_body": self.parent_body_name,
            "grasp_activation_child_body": self.child_body_name,
            "grasp_activation_gate_position_error_m": self.position_error_m,
            "grasp_activation_gate_orientation_error_deg": self.orientation_error_deg,
            "grasp_activation_gate_position_limit_m": self.position_limit_m,
            "grasp_activation_gate_orientation_limit_deg": self.orientation_limit_deg,
            "grasp_activation_gate_relative_linear_velocity_world_m_s": (
                self.relative_linear_velocity_world_m_s.tolist()
            ),
            "grasp_activation_gate_relative_angular_velocity_world_deg_s": (
                self.relative_angular_velocity_world_deg_s.tolist()
            ),
            "grasp_activation_gate_relative_linear_speed_m_s": (
                self.relative_linear_speed_m_s
            ),
            "grasp_activation_gate_relative_angular_speed_deg_s": (
                self.relative_angular_speed_deg_s
            ),
            "grasp_activation_gate_linear_velocity_limit_m_s": (
                self.linear_velocity_limit_m_s
            ),
            "grasp_activation_gate_angular_velocity_limit_deg_s": (
                self.angular_velocity_limit_deg_s
            ),
            "grasp_activation_pose_gate_passed": self.pose_passed,
            "grasp_activation_twist_gate_passed": self.twist_passed,
            "grasp_activation_gate_passed": self.passed,
            "remote_latch_allowed": False,
            "remote_latch_prevention": _REMOTE_LATCH_PREVENTION,
        }


@dataclass(frozen=True, slots=True)
class FixedGraspChildAnchorReadback:
    """Finalized Newton child anchor verified against the authored transform."""

    child_xform: np.ndarray
    runtime_model_readback_count: int
    readback_verified: bool
    authored_match_verified: bool

    def __post_init__(self) -> None:
        child = np.asarray(self.child_xform, dtype=float)
        decompose_pose(child)
        child = child.copy()
        child.setflags(write=False)
        object.__setattr__(self, "child_xform", child)
        if self.runtime_model_readback_count != 1:
            raise ValueError("a child-anchor verification has one model readback")
        if self.readback_verified is not True or self.authored_match_verified is not True:
            raise ValueError("the finalized child anchor must be read and authored-match verified")

    def audit_metadata(self) -> dict[str, Any]:
        return {
            "grasp_anchor_child_runtime_model_readback_count": (
                self.runtime_model_readback_count
            ),
            "grasp_anchor_child_runtime_model_readback_verified": (
                self.readback_verified
            ),
            "grasp_anchor_child_authored_match_verified": (
                self.authored_match_verified
            ),
            "grasp_anchor_finalized_child_xform_xyz_wxyz": (
                _project_pose_from_matrix(self.child_xform).tolist()
            ),
        }


@dataclass(frozen=True, slots=True)
class FixedGraspJointEnabledWriteback:
    """Verified transactional write of one ``model.joint_enabled`` entry."""

    enabled: bool
    runtime_model_write_count: int
    runtime_model_readback_count: int
    readback_verified: bool

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be boolean")
        if self.runtime_model_write_count != 1:
            raise ValueError("a successful joint-enabled transaction has one write")
        if self.runtime_model_readback_count != 1:
            raise ValueError("a successful joint-enabled transaction has one readback")
        if self.readback_verified is not True:
            raise ValueError("a successful joint-enabled transaction must be verified")


@dataclass(frozen=True, slots=True)
class FixedGraspAnchorWriteback:
    """Verified write-back of one Newton fixed-joint parent anchor."""

    parent_xform: np.ndarray
    runtime_model_write_count: int
    runtime_model_readback_count: int
    readback_verified: bool

    def __post_init__(self) -> None:
        parent = np.asarray(self.parent_xform, dtype=float)
        decompose_pose(parent)
        parent = parent.copy()
        parent.setflags(write=False)
        object.__setattr__(self, "parent_xform", parent)
        if self.runtime_model_write_count != 1:
            raise ValueError("a successful grasp-anchor write-back has one write")
        if self.runtime_model_readback_count != 1:
            raise ValueError("a successful grasp-anchor write-back has one readback")
        if self.readback_verified is not True:
            raise ValueError("a successful grasp-anchor write-back must be verified")


@dataclass(frozen=True, slots=True)
class FixedGraspRuntimeCapture:
    """Measured, coincident fixed-grasp anchors installed after plan gating."""

    frame_index: int
    parent_xform: np.ndarray
    child_xform: np.ndarray
    runtime_model_write_count: int
    runtime_model_readback_count: int
    readback_verified: bool
    post_capture_position_error_m: float
    post_capture_orientation_error_deg: float

    def __post_init__(self) -> None:
        if not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        for name in ("parent_xform", "child_xform"):
            value = np.asarray(getattr(self, name), dtype=float)
            decompose_pose(value)
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.runtime_model_write_count != 1:
            raise ValueError("runtime capture must write the parent anchor once")
        if self.runtime_model_readback_count != 1:
            raise ValueError("runtime capture must read the parent anchor back once")
        if self.readback_verified is not True:
            raise ValueError("runtime capture parent-anchor readback must be verified")
        _positive(
            self.post_capture_position_error_m,
            "post_capture_position_error_m",
            allow_zero=True,
        )
        _positive(
            self.post_capture_orientation_error_deg,
            "post_capture_orientation_error_deg",
            allow_zero=True,
        )

    def audit_metadata(self) -> dict[str, Any]:
        return {
            "grasp_anchor_source": (
                "runtime_measured_parent_child_body_poses_after_"
                "planned_pose_and_relative_anchor_twist_gates"
            ),
            "grasp_anchor_capture_frame": self.frame_index,
            "grasp_anchor_runtime_model_write_count": (
                self.runtime_model_write_count
            ),
            "grasp_anchor_runtime_model_readback_count": (
                self.runtime_model_readback_count
            ),
            "grasp_anchor_runtime_model_write_readback_verified": (
                self.readback_verified
            ),
            "grasp_anchor_post_capture_position_error_m": (
                self.post_capture_position_error_m
            ),
            "grasp_anchor_post_capture_orientation_error_deg": (
                self.post_capture_orientation_error_deg
            ),
            "grasp_anchor_parent_xform_xyz_wxyz": _project_pose_from_matrix(
                self.parent_xform
            ).tolist(),
            "grasp_anchor_child_xform_xyz_wxyz": _project_pose_from_matrix(
                self.child_xform
            ).tolist(),
            "remote_latch_allowed": False,
            "remote_latch_prevention": _REMOTE_LATCH_PREVENTION,
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


def _anchor_world_twist(
    body_world: np.ndarray,
    body_qd: Sequence[float] | np.ndarray,
    body_com: Sequence[float] | np.ndarray,
    anchor_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(v_anchor_world, omega_world)`` from Newton body state.

    Newton stores ``body_qd`` as the world-frame COM linear velocity followed
    by world-frame angular velocity.  A joint anchor generally differs from the
    COM, so its linear velocity includes ``omega x r``.
    """

    world = np.asarray(body_world, dtype=float)
    anchor = np.asarray(anchor_local, dtype=float)
    decompose_pose(world)
    decompose_pose(anchor)
    twist = _finite_vector(body_qd, 6, "body_qd")
    com_local = _finite_vector(body_com, 3, "body_com")
    com_world = transform_point(world, com_local)
    anchor_world = compose_transforms(world, anchor)[:3, 3]
    omega_world = twist[3:].copy()
    velocity_world = twist[:3] + np.cross(
        omega_world, anchor_world - com_world
    )
    if not np.isfinite(velocity_world).all():
        raise ValueError("anchor world velocity is non-finite")
    return velocity_world, omega_world


def evaluate_fixed_grasp_activation_gate(
    parent_world: np.ndarray,
    child_world: np.ndarray,
    anchors: PlannedFixedGraspAnchors,
    *,
    parent_body_qd: Sequence[float] | np.ndarray,
    child_body_qd: Sequence[float] | np.ndarray,
    parent_body_com: Sequence[float] | np.ndarray,
    child_body_com: Sequence[float] | np.ndarray,
    parent_body_name: str,
    child_body_name: str,
    frame_index: int,
    position_limit_m: float,
    orientation_limit_deg: float,
    linear_velocity_limit_m_s: float,
    angular_velocity_limit_deg_s: float,
) -> FixedGraspActivationGate:
    """Require both planned-pose agreement and a settled capture-point twist."""

    try:
        parent_anchor_world = compose_transforms(parent_world, anchors.parent_xform)
        child_anchor_world = compose_transforms(child_world, anchors.child_xform)
        position_error, orientation_error = _anchor_pose_error(
            parent_anchor_world, child_anchor_world
        )
        # Twist is evaluated at the anchor that will actually be captured.  The
        # parent point is therefore the point coincident with the finalized
        # child anchor, not the nearby planned parent point used by the pose
        # remote-latch gate.
        captured_parent = captured_fixed_grasp_parent_anchor(
            parent_world, child_world, anchors.child_xform
        )
        parent_anchor_velocity, parent_angular_velocity = _anchor_world_twist(
            parent_world,
            parent_body_qd,
            parent_body_com,
            captured_parent,
        )
        child_anchor_velocity, child_angular_velocity = _anchor_world_twist(
            child_world,
            child_body_qd,
            child_body_com,
            anchors.child_xform,
        )
        relative_linear_velocity = (
            parent_anchor_velocity - child_anchor_velocity
        )
        relative_angular_velocity_deg_s = np.degrees(
            parent_angular_velocity - child_angular_velocity
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
        parent_body_name=parent_body_name,
        child_body_name=child_body_name,
        position_error_m=position_error,
        orientation_error_deg=orientation_error,
        position_limit_m=float(position_limit_m),
        orientation_limit_deg=float(orientation_limit_deg),
        relative_linear_velocity_world_m_s=relative_linear_velocity,
        relative_angular_velocity_world_deg_s=relative_angular_velocity_deg_s,
        linear_velocity_limit_m_s=float(linear_velocity_limit_m_s),
        angular_velocity_limit_deg_s=float(angular_velocity_limit_deg_s),
    )


def captured_fixed_grasp_parent_anchor(
    parent_world: np.ndarray,
    child_world: np.ndarray,
    authored_child_xform: np.ndarray,
) -> np.ndarray:
    """Return a parent-local anchor coincident with the authored child anchor.

    This is intentionally a pure transform operation.  It does not decide
    whether capture is safe: callers must first pass the activation gate using
    the *planned* handle-to-TCP relationship.  Once that gate passes, this
    measured anchor removes the residual pose error that would otherwise be
    resolved as an impulsive fixed-joint snap.
    """

    try:
        parent = np.asarray(parent_world, dtype=float)
        child = np.asarray(child_world, dtype=float)
        child_anchor = np.asarray(authored_child_xform, dtype=float)
        decompose_pose(parent)
        decompose_pose(child)
        decompose_pose(child_anchor)
        captured = compose_transforms(
            invert_transform(parent), child, child_anchor
        )
        decompose_pose(captured)
        return captured
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "cannot derive a measured fixed-grasp parent anchor",
            stage="physics_constraint",
            details={"error": str(exc)},
        ) from exc


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


@dataclass(frozen=True, slots=True)
class RobotJointVelocityLimits:
    """Name-aligned URDF velocity limits for every controlled robot joint."""

    source_urdf: Path
    arm_joint_names: tuple[str, ...]
    arm_joint_types: tuple[str, ...]
    arm_velocity_limits: tuple[float, ...]
    finger_joint_names: tuple[str, ...]
    finger_joint_types: tuple[str, ...]
    finger_velocity_limits: tuple[float, ...]

    def __post_init__(self) -> None:
        groups = (
            (
                "arm",
                self.arm_joint_names,
                self.arm_joint_types,
                self.arm_velocity_limits,
            ),
            (
                "finger",
                self.finger_joint_names,
                self.finger_joint_types,
                self.finger_velocity_limits,
            ),
        )
        for kind, names, joint_types, limits in groups:
            if len(names) != len(joint_types) or len(names) != len(limits):
                raise ValueError(f"{kind} velocity-limit fields must have equal length")
            values = np.asarray(limits, dtype=float)
            if values.shape != (len(names),) or not np.isfinite(values).all() or np.any(
                values <= 0.0
            ):
                raise ValueError(
                    f"{kind} joint velocity limits must be finite and positive"
                )
        all_names = (*self.arm_joint_names, *self.finger_joint_names)
        if len(set(all_names)) != len(all_names):
            raise ValueError("controlled velocity-limit joint names must be unique")

    @property
    def controlled_joint_names(self) -> tuple[str, ...]:
        return (*self.arm_joint_names, *self.finger_joint_names)

    @property
    def controlled_joint_types(self) -> tuple[str, ...]:
        return (*self.arm_joint_types, *self.finger_joint_types)

    @property
    def controlled_velocity_limits(self) -> np.ndarray:
        return np.asarray(
            (*self.arm_velocity_limits, *self.finger_velocity_limits), dtype=float
        )


def load_robot_joint_velocity_limits(
    robot_urdf: str | Path,
    arm_joint_names: Sequence[str],
    finger_joint_names: Sequence[str],
) -> RobotJointVelocityLimits:
    """Load positive velocity limits for every controlled joint or fail closed."""

    path = Path(robot_urdf).expanduser()
    if not path.is_file():
        raise PipelineError(
            FailureCode.ASSET_MISSING,
            f"physics robot URDF does not exist: {path}",
            stage="physics_model",
            details={"kind": "robot", "path": str(path)},
        )
    try:
        model = load_urdf(path)
    except URDFModelError as exc:
        raise PipelineError(
            FailureCode.ASSET_INVALID,
            "cannot load robot URDF velocity limits",
            stage="physics_model",
            details={
                "path": str(path),
                "urdf_error": exc.to_dict(),
            },
        ) from exc

    def resolve_group(
        names: Sequence[str], kind: str
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[float, ...]]:
        resolved_names = tuple(str(name) for name in names)
        joint_types: list[str] = []
        velocity_limits: list[float] = []
        for name in resolved_names:
            try:
                joint = model.require_joint(name)
            except URDFModelError as exc:
                raise PipelineError(
                    FailureCode.ASSET_INVALID,
                    "controlled robot joint is absent from the velocity-limit source",
                    stage="physics_model",
                    details={
                        "path": str(model.path),
                        "joint": name,
                        "joint_kind": kind,
                        "urdf_error": exc.to_dict(),
                    },
                ) from exc
            limit = joint.limit.velocity if joint.limit is not None else None
            if limit is None or not math.isfinite(limit) or limit <= 0.0:
                raise PipelineError(
                    FailureCode.ASSET_INVALID,
                    "controlled robot joint has no positive URDF velocity limit",
                    stage="physics_model",
                    details={
                        "path": str(model.path),
                        "joint": name,
                        "joint_kind": kind,
                        "joint_type": joint.joint_type,
                        "velocity_limit": limit,
                    },
                )
            joint_types.append(joint.joint_type)
            velocity_limits.append(float(limit))
        return resolved_names, tuple(joint_types), tuple(velocity_limits)

    arm_names, arm_types, arm_limits = resolve_group(arm_joint_names, "arm")
    finger_names, finger_types, finger_limits = resolve_group(
        finger_joint_names, "finger"
    )
    try:
        return RobotJointVelocityLimits(
            source_urdf=model.path,
            arm_joint_names=arm_names,
            arm_joint_types=arm_types,
            arm_velocity_limits=arm_limits,
            finger_joint_names=finger_names,
            finger_joint_types=finger_types,
            finger_velocity_limits=finger_limits,
        )
    except ValueError as exc:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "controlled robot velocity-limit mapping is invalid",
            stage="physics_model",
            details={"path": str(model.path), "error": str(exc)},
        ) from exc


def load_robot_arm_joint_position_limits(
    robot_urdf: str | Path,
    arm_joint_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load name-aligned hard URDF arm position limits."""

    path = Path(robot_urdf).expanduser()
    if not path.is_file():
        raise PipelineError(
            FailureCode.ASSET_MISSING,
            f"physics robot URDF does not exist: {path}",
            stage="physics_model",
            details={"kind": "robot", "path": str(path)},
        )
    try:
        model = load_urdf(path)
    except URDFModelError as exc:
        raise PipelineError(
            FailureCode.ASSET_INVALID,
            "cannot load robot URDF position limits",
            stage="physics_model",
            details={"path": str(path), "urdf_error": exc.to_dict()},
        ) from exc
    lower: list[float] = []
    upper: list[float] = []
    for configured_name in arm_joint_names:
        name = str(configured_name)
        try:
            joint = model.require_joint(name)
        except URDFModelError as exc:
            raise PipelineError(
                FailureCode.ASSET_INVALID,
                "controlled arm joint is absent from the position-limit source",
                stage="physics_model",
                details={
                    "path": str(model.path),
                    "joint": name,
                    "urdf_error": exc.to_dict(),
                },
            ) from exc
        if joint.joint_type == "continuous":
            lower.append(-math.inf)
            upper.append(math.inf)
        elif (
            joint.limit is None
            or joint.limit.lower is None
            or joint.limit.upper is None
            or not math.isfinite(float(joint.limit.lower))
            or not math.isfinite(float(joint.limit.upper))
            or float(joint.limit.lower) >= float(joint.limit.upper)
        ):
            raise PipelineError(
                FailureCode.ASSET_INVALID,
                "controlled arm joint has no ordered finite URDF position limits",
                stage="physics_model",
                details={
                    "path": str(model.path),
                    "joint": name,
                    "joint_type": joint.joint_type,
                },
            )
        else:
            lower.append(float(joint.limit.lower))
            upper.append(float(joint.limit.upper))
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def validate_applied_arm_target_position_limits(
    applied_arm_joint_targets: Any,
    arm_joint_names: Sequence[str],
    hard_lower_rad: Any,
    hard_upper_rad: Any,
    *,
    phase_names: Sequence[str] | np.ndarray,
) -> dict[str, Any]:
    """Fail closed if any actual float32 arm target crosses a URDF bound."""

    names = tuple(str(name) for name in arm_joint_names)
    targets = np.asarray(applied_arm_joint_targets)
    lower = np.asarray(hard_lower_rad, dtype=float)
    upper = np.asarray(hard_upper_rad, dtype=float)
    phases = np.asarray(phase_names).astype(str)
    if (
        not names
        or targets.ndim != 2
        or targets.shape != (len(phases), len(names))
        or lower.shape != (len(names),)
        or upper.shape != (len(names),)
        or np.isnan(lower).any()
        or np.isnan(upper).any()
        or np.any(lower >= upper)
    ):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "applied arm target position-limit inputs are not name/shape aligned",
            stage="physics_control",
            details={
                "target_shape": list(targets.shape),
                "phase_shape": list(phases.shape),
                "joint_names": list(names),
                "lower_shape": list(lower.shape),
                "upper_shape": list(upper.shape),
            },
        )
    with np.errstate(over="ignore", invalid="ignore"):
        realized = targets.astype(np.float32).astype(float)
    if not np.isfinite(realized).all():
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "applied arm targets are not finite after float32 realization",
            stage="physics_control",
        )
    lower_violation = realized < lower[None, :]
    upper_violation = realized > upper[None, :]
    violating = lower_violation | upper_violation
    if np.any(violating):
        frame_index, joint_index = (
            int(value) for value in np.argwhere(violating)[0]
        )
        side = "lower" if lower_violation[frame_index, joint_index] else "upper"
        bound = lower[joint_index] if side == "lower" else upper[joint_index]
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "bumpless grasp-release arm target exceeds a hard URDF position limit",
            stage="physics_control",
            details={
                "frame_index": frame_index,
                "phase": str(phases[frame_index]),
                "joint_index": joint_index,
                "joint_name": names[joint_index],
                "value_rad": float(realized[frame_index, joint_index]),
                "limit_side": side,
                "limit_rad": float(bound),
            },
        )
    lower_clearance = realized - lower[None, :]
    upper_clearance = upper[None, :] - realized
    clearance = np.minimum(lower_clearance, upper_clearance)
    finite = np.isfinite(clearance)
    if finite.any():
        comparable = np.where(finite, clearance, math.inf)
        flat_index = int(np.argmin(comparable))
        frame_index, joint_index = (
            int(value) for value in np.unravel_index(flat_index, clearance.shape)
        )
        minimum: float | None = float(clearance[frame_index, joint_index])
        minimum_frame: int | None = frame_index
        minimum_joint: int | None = joint_index
    else:
        minimum = None
        minimum_frame = None
        minimum_joint = None
    return {
        "passed": True,
        "source": "hard_position_limits_from_robot_urdf",
        "target_realization": "round_to_nearest_float32_before_audit",
        "frame_count": int(len(realized)),
        "joint_names": list(names),
        "hard_lower_rad": [
            float(value) if math.isfinite(float(value)) else None
            for value in lower
        ],
        "hard_upper_rad": [
            float(value) if math.isfinite(float(value)) else None
            for value in upper
        ],
        "violation_count": 0,
        "minimum_clearance_rad": minimum,
        "minimum_clearance_frame": minimum_frame,
        "minimum_clearance_phase": (
            None if minimum_frame is None else str(phases[minimum_frame])
        ),
        "minimum_clearance_joint_index": minimum_joint,
        "minimum_clearance_joint_name": (
            None if minimum_joint is None else names[minimum_joint]
        ),
    }


def _controlled_robot_command_positions(
    commands: PhysicsCommandTrajectory,
    velocity_limits: RobotJointVelocityLimits,
) -> np.ndarray:
    if commands.arm_joint_targets.shape[1] != len(
        velocity_limits.arm_joint_names
    ):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "physics arm command width does not match the velocity-limit mapping",
            stage="physics_trajectory",
            details={
                "command_joint_count": int(commands.arm_joint_targets.shape[1]),
                "velocity_limit_joint_count": len(
                    velocity_limits.arm_joint_names
                ),
            },
        )
    finger_targets = np.repeat(
        (0.5 * commands.gripper_width_m)[:, None],
        len(velocity_limits.finger_joint_names),
        axis=1,
    )
    return np.concatenate((commands.arm_joint_targets, finger_targets), axis=1)


def _velocity_limit_tolerance(limits: np.ndarray, dt: float) -> np.ndarray:
    """Round-off allowance in position-delta units, never a motion allowance."""

    allowed_delta = np.asarray(limits, dtype=float) * float(dt)
    return 64.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(allowed_delta))


def _inward_float32_position_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the first float32 coordinates inside closed position bounds.

    Newton stores joint targets as float32.  A direct cast of a lower bound can
    round below that bound, and a direct cast of an upper bound can round above
    it.  Moving outward-rounded values inward by exactly one representable
    float32 value gives the boundary coordinates that the IK projector and the
    physics target writer can actually realize.  This is a representational
    definition, not an ``isclose`` tolerance or an extra motion allowance.
    """

    lower_values = np.asarray(lower, dtype=float)
    upper_values = np.asarray(upper, dtype=float)
    if (
        lower_values.ndim != 1
        or upper_values.shape != lower_values.shape
        or np.isnan(lower_values).any()
        or np.isnan(upper_values).any()
        or np.any(lower_values >= upper_values)
    ):
        raise ValueError("position bounds must be ordered one-dimensional arrays")

    inward_lower = lower_values.copy()
    inward_upper = upper_values.copy()
    for index, (lower_value, upper_value) in enumerate(
        zip(lower_values, upper_values, strict=True)
    ):
        if math.isfinite(float(lower_value)):
            lower32 = np.float32(lower_value)
            if float(lower32) < float(lower_value):
                lower32 = np.nextafter(lower32, np.float32(math.inf))
            inward_lower[index] = float(lower32)
        if math.isfinite(float(upper_value)):
            upper32 = np.float32(upper_value)
            if float(upper32) > float(upper_value):
                upper32 = np.nextafter(upper32, np.float32(-math.inf))
            inward_upper[index] = float(upper32)
        if not inward_lower[index] <= inward_upper[index]:
            raise ValueError(
                "position interval contains no representable float32 coordinate"
            )
    return inward_lower, inward_upper


def audit_arm_joint_reference_reserve(
    reference_arm_q: Any,
    arm_joint_names: Sequence[str],
    hard_lower_rad: Any,
    hard_upper_rad: Any,
    *,
    initial_arm_q: Any,
    control_limit_margin_rad: float,
    arm_joint_tracking_reserve_rad: float,
    phase_names: Sequence[str] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Audit the physical initial state and reference before Newton physics.

    Hard-limit clearance is measured against the URDF limits.  Control bounds
    reproduce Newton's representation path: URDF limits are first realized as
    float32 model values, the configured control margin is applied, and the
    comparison uses the first float32 coordinate inside each resulting bound.
    Any control-bound touch therefore includes the exact saturated endpoint,
    without using a scale-dependent ``isclose`` tolerance.  The configured
    initial arm state is an explicit sample before trajectory frame zero; audit
    metadata keeps that sample separate from rollout frame and phase indices.
    """

    names = tuple(str(name) for name in arm_joint_names)
    if not names or any(not name.strip() for name in names) or len(set(names)) != len(
        names
    ):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "arm joint names for the physics reference reserve audit are invalid",
            stage="ik_motion_limits",
            details={"arm_joint_names": list(names)},
        )
    try:
        reference = np.asarray(reference_arm_q, dtype=float)
        initial = np.asarray(initial_arm_q, dtype=float)
        hard_lower = np.asarray(hard_lower_rad, dtype=float)
        hard_upper = np.asarray(hard_upper_rad, dtype=float)
        margin = float(control_limit_margin_rad)
        reserve = float(arm_joint_tracking_reserve_rad)
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "physics reference reserve audit inputs are not numeric",
            stage="ik_motion_limits",
            details={"exception": str(exc)},
        ) from exc
    joint_count = len(names)
    if (
        reference.ndim != 2
        or reference.shape[0] < 1
        or reference.shape[1] != joint_count
        or initial.shape != (joint_count,)
        or hard_lower.shape != (joint_count,)
        or hard_upper.shape != (joint_count,)
    ):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "physics reference and arm joint limits are not shape-aligned",
            stage="ik_motion_limits",
            details={
                "reference_shape": list(reference.shape),
                "initial_shape": list(initial.shape),
                "joint_count": joint_count,
                "hard_lower_shape": list(hard_lower.shape),
                "hard_upper_shape": list(hard_upper.shape),
            },
        )
    if phase_names is None:
        phases: np.ndarray | None = None
    else:
        phases = np.asarray(phase_names).astype(str)
        if phases.shape != (reference.shape[0],):
            raise PipelineError(
                FailureCode.CONFIG_INVALID,
                "physics reference phases do not match the arm trajectory",
                stage="ik_motion_limits",
                details={
                    "phase_shape": list(phases.shape),
                    "reference_frame_count": int(reference.shape[0]),
                },
            )
    if not math.isfinite(margin) or margin <= 0.0:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "IK control-limit margin must be finite and positive",
            stage="ik_motion_limits",
            details={"control_limit_margin_rad": control_limit_margin_rad},
        )
    if not math.isfinite(reserve) or reserve <= 0.0:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "arm joint tracking reserve must be finite and positive",
            stage="ik_motion_limits",
            details={
                "arm_joint_tracking_reserve_rad": arm_joint_tracking_reserve_rad
            },
        )
    if not np.isfinite(reference).all():
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "kinematic arm reference contains non-finite coordinates",
            stage="ik_motion_limits",
        )
    if not np.isfinite(initial).all():
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "configured initial arm state contains non-finite coordinates",
            stage="ik_motion_limits",
        )
    if (
        np.isnan(hard_lower).any()
        or np.isnan(hard_upper).any()
        or np.any(hard_lower >= hard_upper)
    ):
        raise PipelineError(
            FailureCode.ASSET_INVALID,
            "URDF arm joint limits are unordered or contain NaN",
            stage="ik_motion_limits",
        )

    with np.errstate(over="ignore", invalid="ignore"):
        realized_reference = reference.astype(np.float32).astype(float)
        realized_initial = initial.astype(np.float32).astype(float)
        model_hard_lower = hard_lower.astype(np.float32).astype(float)
        model_hard_upper = hard_upper.astype(np.float32).astype(float)
    if not (
        np.isfinite(realized_reference).all()
        and np.isfinite(realized_initial).all()
    ):
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "initial or reference arm coordinates are not representable as finite float32",
            stage="ik_motion_limits",
        )
    realized_samples = np.vstack((realized_initial[None, :], realized_reference))

    finite_lower = np.isfinite(model_hard_lower)
    finite_upper = np.isfinite(model_hard_upper)
    continuous_control_lower = model_hard_lower.copy()
    continuous_control_upper = model_hard_upper.copy()
    continuous_control_lower[finite_lower] += margin
    continuous_control_upper[finite_upper] -= margin
    try:
        control_lower, control_upper = _inward_float32_position_bounds(
            continuous_control_lower,
            continuous_control_upper,
        )
    except ValueError as exc:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "IK control-limit margin leaves no float32 arm coordinate",
            stage="ik_motion_limits",
            details={"control_limit_margin_rad": margin, "exception": str(exc)},
        ) from exc

    lower_hard_clearance = np.full(
        realized_samples.shape, math.inf, dtype=float
    )
    upper_hard_clearance = np.full(
        realized_samples.shape, math.inf, dtype=float
    )
    lower_hard_clearance[:, np.isfinite(hard_lower)] = (
        realized_samples[:, np.isfinite(hard_lower)]
        - hard_lower[np.isfinite(hard_lower)]
    )
    upper_hard_clearance[:, np.isfinite(hard_upper)] = (
        hard_upper[np.isfinite(hard_upper)]
        - realized_samples[:, np.isfinite(hard_upper)]
    )
    hard_clearance = np.minimum(lower_hard_clearance, upper_hard_clearance)
    hard_reserve_violations = np.isfinite(hard_clearance) & (
        hard_clearance < reserve
    )

    finite_hard_clearance = np.isfinite(hard_clearance)
    if finite_hard_clearance.any():
        comparable_hard = np.where(finite_hard_clearance, hard_clearance, math.inf)
        hard_flat_index = int(np.argmin(comparable_hard))
        hard_sample_index, hard_joint_index = np.unravel_index(
            hard_flat_index,
            hard_clearance.shape,
        )
        hard_side = (
            "lower"
            if lower_hard_clearance[hard_sample_index, hard_joint_index]
            <= upper_hard_clearance[hard_sample_index, hard_joint_index]
            else "upper"
        )
        min_hard_clearance: float | None = float(
            hard_clearance[hard_sample_index, hard_joint_index]
        )
        min_hard_sample: int | None = int(hard_sample_index)
        min_hard_joint: int | None = int(hard_joint_index)
    else:
        min_hard_clearance = None
        min_hard_sample = None
        min_hard_joint = None
        hard_side = None

    lower_control_clearance = np.full(
        realized_samples.shape, math.inf, dtype=float
    )
    upper_control_clearance = np.full(
        realized_samples.shape, math.inf, dtype=float
    )
    lower_control_clearance[:, np.isfinite(control_lower)] = (
        realized_samples[:, np.isfinite(control_lower)]
        - control_lower[np.isfinite(control_lower)]
    )
    upper_control_clearance[:, np.isfinite(control_upper)] = (
        control_upper[np.isfinite(control_upper)]
        - realized_samples[:, np.isfinite(control_upper)]
    )
    lower_touches = np.isfinite(control_lower)[None, :] & (
        realized_samples <= control_lower[None, :]
    )
    upper_touches = np.isfinite(control_upper)[None, :] & (
        realized_samples >= control_upper[None, :]
    )
    control_touches = lower_touches | upper_touches
    touch_indices = np.argwhere(control_touches)
    if touch_indices.size:
        first_touch_sample = int(touch_indices[0, 0])
        first_touch_joint = int(touch_indices[0, 1])
        first_touch_side = (
            "lower"
            if lower_touches[first_touch_sample, first_touch_joint]
            else "upper"
        )
        first_touch_bound = (
            control_lower[first_touch_joint]
            if first_touch_side == "lower"
            else control_upper[first_touch_joint]
        )
        first_touch_clearance = (
            lower_control_clearance[first_touch_sample, first_touch_joint]
            if first_touch_side == "lower"
            else upper_control_clearance[first_touch_sample, first_touch_joint]
        )
    else:
        first_touch_sample = None
        first_touch_joint = None
        first_touch_side = None
        first_touch_bound = None
        first_touch_clearance = None

    control_clearance = np.minimum(
        lower_control_clearance,
        upper_control_clearance,
    )
    finite_control_clearance = np.isfinite(control_clearance)
    if finite_control_clearance.any():
        comparable_control = np.where(
            finite_control_clearance,
            control_clearance,
            math.inf,
        )
        control_flat_index = int(np.argmin(comparable_control))
        min_control_sample, min_control_joint = np.unravel_index(
            control_flat_index,
            control_clearance.shape,
        )
        min_control_clearance: float | None = float(
            control_clearance[min_control_sample, min_control_joint]
        )
    else:
        min_control_clearance = None
        min_control_sample = None
        min_control_joint = None

    def frame_at_sample(sample_index: int | None) -> int | None:
        if sample_index is None or int(sample_index) == 0:
            return None
        return int(sample_index) - 1

    def scope_at_sample(sample_index: int | None) -> str | None:
        if sample_index is None:
            return None
        return "initial_state" if int(sample_index) == 0 else "trajectory_frame"

    def phase_at_sample(sample_index: int | None) -> str | None:
        frame_index = frame_at_sample(sample_index)
        return (
            None
            if phases is None or frame_index is None
            else str(phases[frame_index])
        )

    def optional_bounds(values: np.ndarray) -> list[float | None]:
        return [float(value) if math.isfinite(float(value)) else None for value in values]

    per_joint_min_hard_clearance: list[float | None] = []
    for joint_index in range(joint_count):
        values = hard_clearance[:, joint_index]
        finite = values[np.isfinite(values)]
        per_joint_min_hard_clearance.append(
            float(np.min(finite)) if finite.size else None
        )

    failed_checks: list[str] = []
    if np.any(hard_reserve_violations):
        failed_checks.append("hard_limit_clearance")
    if touch_indices.size:
        failed_checks.append("control_bound_touch")
    initial_hard_violations = hard_reserve_violations[0]
    trajectory_hard_violations = hard_reserve_violations[1:]
    initial_control_touches = control_touches[0]
    trajectory_control_touches = control_touches[1:]
    touch_sample_count = int(
        np.count_nonzero(np.any(control_touches, axis=1))
    )
    touch_frame_count = int(
        np.count_nonzero(np.any(trajectory_control_touches, axis=1))
    )
    return {
        "source": (
            "configured_initial_arm_joint_q_and_kinematic_reference_"
            "before_newton_physics"
        ),
        "constraint": "arm_joint_tracking_reserve",
        "sample_index_semantics": (
            "sample_0_is_configured_initial_state_then_sample_i_plus_1_"
            "is_trajectory_frame_i"
        ),
        "reference_command_realization": "round_to_nearest_float32",
        "initial_state_included": True,
        "initial_state_sample_index": 0,
        "initial_state_frame_index": None,
        "initial_state_phase": None,
        "trajectory_first_sample_index": 1,
        "control_bound_definition": (
            "float32_urdf_limit_plus_or_minus_margin_then_first_inward_float32"
        ),
        "control_bound_touch_definition": (
            "float32_realized_q_at_or_beyond_first_inward_float32_bound"
        ),
        "hard_limit_reserve_comparison": "strict_clearance_less_than_reserve",
        "arm_joint_tracking_reserve_rad": reserve,
        "control_limit_margin_rad": margin,
        "frame_count": int(reference.shape[0]),
        "reference_frame_count": int(reference.shape[0]),
        "audited_sample_count": int(realized_samples.shape[0]),
        "joint_count": joint_count,
        "arm_joint_names": list(names),
        "initial_state_joint_q_rad": realized_initial.tolist(),
        "initial_state_float32_roundtrip_max_abs_error_rad": float(
            np.max(np.abs(realized_initial - initial))
        ),
        "reference_float32_roundtrip_max_abs_error_rad": float(
            np.max(np.abs(realized_reference - reference))
        ),
        "hard_lower_rad": optional_bounds(hard_lower),
        "hard_upper_rad": optional_bounds(hard_upper),
        "control_lower_rad": optional_bounds(control_lower),
        "control_upper_rad": optional_bounds(control_upper),
        "per_joint_min_hard_limit_clearance_rad": per_joint_min_hard_clearance,
        "min_hard_limit_clearance_rad": min_hard_clearance,
        "min_hard_limit_clearance_sample_index": min_hard_sample,
        "min_hard_limit_clearance_sample_scope": scope_at_sample(
            min_hard_sample
        ),
        "min_hard_limit_clearance_frame_index": frame_at_sample(
            min_hard_sample
        ),
        "min_hard_limit_clearance_phase": phase_at_sample(min_hard_sample),
        "min_hard_limit_clearance_joint_index": min_hard_joint,
        "min_hard_limit_clearance_joint_name": (
            None if min_hard_joint is None else names[min_hard_joint]
        ),
        "min_hard_limit_clearance_side": hard_side,
        "min_hard_limit_clearance_joint_q_rad": (
            None
            if min_hard_joint is None or min_hard_sample is None
            else float(realized_samples[min_hard_sample, min_hard_joint])
        ),
        "hard_limit_reserve_violation_count": int(
            np.count_nonzero(hard_reserve_violations)
        ),
        "hard_limit_reserve_violation_sample_count": int(
            np.count_nonzero(np.any(hard_reserve_violations, axis=1))
        ),
        "hard_limit_reserve_violation_frame_count": int(
            np.count_nonzero(np.any(trajectory_hard_violations, axis=1))
        ),
        "initial_state_hard_limit_reserve_violation_count": int(
            np.count_nonzero(initial_hard_violations)
        ),
        "trajectory_hard_limit_reserve_violation_count": int(
            np.count_nonzero(trajectory_hard_violations)
        ),
        "min_control_bound_clearance_rad": min_control_clearance,
        "min_control_bound_clearance_sample_index": (
            None if min_control_sample is None else int(min_control_sample)
        ),
        "min_control_bound_clearance_sample_scope": scope_at_sample(
            min_control_sample
        ),
        "min_control_bound_clearance_frame_index": (
            frame_at_sample(min_control_sample)
        ),
        "min_control_bound_clearance_phase": phase_at_sample(
            min_control_sample
        ),
        "min_control_bound_clearance_joint_index": (
            None if min_control_joint is None else int(min_control_joint)
        ),
        "min_control_bound_clearance_joint_name": (
            None if min_control_joint is None else names[int(min_control_joint)]
        ),
        "control_bound_touch_count": int(np.count_nonzero(control_touches)),
        "control_bound_touch_sample_count": touch_sample_count,
        "control_bound_touch_frame_count": touch_frame_count,
        "initial_state_control_bound_touch_count": int(
            np.count_nonzero(initial_control_touches)
        ),
        "trajectory_control_bound_touch_count": int(
            np.count_nonzero(trajectory_control_touches)
        ),
        "first_control_bound_touch_sample_index": first_touch_sample,
        "first_control_bound_touch_sample_scope": scope_at_sample(
            first_touch_sample
        ),
        "first_control_bound_touch_frame_index": frame_at_sample(
            first_touch_sample
        ),
        "first_control_bound_touch_phase": phase_at_sample(
            first_touch_sample
        ),
        "first_control_bound_touch_joint_index": first_touch_joint,
        "first_control_bound_touch_joint_name": (
            None if first_touch_joint is None else names[first_touch_joint]
        ),
        "first_control_bound_touch_side": first_touch_side,
        "first_control_bound_touch_joint_q_rad": (
            None
            if first_touch_sample is None or first_touch_joint is None
            else float(realized_samples[first_touch_sample, first_touch_joint])
        ),
        "first_control_bound_touch_bound_rad": (
            None if first_touch_bound is None else float(first_touch_bound)
        ),
        "first_control_bound_touch_clearance_rad": (
            None if first_touch_clearance is None else float(first_touch_clearance)
        ),
        "failed_checks": failed_checks,
        "passed": not failed_checks,
    }


def _bounded_float32_velocity_targets(
    requested: np.ndarray, limits: np.ndarray
) -> np.ndarray:
    """Round finite in-limit velocities without crossing a URDF limit."""

    bounded = np.clip(np.asarray(requested, dtype=float), -limits, limits)
    float32_limits = np.asarray(limits, dtype=float).astype(np.float32)
    rounded_above = float32_limits.astype(float) > limits
    float32_limits[rounded_above] = np.nextafter(
        float32_limits[rounded_above], np.float32(0.0)
    )
    targets = bounded.astype(np.float32)
    return np.clip(targets, -float32_limits, float32_limits).astype(
        np.float32, copy=False
    )


def _initial_controlled_robot_position(
    velocity_limits: RobotJointVelocityLimits,
    initial_arm_joint_positions: Sequence[float] | np.ndarray,
    initial_gripper_width_m: float,
) -> np.ndarray:
    initial_arm = _finite_vector(
        initial_arm_joint_positions,
        len(velocity_limits.arm_joint_names),
        "initial_arm_joint_positions",
    )
    initial_width = _positive(
        initial_gripper_width_m,
        "initial_gripper_width_m",
        allow_zero=True,
    )
    return np.concatenate(
        (
            initial_arm,
            np.full(
                len(velocity_limits.finger_joint_names),
                0.5 * initial_width,
                dtype=float,
            ),
        )
    )


def _command_frame_context(
    commands: PhysicsCommandTrajectory, frame_index: int
) -> dict[str, Any]:
    index = int(frame_index)
    previous_frame = index - 1
    if index >= commands.frame_count:
        hold_index = index - commands.frame_count
        return {
            "previous_frame_index": previous_frame,
            "frame_index": index,
            "previous_phase": (
                str(commands.phase_names[-1])
                if hold_index == 0
                else "virtual_terminal_hold"
            ),
            "phase": "virtual_terminal_hold",
            "frame_kind": "virtual_terminal_hold",
            "virtual_terminal_hold_index": hold_index,
        }
    return {
        "previous_frame_index": previous_frame,
        "frame_index": index,
        "previous_phase": (
            "configured_initial_state"
            if previous_frame < 0
            else str(commands.phase_names[previous_frame])
        ),
        "phase": str(commands.phase_names[index]),
    }


def _raise_velocity_limit_violation(
    commands: PhysicsCommandTrajectory,
    velocity_limits: RobotJointVelocityLimits,
    deltas: np.ndarray,
    speeds: np.ndarray,
    utilization: np.ndarray,
    violating: np.ndarray,
    dt: float,
) -> None:
    names = velocity_limits.controlled_joint_names
    joint_types = velocity_limits.controlled_joint_types
    limits = velocity_limits.controlled_velocity_limits
    arm_count = len(velocity_limits.arm_joint_names)
    ranked = sorted(
        (
            (int(frame_index), int(joint_index))
            for frame_index, joint_index in np.argwhere(violating)
        ),
        key=lambda item: float(utilization[item]),
        reverse=True,
    )
    violations: list[dict[str, Any]] = []
    for frame_index, joint_index in ranked[:_MAX_REPORTED_MOTION_VIOLATIONS]:
        joint_type = joint_types[joint_index]
        unit = "m/s" if joint_type == "prismatic" else "rad/s"
        delta_unit = "m" if joint_type == "prismatic" else "rad"
        signed_delta = float(deltas[frame_index, joint_index])
        limit = float(limits[joint_index])
        violations.append(
            {
                **_command_frame_context(commands, frame_index),
                "joint": names[joint_index],
                "joint_kind": "arm" if joint_index < arm_count else "finger",
                "joint_type": joint_type,
                "target_delta": signed_delta,
                "target_delta_unit": delta_unit,
                "requested_velocity": signed_delta / dt,
                "requested_speed": float(speeds[frame_index, joint_index]),
                "velocity_limit": limit,
                "velocity_unit": unit,
                "requested_to_limit_ratio": float(
                    utilization[frame_index, joint_index]
                ),
                "required_transition_dt_s": abs(signed_delta) / limit,
            }
        )
    raise PipelineError(
        FailureCode.JOINT_LIMIT,
        "physics trajectory exceeds URDF joint velocity limits; refusing to compress the motion into one frame",
        stage="physics_trajectory",
        details={
            "limit_kind": "velocity",
            "source": "robot_urdf",
            "source_urdf": str(velocity_limits.source_urdf),
            "initial_state_included": True,
            "dt_s": dt,
            "violation_count": int(np.count_nonzero(violating)),
            "reported_violation_count": len(violations),
            "max_requested_to_limit_ratio": float(np.max(utilization)),
            "violations": violations,
        },
    )


def _raise_derivative_limit_violation(
    commands: PhysicsCommandTrajectory,
    values: np.ndarray,
    threshold: float,
    violating: np.ndarray,
    *,
    derivative: str,
    unit: str,
    joint_names: Sequence[str],
    joint_types: Sequence[str],
    joint_kind: str,
    source: str,
) -> None:
    utilization = np.abs(values) / threshold
    ranked = sorted(
        (
            (int(sample_index), int(joint_index))
            for sample_index, joint_index in np.argwhere(violating)
        ),
        key=lambda item: float(utilization[item]),
        reverse=True,
    )
    violations: list[dict[str, Any]] = []
    for sample_index, joint_index in ranked[:_MAX_REPORTED_MOTION_VIOLATIONS]:
        frame_index = sample_index
        signed_value = float(values[sample_index, joint_index])
        violations.append(
            {
                **_command_frame_context(commands, frame_index),
                "joint": str(joint_names[joint_index]),
                "joint_kind": joint_kind,
                "joint_type": str(joint_types[joint_index]),
                f"requested_{derivative}": signed_value,
                f"{derivative}_magnitude": abs(signed_value),
                f"{derivative}_limit": threshold,
                f"{derivative}_unit": unit,
                "requested_to_limit_ratio": float(
                    utilization[sample_index, joint_index]
                ),
            }
        )
    raise PipelineError(
        FailureCode.JOINT_LIMIT,
        f"physics trajectory exceeds configured {joint_kind} joint {derivative} limit",
        stage="physics_trajectory",
        details={
            "limit_kind": derivative,
            "source": source,
            "scope": f"{joint_kind}_joints",
            "initial_state_included": True,
            f"{derivative}_limit": threshold,
            f"{derivative}_unit": unit,
            "violation_count": int(np.count_nonzero(violating)),
            "reported_violation_count": len(violations),
            "max_requested_to_limit_ratio": float(np.max(utilization)),
            "violations": violations,
        },
    )


def validate_robot_command_motion_limits(
    commands: PhysicsCommandTrajectory,
    velocity_limits: RobotJointVelocityLimits,
    dt: float,
    *,
    initial_arm_joint_positions: Sequence[float] | np.ndarray,
    initial_gripper_width_m: float,
    max_joint_acceleration_rad_s2: float,
    max_joint_jerk_rad_s3: float,
    max_finger_acceleration_m_s2: float,
    max_finger_jerk_m_s3: float,
) -> dict[str, Any]:
    """Fail closed on velocity, acceleration, or jerk before Newton starts.

    The checked position sequence is exactly ``configured initial state`` plus
    every command frame.  Thus velocity sample zero is the real
    nominal/open-to-frame-zero target, not an invented zero target.  The
    derivative chain starts from Newton's real zero initial velocity and zero
    initial acceleration.  Revolute arm and prismatic finger derivatives use
    separate, dimensionally-correct configured acceleration and jerk limits;
    URDF velocity limits cover every controlled joint.
    """

    step = _positive(dt, "robot command dt")
    acceleration_limit = _positive(
        max_joint_acceleration_rad_s2,
        "max_joint_acceleration_rad_s2",
    )
    jerk_limit = _positive(max_joint_jerk_rad_s3, "max_joint_jerk_rad_s3")
    finger_acceleration_limit = _positive(
        max_finger_acceleration_m_s2,
        "max_finger_acceleration_m_s2",
    )
    finger_jerk_limit = _positive(
        max_finger_jerk_m_s3,
        "max_finger_jerk_m_s3",
    )
    command_targets = _controlled_robot_command_positions(commands, velocity_limits)
    initial_target = _initial_controlled_robot_position(
        velocity_limits,
        initial_arm_joint_positions,
        initial_gripper_width_m,
    )
    targets = np.vstack((initial_target, command_targets))
    limits = velocity_limits.controlled_velocity_limits
    if targets.shape[1] != len(limits):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "controlled robot target width does not match URDF velocity limits",
            stage="physics_trajectory",
        )

    deltas = np.diff(targets, axis=0)
    velocities = deltas / step
    speeds = np.abs(velocities)
    velocity_tolerances = _velocity_limit_tolerance(limits, step)
    velocity_violating = np.abs(deltas) > (limits * step + velocity_tolerances)
    velocity_utilization = speeds / limits[None, :]
    if np.any(velocity_violating):
        _raise_velocity_limit_violation(
            commands,
            velocity_limits,
            deltas,
            speeds,
            velocity_utilization,
            velocity_violating,
            step,
        )
    written_velocities = _bounded_float32_velocity_targets(velocities, limits)
    if np.any(np.abs(written_velocities.astype(float)) > limits[None, :]):
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "float32 robot velocity trajectory cannot be represented within URDF limits",
            stage="physics_trajectory",
        )

    arm_count = len(velocity_limits.arm_joint_names)
    arm_velocities = velocities[:, :arm_count]
    written_arm_velocities = written_velocities[:, :arm_count].astype(float)
    finger_velocities = velocities[:, arm_count:]
    if any(
        joint_type != "prismatic"
        for joint_type in velocity_limits.finger_joint_types
    ):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "finger acceleration and jerk thresholds in SI metres require prismatic finger joints",
            stage="physics_trajectory",
            details={
                "finger_joint_names": list(velocity_limits.finger_joint_names),
                "finger_joint_types": list(velocity_limits.finger_joint_types),
            },
        )
    # Newton starts from zero joint velocity and zero joint acceleration.
    # Preserve those actual initial conditions in the derivative chain so the
    # nominal-to-frame-zero target is checked at every derivative order.  Two
    # virtual holds at the final position then prove that the command can
    # return first to zero velocity and then to zero acceleration; without
    # them the last nonzero velocity would disappear outside the audit window.
    arm_derivative_velocities = np.vstack(
        (arm_velocities, np.zeros((2, arm_count), dtype=float))
    )
    arm_accelerations = np.diff(
        np.vstack(
            (np.zeros((1, arm_count), dtype=float), arm_derivative_velocities)
        ),
        axis=0,
    ) / step
    arm_jerks = np.diff(
        np.vstack((np.zeros((1, arm_count), dtype=float), arm_accelerations)),
        axis=0,
    ) / step
    written_arm_derivative_velocities = np.vstack(
        (written_arm_velocities, np.zeros((2, arm_count), dtype=float))
    )
    written_arm_accelerations = np.diff(
        np.vstack(
            (
                np.zeros((1, arm_count), dtype=float),
                written_arm_derivative_velocities,
            )
        ),
        axis=0,
    ) / step
    written_arm_jerks = np.diff(
        np.vstack(
            (np.zeros((1, arm_count), dtype=float), written_arm_accelerations)
        ),
        axis=0,
    ) / step
    arm_accelerations = np.where(
        np.abs(written_arm_accelerations) > np.abs(arm_accelerations),
        written_arm_accelerations,
        arm_accelerations,
    )
    arm_jerks = np.where(
        np.abs(written_arm_jerks) > np.abs(arm_jerks),
        written_arm_jerks,
        arm_jerks,
    )
    acceleration_violating = np.abs(arm_accelerations) > (
        acceleration_limit
        + 64.0 * np.finfo(float).eps * max(1.0, acceleration_limit)
    )
    if np.any(acceleration_violating):
        _raise_derivative_limit_violation(
            commands,
            arm_accelerations,
            acceleration_limit,
            acceleration_violating,
            derivative="acceleration",
            unit="rad/s^2",
            joint_names=velocity_limits.arm_joint_names,
            joint_types=velocity_limits.arm_joint_types,
            joint_kind="arm",
            source="thresholds.max_joint_acceleration_rad_s2",
        )
    jerk_violating = np.abs(arm_jerks) > (
        jerk_limit + 64.0 * np.finfo(float).eps * max(1.0, jerk_limit)
    )
    if np.any(jerk_violating):
        _raise_derivative_limit_violation(
            commands,
            arm_jerks,
            jerk_limit,
            jerk_violating,
            derivative="jerk",
            unit="rad/s^3",
            joint_names=velocity_limits.arm_joint_names,
            joint_types=velocity_limits.arm_joint_types,
            joint_kind="arm",
            source="thresholds.max_joint_jerk_rad_s3",
        )

    finger_count = len(velocity_limits.finger_joint_names)
    written_finger_velocities = written_velocities[:, arm_count:].astype(float)
    finger_derivative_velocities = np.vstack(
        (finger_velocities, np.zeros((2, finger_count), dtype=float))
    )
    finger_accelerations = np.diff(
        np.vstack(
            (
                np.zeros((1, finger_count), dtype=float),
                finger_derivative_velocities,
            )
        ),
        axis=0,
    ) / step
    finger_jerks = np.diff(
        np.vstack(
            (np.zeros((1, finger_count), dtype=float), finger_accelerations)
        ),
        axis=0,
    ) / step
    written_finger_derivative_velocities = np.vstack(
        (written_finger_velocities, np.zeros((2, finger_count), dtype=float))
    )
    written_finger_accelerations = np.diff(
        np.vstack(
            (
                np.zeros((1, finger_count), dtype=float),
                written_finger_derivative_velocities,
            )
        ),
        axis=0,
    ) / step
    written_finger_jerks = np.diff(
        np.vstack(
            (
                np.zeros((1, finger_count), dtype=float),
                written_finger_accelerations,
            )
        ),
        axis=0,
    ) / step
    finger_accelerations = np.where(
        np.abs(written_finger_accelerations) > np.abs(finger_accelerations),
        written_finger_accelerations,
        finger_accelerations,
    )
    finger_jerks = np.where(
        np.abs(written_finger_jerks) > np.abs(finger_jerks),
        written_finger_jerks,
        finger_jerks,
    )
    finger_acceleration_violating = np.abs(finger_accelerations) > (
        finger_acceleration_limit
        + 64.0
        * np.finfo(float).eps
        * max(1.0, finger_acceleration_limit)
    )
    if np.any(finger_acceleration_violating):
        _raise_derivative_limit_violation(
            commands,
            finger_accelerations,
            finger_acceleration_limit,
            finger_acceleration_violating,
            derivative="acceleration",
            unit="m/s^2",
            joint_names=velocity_limits.finger_joint_names,
            joint_types=velocity_limits.finger_joint_types,
            joint_kind="finger",
            source="thresholds.max_finger_acceleration_m_s2",
        )
    finger_jerk_violating = np.abs(finger_jerks) > (
        finger_jerk_limit
        + 64.0 * np.finfo(float).eps * max(1.0, finger_jerk_limit)
    )
    if np.any(finger_jerk_violating):
        _raise_derivative_limit_violation(
            commands,
            finger_jerks,
            finger_jerk_limit,
            finger_jerk_violating,
            derivative="jerk",
            unit="m/s^3",
            joint_names=velocity_limits.finger_joint_names,
            joint_types=velocity_limits.finger_joint_types,
            joint_kind="finger",
            source="thresholds.max_finger_jerk_m_s3",
        )

    def max_abs(values: np.ndarray) -> float:
        return float(np.max(np.abs(values))) if values.size else 0.0

    return {
        "source": "robot_urdf_and_project_thresholds",
        "source_urdf": str(velocity_limits.source_urdf),
        "scope": {
            "velocity": "arm_and_finger_joints",
            "acceleration": "arm_and_finger_joints_type_specific_si_limits",
            "jerk": "arm_and_finger_joints_type_specific_si_limits",
        },
        "initial_state_included": True,
        "float32_target_velocity_derivatives_included": True,
        "initial_joint_velocity_rad_s": 0.0,
        "initial_joint_acceleration_rad_s2": 0.0,
        "transition_count": commands.frame_count,
        "terminal_hold_sample_count": 2,
        "terminal_velocity_rad_s": 0.0,
        "terminal_acceleration_rad_s2": 0.0,
        "acceleration_sample_count": commands.frame_count + 2,
        "jerk_sample_count": commands.frame_count + 2,
        "max_requested_to_limit_ratio": float(np.max(velocity_utilization)),
        "max_arm_joint_velocity_rad_s": max_abs(arm_velocities),
        "max_arm_joint_acceleration_rad_s2": max_abs(arm_accelerations),
        "max_arm_joint_jerk_rad_s3": max_abs(arm_jerks),
        "max_finger_joint_velocity_m_s": max_abs(finger_velocities),
        "max_finger_joint_acceleration_m_s2": max_abs(finger_accelerations),
        "max_finger_joint_jerk_m_s3": max_abs(finger_jerks),
        "max_joint_acceleration_rad_s2": acceleration_limit,
        "max_joint_jerk_rad_s3": jerk_limit,
        "max_finger_acceleration_m_s2": finger_acceleration_limit,
        "max_finger_jerk_m_s3": finger_jerk_limit,
        "passed": True,
    }


def audit_bumpless_grasp_release_transfer(
    transfer: BumplessGraspReleaseTransfer,
    commands: PhysicsCommandTrajectory,
    velocity_limits: RobotJointVelocityLimits,
    hard_arm_lower_rad: Any,
    hard_arm_upper_rad: Any,
    *,
    initial_arm_joint_positions: Sequence[float] | np.ndarray,
    initial_gripper_width_m: float,
    control_limit_margin_rad: float,
    arm_joint_tracking_reserve_rad: float,
    dt: float,
    max_joint_acceleration_rad_s2: float,
    max_joint_jerk_rad_s3: float,
    max_finger_acceleration_m_s2: float,
    max_finger_jerk_m_s3: float,
) -> dict[str, Any]:
    """Audit the exact float32 release targets before advancing physics."""

    effective_commands = PhysicsCommandTrajectory(
        phase_names=commands.phase_names,
        arm_joint_targets=transfer.applied_arm_joint_targets,
        gripper_width_m=commands.gripper_width_m,
        door_reference_rad=commands.door_reference_rad,
        handle_frame_in_link=commands.handle_frame_in_link,
        expected_handle_to_tcp=commands.expected_handle_to_tcp,
    )
    position = validate_applied_arm_target_position_limits(
        transfer.applied_arm_joint_targets,
        velocity_limits.arm_joint_names,
        hard_arm_lower_rad,
        hard_arm_upper_rad,
        phase_names=commands.phase_names,
    )
    reserve = audit_arm_joint_reference_reserve(
        transfer.applied_arm_joint_targets,
        velocity_limits.arm_joint_names,
        hard_arm_lower_rad,
        hard_arm_upper_rad,
        initial_arm_q=initial_arm_joint_positions,
        control_limit_margin_rad=control_limit_margin_rad,
        arm_joint_tracking_reserve_rad=arm_joint_tracking_reserve_rad,
        phase_names=commands.phase_names,
    )
    if not bool(reserve["passed"]):
        raise PipelineError(
            FailureCode.IK_UNREACHABLE,
            "measured release equilibrium leaves insufficient arm joint tracking reserve",
            stage="ik_motion_limits",
            details={
                "feasibility_scope": (
                    "runtime_bumpless_release_applied_targets_before_"
                    "release_unload"
                ),
                **reserve,
            },
        )
    motion = validate_robot_command_motion_limits(
        effective_commands,
        velocity_limits,
        dt,
        initial_arm_joint_positions=initial_arm_joint_positions,
        initial_gripper_width_m=initial_gripper_width_m,
        max_joint_acceleration_rad_s2=max_joint_acceleration_rad_s2,
        max_joint_jerk_rad_s3=max_joint_jerk_rad_s3,
        max_finger_acceleration_m_s2=max_finger_acceleration_m_s2,
        max_finger_jerk_m_s3=max_finger_jerk_m_s3,
    )
    return {
        "passed": True,
        "position_limits": position,
        "tracking_reserve": reserve,
        "motion_limits": motion,
    }


def build_robot_velocity_targets(
    previous_arm_targets: Sequence[float] | np.ndarray,
    current_arm_targets: Sequence[float] | np.ndarray,
    previous_gripper_width_m: float,
    current_gripper_width_m: float,
    velocity_limits: RobotJointVelocityLimits,
    dt: float,
    *,
    frame_index: int | None = None,
) -> np.ndarray:
    """Build float32 Newton targets without exceeding the URDF limits.

    Material over-limit requests fail closed.  Clipping is used only after
    that check to keep a mathematically on-limit value from rounding above its
    limit when represented as Newton's float32 target.
    """

    step = _positive(dt, "robot command dt")
    previous_arm = _finite_vector(
        previous_arm_targets,
        len(velocity_limits.arm_joint_names),
        "previous_arm_targets",
    )
    current_arm = _finite_vector(
        current_arm_targets,
        len(velocity_limits.arm_joint_names),
        "current_arm_targets",
    )
    previous_width = _positive(
        previous_gripper_width_m, "previous_gripper_width_m", allow_zero=True
    )
    current_width = _positive(
        current_gripper_width_m, "current_gripper_width_m", allow_zero=True
    )
    requested = np.concatenate(
        (
            (current_arm - previous_arm) / step,
            np.full(
                len(velocity_limits.finger_joint_names),
                0.5 * (current_width - previous_width) / step,
                dtype=float,
            ),
        )
    )
    limits = velocity_limits.controlled_velocity_limits
    tolerance = _velocity_limit_tolerance(limits, step) / step
    over_limit = np.abs(requested) > limits + tolerance
    if np.any(over_limit):
        indices = np.flatnonzero(over_limit)
        raise PipelineError(
            FailureCode.JOINT_LIMIT,
            "robot velocity target exceeds its URDF limit",
            stage="physics_control",
            details={
                "frame_index": frame_index,
                "joints": [
                    velocity_limits.controlled_joint_names[int(index)]
                    for index in indices
                ],
                "requested_velocity": [
                    float(requested[int(index)]) for index in indices
                ],
                "velocity_limit": [float(limits[int(index)]) for index in indices],
            },
        )

    # Remove only the tiny tolerance admitted above, then choose a float32
    # bound whose represented value is itself no greater than the URDF limit.
    targets = _bounded_float32_velocity_targets(requested, limits)
    if np.any(np.abs(targets.astype(float)) > limits):
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "float32 robot velocity target cannot be represented within its URDF limit",
            stage="physics_control",
            details={"frame_index": frame_index},
        )
    return targets


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
    arm_joint_tracking_reserve_rad: float
    control_limit_margin_rad: float
    grasp_release_blend_frames: int
    max_joint_acceleration_rad_s2: float
    max_joint_jerk_rad_s3: float
    max_finger_acceleration_m_s2: float
    max_finger_jerk_m_s3: float
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
    grasp_activation_linear_velocity_tolerance_m_s: float
    grasp_activation_angular_velocity_tolerance_deg_s: float
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
        _positive_int(self.grasp_release_blend_frames, "grasp_release_blend_frames")
        if self.grasp_release_blend_frames < 2:
            raise ValueError("grasp_release_blend_frames must be at least 2")
        for name in (
            "arm_joint_tracking_reserve_rad",
            "control_limit_margin_rad",
            "max_joint_acceleration_rad_s2",
            "max_joint_jerk_rad_s3",
            "max_finger_acceleration_m_s2",
            "max_finger_jerk_m_s3",
            "arm_stiffness",
            "arm_damping",
            "finger_stiffness",
            "finger_damping",
            "grasp_activation_position_tolerance_m",
            "grasp_activation_orientation_tolerance_deg",
            "grasp_activation_linear_velocity_tolerance_m_s",
            "grasp_activation_angular_velocity_tolerance_deg_s",
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
            arm_joint_tracking_reserve_rad=float(
                control["arm_joint_tracking_reserve_rad"]
            ),
            control_limit_margin_rad=float(
                config.get("ik.control_limit_margin_rad")
            ),
            grasp_release_blend_frames=int(
                control["grasp_release_blend_frames"]
            ),
            max_joint_acceleration_rad_s2=float(
                config.get("thresholds.max_joint_acceleration_rad_s2")
            ),
            max_joint_jerk_rad_s3=float(
                config.get("thresholds.max_joint_jerk_rad_s3")
            ),
            max_finger_acceleration_m_s2=float(
                config.get("thresholds.max_finger_acceleration_m_s2")
            ),
            max_finger_jerk_m_s3=float(
                config.get("thresholds.max_finger_jerk_m_s3")
            ),
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
            grasp_activation_linear_velocity_tolerance_m_s=float(
                fixed["activation_linear_velocity_tolerance_m_s"]
            ),
            grasp_activation_angular_velocity_tolerance_deg_s=float(
                fixed["activation_angular_velocity_tolerance_deg_s"]
            ),
            device=str(config.get("runtime.device")),
        )


@dataclass(frozen=True, slots=True)
class PhysicsRollout:
    """Measured, audit-ready physics rollout."""

    phase_names: np.ndarray
    time_s: np.ndarray
    command_joint_q: np.ndarray
    applied_arm_joint_target_q: np.ndarray
    applied_arm_joint_target_qd: np.ndarray
    measured_joint_q: np.ndarray
    measured_joint_qd: np.ndarray
    measured_arm_joint_q: np.ndarray
    measured_arm_joint_qd: np.ndarray
    measured_finger_joint_qd: np.ndarray
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
    finger_joint_names: tuple[str, ...]
    metadata: Mapping[str, Any]

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "phase_names": self.phase_names,
            "time_s": self.time_s,
            "command_joint_q": self.command_joint_q,
            "applied_arm_joint_target_q": self.applied_arm_joint_target_q,
            "applied_arm_joint_target_qd": self.applied_arm_joint_target_qd,
            "measured_joint_q": self.measured_joint_q,
            "measured_joint_qd": self.measured_joint_qd,
            "measured_arm_joint_q": self.measured_arm_joint_q,
            "measured_arm_joint_qd": self.measured_arm_joint_qd,
            "measured_finger_joint_qd": self.measured_finger_joint_qd,
            "door_angle_rad": self.door_angle_rad,
            "ee_pose_wxyz": self.ee_pose_wxyz,
            "handle_link_pose_wxyz": self.handle_link_pose_wxyz,
            "body_pose_wxyz": self.body_pose_wxyz,
            "collision_flags": self.collision_flags,
            "grasp_constraint_active": self.grasp_constraint_active,
            "external_robot_joint_force_command": self.external_robot_joint_force_command,
            "finger_joint_names": np.asarray(self.finger_joint_names, dtype="U"),
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
    robot_initial_joint_velocity_zero_verified: bool
    robot_body_inverse_mass_positive_verified: bool
    robot_body_flags_dynamic_verified: bool
    planned_grasp_anchors: PlannedFixedGraspAnchors | None
    grasp_child_anchor_readback: FixedGraspChildAnchorReadback | None
    grasp_initial_enabled_writeback: FixedGraspJointEnabledWriteback | None
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


def write_fixed_grasp_parent_anchor(
    model: Any,
    joint_index: int,
    parent_xform: np.ndarray,
    *,
    warp_module: Any | None = None,
) -> FixedGraspAnchorWriteback:
    """Write and verify exactly one ``model.joint_X_p`` transform.

    The full array is read before and after the write so the helper also proves
    that no non-target joint anchor changed.  A failed or mismatched readback
    is reported as a structured constraint failure; when possible, the
    original array is restored before raising.

    ``warp_module`` is injectable solely so the array transaction can be
    exercised with deterministic fake arrays on machines without Warp.
    """

    backend = wp if warp_module is None else warp_module
    if backend is None:
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "Warp is unavailable for fixed-grasp anchor capture",
            stage="physics_constraint",
        )
    if (
        not isinstance(joint_index, (int, np.integer))
        or isinstance(joint_index, (bool, np.bool_))
    ):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "fixed-grasp joint index must be an integer",
            stage="physics_constraint",
            details={"joint_index": _json_safe(joint_index)},
        )
    index = int(joint_index)
    try:
        target_matrix = np.asarray(parent_xform, dtype=float)
        decompose_pose(target_matrix)
        expected_row = np.asarray(
            _newton_pose_row_from_matrix(target_matrix), dtype=np.float32
        )
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "captured fixed-grasp parent anchor is malformed",
            stage="physics_constraint",
            details={"joint_index": index, "error": str(exc)},
        ) from exc

    joint_x_p = getattr(model, "joint_X_p", None)
    if joint_x_p is None or not callable(getattr(joint_x_p, "numpy", None)) or not callable(
        getattr(joint_x_p, "assign", None)
    ):
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "Newton model has no readable/writable joint_X_p array",
            stage="physics_constraint",
            details={"joint_index": index},
        )
    try:
        original = np.asarray(joint_x_p.numpy(), dtype=np.float32).copy()
    except Exception as exc:
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "cannot read Newton fixed-joint parent anchors",
            stage="physics_constraint",
            details={
                "joint_index": index,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            },
        ) from exc
    if (
        original.ndim != 2
        or original.shape[1:] != (7,)
        or not np.isfinite(original).all()
        or not 0 <= index < len(original)
    ):
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "Newton joint_X_p layout is invalid for fixed-grasp capture",
            stage="physics_constraint",
            details={
                "joint_index": index,
                "joint_X_p_shape": list(original.shape),
                "joint_X_p_finite": bool(np.isfinite(original).all()),
            },
        )

    updated = original.copy()
    updated[index] = expected_row
    device = getattr(model, "device", None)

    def payload(rows: np.ndarray) -> Any:
        return backend.array(
            np.asarray(rows, dtype=np.float32),
            dtype=backend.transform,
            device=device,
        )

    def rollback() -> tuple[bool, bool, str | None]:
        try:
            joint_x_p.assign(payload(original))
            restored = np.asarray(joint_x_p.numpy(), dtype=np.float32)
            return True, bool(np.array_equal(restored, original)), None
        except Exception as rollback_exc:  # pragma: no cover - remote backend only
            return False, False, f"{type(rollback_exc).__name__}: {rollback_exc}"

    try:
        joint_x_p.assign(payload(updated))
    except Exception as exc:
        rollback_attempted, rollback_verified, rollback_error = rollback()
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "cannot write the captured Newton fixed-joint parent anchor",
            stage="physics_constraint",
            details={
                "joint_index": index,
                "grasp_anchor_runtime_model_write_attempt_count": 1,
                "grasp_anchor_runtime_model_write_count": 0,
                "rollback_attempted": rollback_attempted,
                "rollback_verified": rollback_verified,
                "rollback_error": rollback_error,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            },
        ) from exc

    try:
        readback = np.asarray(joint_x_p.numpy(), dtype=np.float32).copy()
    except Exception as exc:
        rollback_attempted, rollback_verified, rollback_error = rollback()
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "cannot read back the captured Newton fixed-joint parent anchor",
            stage="physics_constraint",
            details={
                "joint_index": index,
                "grasp_anchor_runtime_model_write_count": 1,
                "grasp_anchor_runtime_model_readback_count": 0,
                "rollback_attempted": rollback_attempted,
                "rollback_verified": rollback_verified,
                "rollback_error": rollback_error,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            },
        ) from exc
    if readback.shape != updated.shape or not np.array_equal(readback, updated):
        rollback_attempted, rollback_verified, rollback_error = rollback()
        differing_rows = (
            np.flatnonzero(np.any(readback != updated, axis=1)).tolist()
            if readback.shape == updated.shape
            else []
        )
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "Newton fixed-joint parent-anchor write-back verification failed",
            stage="physics_constraint",
            details={
                "joint_index": index,
                "expected_shape": list(updated.shape),
                "readback_shape": list(readback.shape),
                "differing_joint_rows": differing_rows,
                "grasp_anchor_runtime_model_write_count": 1,
                "grasp_anchor_runtime_model_readback_count": 1,
                "grasp_anchor_runtime_model_write_readback_verified": False,
                "rollback_attempted": rollback_attempted,
                "rollback_verified": rollback_verified,
                "rollback_error": rollback_error,
            },
        )
    try:
        realized_parent_xform = _matrix_from_newton_pose(readback[index])
    except PipelineError as exc:
        rollback_attempted, rollback_verified, rollback_error = rollback()
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "verified Newton parent-anchor row is not a rigid transform",
            stage="physics_constraint",
            details={
                "joint_index": index,
                "grasp_anchor_runtime_model_write_count": 1,
                "grasp_anchor_runtime_model_readback_count": 1,
                "rollback_attempted": rollback_attempted,
                "rollback_verified": rollback_verified,
                "rollback_error": rollback_error,
                "error": str(exc),
            },
        ) from exc
    return FixedGraspAnchorWriteback(
        parent_xform=realized_parent_xform,
        runtime_model_write_count=1,
        runtime_model_readback_count=1,
        readback_verified=True,
    )


def read_fixed_grasp_child_anchor(
    model: Any,
    joint_index: int,
    authored_child_xform: np.ndarray,
) -> FixedGraspChildAnchorReadback:
    """Read the finalized ``joint_X_c`` row and verify the authored anchor."""

    if (
        not isinstance(joint_index, (int, np.integer))
        or isinstance(joint_index, (bool, np.bool_))
    ):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "fixed-grasp joint index must be an integer",
            stage="physics_constraint",
            details={"joint_index": _json_safe(joint_index)},
        )
    index = int(joint_index)
    try:
        authored_matrix = np.asarray(authored_child_xform, dtype=float)
        decompose_pose(authored_matrix)
        expected_row = np.asarray(
            _newton_pose_row_from_matrix(authored_matrix), dtype=np.float32
        )
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "authored fixed-grasp child anchor is malformed",
            stage="physics_constraint",
            details={"joint_index": index, "error": str(exc)},
        ) from exc

    joint_x_c = getattr(model, "joint_X_c", None)
    if joint_x_c is None or not callable(getattr(joint_x_c, "numpy", None)):
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "Newton model has no readable joint_X_c array",
            stage="physics_constraint",
            details={"joint_index": index},
        )
    try:
        rows = np.asarray(joint_x_c.numpy(), dtype=np.float32).copy()
    except Exception as exc:
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "cannot read finalized Newton fixed-joint child anchors",
            stage="physics_constraint",
            details={
                "joint_index": index,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            },
        ) from exc
    if (
        rows.ndim != 2
        or rows.shape[1:] != (7,)
        or not np.isfinite(rows).all()
        or not 0 <= index < len(rows)
    ):
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "Newton joint_X_c layout is invalid for fixed-grasp capture",
            stage="physics_constraint",
            details={
                "joint_index": index,
                "joint_X_c_shape": list(rows.shape),
                "joint_X_c_finite": bool(np.isfinite(rows).all()),
            },
        )
    if not np.array_equal(rows[index], expected_row):
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "finalized Newton fixed-joint child anchor differs from the authored anchor",
            stage="physics_constraint",
            details={
                "joint_index": index,
                "expected_child_anchor_xyzw": expected_row.tolist(),
                "actual_child_anchor_xyzw": rows[index].tolist(),
                "grasp_anchor_child_runtime_model_readback_count": 1,
                "grasp_anchor_child_runtime_model_readback_verified": True,
                "grasp_anchor_child_authored_match_verified": False,
            },
        )
    realized_child_xform = _matrix_from_newton_pose(rows[index])
    return FixedGraspChildAnchorReadback(
        child_xform=realized_child_xform,
        runtime_model_readback_count=1,
        readback_verified=True,
        authored_match_verified=True,
    )


def write_fixed_grasp_joint_enabled(
    model: Any,
    joint_index: int,
    enabled: bool,
    *,
    warp_module: Any | None = None,
) -> FixedGraspJointEnabledWriteback:
    """Transactionally write and verify one ``joint_enabled`` entry."""

    backend = wp if warp_module is None else warp_module
    if backend is None:
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "Warp is unavailable for fixed-grasp joint-enabled writeback",
            stage="physics_constraint",
        )
    if (
        not isinstance(joint_index, (int, np.integer))
        or isinstance(joint_index, (bool, np.bool_))
        or not isinstance(enabled, (bool, np.bool_))
    ):
        raise PipelineError(
            FailureCode.CONFIG_INVALID,
            "fixed-grasp enabled transaction requires an integer index and boolean state",
            stage="physics_constraint",
            details={
                "joint_index": _json_safe(joint_index),
                "enabled": _json_safe(enabled),
            },
        )
    index = int(joint_index)
    desired_enabled = bool(enabled)
    joint_enabled = getattr(model, "joint_enabled", None)
    if joint_enabled is None or not callable(
        getattr(joint_enabled, "numpy", None)
    ) or not callable(getattr(joint_enabled, "assign", None)):
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "Newton model has no readable/writable joint_enabled array",
            stage="physics_constraint",
            details={"joint_index": index},
        )
    try:
        original = np.asarray(joint_enabled.numpy(), dtype=bool).copy()
    except Exception as exc:
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "cannot read Newton joint-enabled states",
            stage="physics_constraint",
            details={
                "joint_index": index,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            },
        ) from exc
    if original.ndim != 1 or not 0 <= index < len(original):
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "Newton joint_enabled layout is invalid",
            stage="physics_constraint",
            details={
                "joint_index": index,
                "joint_enabled_shape": list(original.shape),
            },
        )
    updated = original.copy()
    updated[index] = desired_enabled
    device = getattr(model, "device", None)

    def payload(values: np.ndarray) -> Any:
        return backend.array(values, dtype=bool, device=device)

    def rollback() -> tuple[bool, bool, str | None]:
        try:
            joint_enabled.assign(payload(original))
            restored = np.asarray(joint_enabled.numpy(), dtype=bool)
            return True, bool(np.array_equal(restored, original)), None
        except Exception as rollback_exc:  # pragma: no cover - remote backend only
            return False, False, f"{type(rollback_exc).__name__}: {rollback_exc}"

    try:
        joint_enabled.assign(payload(updated))
        readback = np.asarray(joint_enabled.numpy(), dtype=bool).copy()
    except Exception as exc:
        rollback_attempted, rollback_verified, rollback_error = rollback()
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "cannot write/read back the Newton fixed-grasp enabled state",
            stage="physics_constraint",
            details={
                "joint_index": index,
                "enabled": desired_enabled,
                "rollback_attempted": rollback_attempted,
                "rollback_verified": rollback_verified,
                "rollback_error": rollback_error,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            },
        ) from exc
    if readback.shape != updated.shape or not np.array_equal(readback, updated):
        rollback_attempted, rollback_verified, rollback_error = rollback()
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "Newton fixed-grasp enabled-state writeback verification failed",
            stage="physics_constraint",
            details={
                "joint_index": index,
                "enabled": desired_enabled,
                "expected_shape": list(updated.shape),
                "readback_shape": list(readback.shape),
                "runtime_model_write_count": 1,
                "runtime_model_readback_count": 1,
                "runtime_model_write_readback_verified": False,
                "rollback_attempted": rollback_attempted,
                "rollback_verified": rollback_verified,
                "rollback_error": rollback_error,
            },
        )
    return FixedGraspJointEnabledWriteback(
        enabled=desired_enabled,
        runtime_model_write_count=1,
        runtime_model_readback_count=1,
        readback_verified=True,
    )


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
    ) -> None:
        initial_arm = _finite_vector(
            self.parameters.nominal_arm_joint_positions,
            len(arm_refs),
            "nominal_arm_joint_positions",
        )
        initial_width = _positive(
            self.parameters.open_gripper_width_m,
            "open_gripper_width_m",
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
                        # relation for gating.  The finalized child anchor is
                        # verified and the parent anchor is captured at runtime.
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
                grasp_child_anchor_readback: (
                    FixedGraspChildAnchorReadback | None
                ) = None
                grasp_initial_enabled_writeback: (
                    FixedGraspJointEnabledWriteback | None
                ) = None
                if grasp_joint_index is not None:
                    grasp_joint_index = resolve_unique_label(
                        model.joint_label, _GRASP_JOINT_LABEL, kind="joint"
                    ).index
                    if planned_anchors is None:
                        raise PipelineError(
                            FailureCode.CONFIG_INVALID,
                            "fixed grasp has no planned anchors after finalization",
                            stage="physics_constraint",
                        )
                    grasp_child_anchor_readback = read_fixed_grasp_child_anchor(
                        model,
                        grasp_joint_index,
                        planned_anchors.child_xform,
                    )
                    # Verify the initially-disabled state through the same
                    # transactional write/readback path used at activation and
                    # release; never trust builder state without model proof.
                    grasp_initial_enabled_writeback = (
                        write_fixed_grasp_joint_enabled(
                            model, grasp_joint_index, False
                        )
                    )

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

                initial_joint_q = model.joint_q.numpy().astype(float, copy=True)
                controlled_initial_q = initial_joint_q[
                    list(final_controlled_coords)
                ]
                expected_initial_q = np.asarray(
                    [*params.nominal_arm_joint_positions]
                    + [
                        0.5 * params.open_gripper_width_m
                    ] * len(finger_refs),
                    dtype=np.float32,
                ).astype(float)
                if not (
                    np.isfinite(controlled_initial_q).all()
                    and np.array_equal(controlled_initial_q, expected_initial_q)
                ):
                    raise PipelineError(
                        FailureCode.ASSET_INVALID,
                        "finalized controlled robot coordinates differ from the configured nominal/open state",
                        stage="physics_model",
                        details={
                            "joint_names": list(
                                (*params.arm_joint_names, *params.finger_joint_names)
                            ),
                            "expected_initial_joint_q": expected_initial_q.tolist(),
                            "actual_initial_joint_q": controlled_initial_q.tolist(),
                        },
                    )

                initial_joint_qd = model.joint_qd.numpy().astype(
                    float, copy=False
                )
                controlled_initial_qd = initial_joint_qd[
                    list(final_controlled_dofs)
                ]
                robot_initial_joint_velocity_zero_verified = bool(
                    np.isfinite(controlled_initial_qd).all()
                    and np.all(controlled_initial_qd == 0.0)
                )
                if not robot_initial_joint_velocity_zero_verified:
                    raise PipelineError(
                        FailureCode.ASSET_INVALID,
                        "controlled robot joints do not start at zero velocity",
                        stage="physics_model",
                        details={
                            "joint_names": list(
                                (*params.arm_joint_names, *params.finger_joint_names)
                            ),
                            "initial_joint_velocity": controlled_initial_qd.tolist(),
                        },
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
                    initial_joint_q=initial_joint_q,
                    collapsed_robot_fixed_joints=collapsed_robot_fixed_joints,
                    controlled_coord_indices=final_controlled_coords,
                    controlled_dof_indices=final_controlled_dofs,
                    controlled_coord_indices_device=controlled_coord_indices_device,
                    controlled_dof_indices_device=controlled_dof_indices_device,
                    initial_door_target_q=initial_door_target_q,
                    initial_door_target_qd=initial_door_target_qd,
                    authored_door_target_ke=authored_door_target_ke,
                    authored_door_target_kd=authored_door_target_kd,
                    robot_initial_joint_velocity_zero_verified=(
                        robot_initial_joint_velocity_zero_verified
                    ),
                    robot_body_inverse_mass_positive_verified=(
                        robot_body_inverse_mass_positive_verified
                    ),
                    robot_body_flags_dynamic_verified=(
                        robot_body_flags_dynamic_verified
                    ),
                    planned_grasp_anchors=planned_anchors,
                    grasp_child_anchor_readback=grasp_child_anchor_readback,
                    grasp_initial_enabled_writeback=(
                        grasp_initial_enabled_writeback
                    ),
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
    def _set_joint_enabled(
        runtime: _PhysicsRuntime, joint_index: int, enabled: bool
    ) -> FixedGraspJointEnabledWriteback:
        return write_fixed_grasp_joint_enabled(
            runtime.model, joint_index, enabled
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
        velocity_limits = load_robot_joint_velocity_limits(
            params.robot_urdf,
            params.arm_joint_names,
            params.finger_joint_names,
        )
        hard_arm_lower, hard_arm_upper = load_robot_arm_joint_position_limits(
            params.robot_urdf,
            params.arm_joint_names,
        )
        planned_motion_limit_audit = validate_robot_command_motion_limits(
            commands,
            velocity_limits,
            params.dt,
            initial_arm_joint_positions=params.nominal_arm_joint_positions,
            initial_gripper_width_m=params.open_gripper_width_m,
            max_joint_acceleration_rad_s2=(
                params.max_joint_acceleration_rad_s2
            ),
            max_joint_jerk_rad_s3=params.max_joint_jerk_rad_s3,
            max_finger_acceleration_m_s2=(
                params.max_finger_acceleration_m_s2
            ),
            max_finger_jerk_m_s3=params.max_finger_jerk_m_s3,
        )
        applied_motion_limit_audit = planned_motion_limit_audit
        applied_arm_targets = commands.arm_joint_targets.astype(np.float32)
        applied_position_limit_audit = validate_applied_arm_target_position_limits(
            applied_arm_targets,
            params.arm_joint_names,
            hard_arm_lower,
            hard_arm_upper,
            phase_names=commands.phase_names,
        )
        applied_reserve_audit = audit_arm_joint_reference_reserve(
            applied_arm_targets,
            params.arm_joint_names,
            hard_arm_lower,
            hard_arm_upper,
            initial_arm_q=params.nominal_arm_joint_positions,
            control_limit_margin_rad=params.control_limit_margin_rad,
            arm_joint_tracking_reserve_rad=(
                params.arm_joint_tracking_reserve_rad
            ),
            phase_names=commands.phase_names,
        )
        if not bool(applied_reserve_audit["passed"]):
            raise PipelineError(
                FailureCode.IK_UNREACHABLE,
                "planned arm targets lack the reserve required for physics tracking",
                stage="ik_motion_limits",
                details={
                    "feasibility_scope": "initial_applied_arm_targets_before_physics",
                    **applied_reserve_audit,
                },
            )
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
        applied_arm_target_qd = np.empty(
            (frame_count, len(runtime.arm_refs)), dtype=np.float32
        )
        measured_q = np.empty((frame_count, coord_count), dtype=float)
        measured_qd = np.empty((frame_count, dof_count), dtype=float)
        arm_q = np.empty((frame_count, len(runtime.arm_refs)), dtype=float)
        arm_qd = np.empty((frame_count, len(runtime.arm_refs)), dtype=float)
        finger_qd = np.empty(
            (frame_count, len(runtime.finger_refs)), dtype=float
        )
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
        activation_capture: FixedGraspRuntimeCapture | None = None
        activation_enabled_writeback: (
            FixedGraspJointEnabledWriteback | None
        ) = None
        release_enabled_writeback: FixedGraspJointEnabledWriteback | None = None
        release_transfer: BumplessGraspReleaseTransfer | None = None
        release_unload_start_frame = (
            int(np.flatnonzero(commands.phase_names == "release")[0])
            if grasp_window is not None
            and np.any(commands.phase_names == "release")
            else None
        )
        substep_dt = params.dt / params.substeps

        try:
            for frame_index in range(frame_count):
                should_enable = bool(
                    grasp_window is not None and grasp_window.is_active(frame_index)
                )
                if (
                    grasp_window is not None
                    and release_transfer is None
                    and frame_index == release_unload_start_frame
                ):
                    if not should_enable or not fixed_enabled:
                        raise PipelineError(
                            FailureCode.NUMERICAL_INSTABILITY,
                            "fixed grasp is not active at the pre-release equilibrium capture",
                            stage="physics_constraint",
                            details={"frame_index": frame_index},
                        )
                    # Capture the prior post-step measured generalized arm state
                    # while the constraint is still enabled.  This is the PD
                    # equilibrium used both at the active release endpoint and
                    # the disabled retreat start; no planned-to-measured target
                    # jump is deferred to the disable transaction.
                    newton.eval_ik(
                        runtime.model,
                        runtime.state_in,
                        runtime.measured_q,
                        runtime.measured_qd,
                    )
                    prior_measured_q = (
                        runtime.measured_q.numpy().astype(float, copy=True)
                    )
                    captured_equilibrium = np.asarray(
                        prior_measured_q[
                            [ref.coord_index for ref in runtime.arm_refs]
                        ],
                        dtype=np.float32,
                    )
                    release_transfer = build_bumpless_grasp_release_transfer(
                        commands.phase_names,
                        commands.arm_joint_targets,
                        captured_equilibrium,
                        grasp_window,
                        params.grasp_release_blend_frames,
                    )
                    applied_arm_targets = (
                        release_transfer.applied_arm_joint_targets
                    )
                    effective_commands = PhysicsCommandTrajectory(
                        phase_names=commands.phase_names,
                        arm_joint_targets=applied_arm_targets,
                        gripper_width_m=commands.gripper_width_m,
                        door_reference_rad=commands.door_reference_rad,
                        handle_frame_in_link=commands.handle_frame_in_link,
                        expected_handle_to_tcp=commands.expected_handle_to_tcp,
                    )
                    applied_position_limit_audit = (
                        validate_applied_arm_target_position_limits(
                            applied_arm_targets,
                            params.arm_joint_names,
                            hard_arm_lower,
                            hard_arm_upper,
                            phase_names=commands.phase_names,
                        )
                    )
                    applied_reserve_audit = audit_arm_joint_reference_reserve(
                        applied_arm_targets,
                        params.arm_joint_names,
                        hard_arm_lower,
                        hard_arm_upper,
                        initial_arm_q=params.nominal_arm_joint_positions,
                        control_limit_margin_rad=(
                            params.control_limit_margin_rad
                        ),
                        arm_joint_tracking_reserve_rad=(
                            params.arm_joint_tracking_reserve_rad
                        ),
                        phase_names=commands.phase_names,
                    )
                    if not bool(applied_reserve_audit["passed"]):
                        raise PipelineError(
                            FailureCode.IK_UNREACHABLE,
                            "measured release equilibrium leaves insufficient arm joint tracking reserve",
                            stage="ik_motion_limits",
                            details={
                                "feasibility_scope": (
                                    "runtime_bumpless_release_applied_targets_"
                                    "before_release_unload"
                                ),
                                **applied_reserve_audit,
                            },
                        )
                    applied_motion_limit_audit = (
                        validate_robot_command_motion_limits(
                            effective_commands,
                            velocity_limits,
                            params.dt,
                            initial_arm_joint_positions=(
                                params.nominal_arm_joint_positions
                            ),
                            initial_gripper_width_m=(
                                params.open_gripper_width_m
                            ),
                            max_joint_acceleration_rad_s2=(
                                params.max_joint_acceleration_rad_s2
                            ),
                            max_joint_jerk_rad_s3=(
                                params.max_joint_jerk_rad_s3
                            ),
                            max_finger_acceleration_m_s2=(
                                params.max_finger_acceleration_m_s2
                            ),
                            max_finger_jerk_m_s3=(
                                params.max_finger_jerk_m_s3
                            ),
                        )
                    )
                if runtime.grasp_joint_index is not None and should_enable != fixed_enabled:
                    if should_enable:
                        if (
                            runtime.planned_grasp_anchors is None
                            or runtime.grasp_child_anchor_readback is None
                        ):
                            raise PipelineError(
                                FailureCode.CONFIG_INVALID,
                                "fixed grasp lacks planned/finalized anchor evidence",
                                stage="physics_constraint",
                            )
                        body_q_before_enable = (
                            runtime.state_in.body_q.numpy().astype(float, copy=True)
                        )
                        body_qd_before_enable = (
                            runtime.state_in.body_qd.numpy().astype(float, copy=True)
                        )
                        body_com = runtime.model.body_com.numpy().astype(
                            float, copy=True
                        )
                        parent_world_before_enable = _matrix_from_newton_pose(
                            body_q_before_enable[
                                runtime.end_effector_body_index
                            ]
                        )
                        child_world_before_enable = _matrix_from_newton_pose(
                            body_q_before_enable[runtime.handle_body_index]
                        )
                        finalized_anchors = PlannedFixedGraspAnchors(
                            parent_xform=(
                                runtime.planned_grasp_anchors.parent_xform
                            ),
                            child_xform=(
                                runtime.grasp_child_anchor_readback.child_xform
                            ),
                        )
                        activation_gate = evaluate_fixed_grasp_activation_gate(
                            parent_world_before_enable,
                            child_world_before_enable,
                            finalized_anchors,
                            parent_body_qd=body_qd_before_enable[
                                runtime.end_effector_body_index
                            ],
                            child_body_qd=body_qd_before_enable[
                                runtime.handle_body_index
                            ],
                            parent_body_com=body_com[
                                runtime.end_effector_body_index
                            ],
                            child_body_com=body_com[runtime.handle_body_index],
                            parent_body_name=str(
                                runtime.model.body_label[
                                    runtime.end_effector_body_index
                                ]
                            ),
                            child_body_name=str(
                                runtime.model.body_label[runtime.handle_body_index]
                            ),
                            frame_index=frame_index,
                            position_limit_m=(
                                params.grasp_activation_position_tolerance_m
                            ),
                            orientation_limit_deg=(
                                params.grasp_activation_orientation_tolerance_deg
                            ),
                            linear_velocity_limit_m_s=(
                                params.grasp_activation_linear_velocity_tolerance_m_s
                            ),
                            angular_velocity_limit_deg_s=(
                                params.grasp_activation_angular_velocity_tolerance_deg_s
                            ),
                        )
                        if not activation_gate.passed:
                            raise PipelineError(
                                FailureCode.IK_UNREACHABLE,
                                "fixed grasp activation rejected: pose or relative anchor twist is unsafe",
                                stage="physics_constraint",
                                details=activation_gate.audit_metadata(),
                            )
                        captured_parent_xform = captured_fixed_grasp_parent_anchor(
                            parent_world_before_enable,
                            child_world_before_enable,
                            runtime.grasp_child_anchor_readback.child_xform,
                        )
                        writeback = write_fixed_grasp_parent_anchor(
                            runtime.model,
                            runtime.grasp_joint_index,
                            captured_parent_xform,
                        )
                        post_capture_position_error, post_capture_orientation_error = (
                            _anchor_pose_error(
                                compose_transforms(
                                    parent_world_before_enable,
                                    writeback.parent_xform,
                                ),
                                compose_transforms(
                                    child_world_before_enable,
                                    runtime.grasp_child_anchor_readback.child_xform,
                                ),
                            )
                        )
                        activation_capture = FixedGraspRuntimeCapture(
                            frame_index=frame_index,
                            parent_xform=writeback.parent_xform,
                            child_xform=(
                                runtime.grasp_child_anchor_readback.child_xform
                            ),
                            runtime_model_write_count=(
                                writeback.runtime_model_write_count
                            ),
                            runtime_model_readback_count=(
                                writeback.runtime_model_readback_count
                            ),
                            readback_verified=writeback.readback_verified,
                            post_capture_position_error_m=(
                                post_capture_position_error
                            ),
                            post_capture_orientation_error_deg=(
                                post_capture_orientation_error
                            ),
                        )
                        if (
                            post_capture_position_error
                            > _CAPTURE_POSITION_TOLERANCE_M
                            or post_capture_orientation_error
                            > _CAPTURE_ORIENTATION_TOLERANCE_DEG
                        ):
                            raise PipelineError(
                                FailureCode.NUMERICAL_INSTABILITY,
                                "captured fixed-grasp anchors are not coincident after write-back",
                                stage="physics_constraint",
                                details={
                                    **activation_gate.audit_metadata(),
                                    **activation_capture.audit_metadata(),
                                    "grasp_anchor_post_capture_position_limit_m": (
                                        _CAPTURE_POSITION_TOLERANCE_M
                                    ),
                                    "grasp_anchor_post_capture_orientation_limit_deg": (
                                        _CAPTURE_ORIENTATION_TOLERANCE_DEG
                                    ),
                                },
                            )
                    enabled_writeback = self._set_joint_enabled(
                        runtime, runtime.grasp_joint_index, should_enable
                    )
                    if should_enable:
                        activation_enabled_writeback = enabled_writeback
                    else:
                        release_enabled_writeback = enabled_writeback
                    fixed_enabled = enabled_writeback.enabled
                constraint_active[frame_index] = fixed_enabled

                desired_q = build_robot_position_targets(
                    runtime.initial_joint_q,
                    runtime.arm_refs,
                    runtime.finger_refs,
                    applied_arm_targets[frame_index],
                    float(commands.gripper_width_m[frame_index]),
                    protected_coord_indices=(runtime.door_ref.coord_index,),
                )
                command_q[frame_index] = desired_q

                if frame_index == 0:
                    previous_arm = np.asarray(
                        params.nominal_arm_joint_positions, dtype=float
                    )
                    previous_width = params.open_gripper_width_m
                else:
                    previous_arm = applied_arm_targets[frame_index - 1]
                    previous_width = float(
                        commands.gripper_width_m[frame_index - 1]
                    )
                current_arm = applied_arm_targets[frame_index]
                current_width = float(commands.gripper_width_m[frame_index])
                controlled_velocity_targets = build_robot_velocity_targets(
                    previous_arm,
                    current_arm,
                    previous_width,
                    current_width,
                    velocity_limits,
                    params.dt,
                    frame_index=frame_index,
                )
                applied_arm_target_qd[frame_index] = (
                    controlled_velocity_targets[: len(runtime.arm_refs)]
                )
                if len(controlled_velocity_targets) != len(
                    runtime.controlled_dof_indices
                ):
                    raise PipelineError(
                        FailureCode.CONFIG_INVALID,
                        "URDF velocity limits do not match finalized controlled robot DoFs",
                        stage="physics_control",
                        details={
                            "velocity_limit_count": len(controlled_velocity_targets),
                            "controlled_dof_count": len(
                                runtime.controlled_dof_indices
                            ),
                        },
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

                written_control_q = (
                    runtime.control.joint_target_q.numpy()
                    .astype(np.float32, copy=False)[
                        list(runtime.controlled_coord_indices)
                    ]
                )
                written_control_qd = (
                    runtime.control.joint_target_qd.numpy()
                    .astype(np.float32, copy=False)[
                        list(runtime.controlled_dof_indices)
                    ]
                )
                if not np.array_equal(
                    written_control_q, controlled_position_targets
                ):
                    raise PipelineError(
                        FailureCode.NUMERICAL_INSTABILITY,
                        "Newton joint target position write-back differs from the applied float32 command",
                        stage="physics_control",
                        details={"frame_index": frame_index},
                    )
                if not np.array_equal(
                    written_control_qd, controlled_velocity_targets
                ) or np.any(
                    np.abs(written_control_qd.astype(float))
                    > velocity_limits.controlled_velocity_limits
                ):
                    raise PipelineError(
                        FailureCode.NUMERICAL_INSTABILITY,
                        "Newton joint target velocity write-back differs from the bounded float32 command",
                        stage="physics_control",
                        details={"frame_index": frame_index},
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
                arm_qd[frame_index] = qd_np[
                    [ref.dof_index for ref in runtime.arm_refs]
                ]
                finger_qd[frame_index] = qd_np[
                    [ref.dof_index for ref in runtime.finger_refs]
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

        fixed_grasp_evidence_complete = (
            activation_capture is not None
            and release_transfer is not None
            and runtime.grasp_child_anchor_readback is not None
            and runtime.grasp_initial_enabled_writeback is not None
            and activation_enabled_writeback is not None
            and release_enabled_writeback is not None
            and runtime.grasp_initial_enabled_writeback.enabled is False
            and activation_enabled_writeback.enabled is True
            and release_enabled_writeback.enabled is False
            and fixed_enabled is False
            and grasp_window is not None
            and release_transfer.retreat_rejoin_start_frame
            == grasp_window.release_frame
            and np.array_equal(
                release_transfer.applied_arm_joint_targets[
                    release_transfer.release_unload_end_frame
                ],
                release_transfer.captured_equilibrium_arm_q,
            )
            and np.array_equal(
                release_transfer.applied_arm_joint_targets[
                    release_transfer.retreat_rejoin_start_frame
                ],
                release_transfer.captured_equilibrium_arm_q,
            )
        )
        if params.fixed_grasp_enabled and not fixed_grasp_evidence_complete:
            raise PipelineError(
                FailureCode.NUMERICAL_INSTABILITY,
                "fixed-grasp rollout lacks verified capture/enable/release evidence",
                stage="physics_constraint",
                details={
                    "grasp_activation_frame": (
                        grasp_window.activation_frame
                        if grasp_window is not None
                        else None
                    ),
                    "grasp_anchor_runtime_model_write_readback_verified": False,
                    "grasp_joint_initial_disabled_verified": bool(
                        runtime.grasp_initial_enabled_writeback is not None
                        and not runtime.grasp_initial_enabled_writeback.enabled
                    ),
                    "grasp_joint_activation_enabled_verified": bool(
                        activation_enabled_writeback is not None
                        and activation_enabled_writeback.enabled
                    ),
                    "grasp_joint_release_disabled_verified": bool(
                        release_enabled_writeback is not None
                        and not release_enabled_writeback.enabled
                    ),
                    "remote_latch_allowed": False,
                    "remote_latch_prevention": _REMOTE_LATCH_PREVENTION,
                },
            )

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
            "simulation_dt_s": params.dt,
            "control_backend": "joint_pd",
            "robot_control_implementation": params.robot_control_implementation,
            "arm_joint_tracking_reserve_rad": (
                params.arm_joint_tracking_reserve_rad
            ),
            "control_limit_margin_rad": params.control_limit_margin_rad,
            "robot_control_semantics": "dynamic_robot_newton_xpbd_joint_position_velocity_pd",
            "robot_joint_coordinate_semantics": "newton_eval_ik_measured_dynamic_state",
            "measured_finger_joint_velocity_source": (
                "newton_eval_ik_post_step_name_resolved_joint_qd"
            ),
            "measured_finger_joint_names": [
                ref.configured_name for ref in runtime.finger_refs
            ],
            "door_coordinate_measurement_semantics": "newton_eval_ik_from_dynamic_body_state",
            "robot_body_state_write_backend": "none",
            "robot_body_state_runtime_write_count": 0,
            "robot_target_write_backend": "indexed_scatter_controlled_robot_coordinates_and_dofs_only",
            "joint_force_write_backend": "none",
            "external_robot_joint_force_command_semantics": "all_zero_no_joint_f_writes",
            "joint_pd_controller_used": True,
            "torque_pd_controller_used": False,
            "target_velocity_mode": params.target_velocity_mode,
            "grasp_release_blend_frames": params.grasp_release_blend_frames,
            "robot_target_velocity_limit_source": "robot_urdf",
            "robot_target_velocity_limit_urdf": str(
                velocity_limits.source_urdf
            ),
            "robot_target_velocity_limits_enforced": True,
            "robot_target_velocity_write_readback_verified": True,
            "robot_target_velocity_write_readback_frame_count": frame_count,
            "robot_target_velocity_limit_preflight_passed": bool(
                applied_motion_limit_audit["passed"]
            ),
            "robot_target_velocity_max_utilization": float(
                applied_motion_limit_audit["max_requested_to_limit_ratio"]
            ),
            "robot_command_motion_preflight": dict(applied_motion_limit_audit),
            "robot_planned_command_motion_preflight": dict(
                planned_motion_limit_audit
            ),
            "robot_applied_arm_target_position_limit_preflight": dict(
                applied_position_limit_audit
            ),
            "robot_applied_arm_target_reserve_preflight": dict(
                applied_reserve_audit
            ),
            "robot_applied_arm_target_source": (
                "runtime_bumpless_release_transfer_float32"
                if release_transfer is not None
                else "planned_reference_realized_float32"
            ),
            "robot_applied_arm_target_velocity_source": (
                "finite_difference_of_applied_float32_frame_targets"
            ),
            "robot_applied_arm_target_position_write_readback_verified": True,
            "robot_applied_arm_target_velocity_write_readback_verified": True,
            "robot_applied_arm_target_write_readback_frame_count": frame_count,
            "robot_command_motion_initial_state_included": True,
            "robot_initial_joint_velocity_zero_verified": (
                runtime.robot_initial_joint_velocity_zero_verified
            ),
            "robot_initial_joint_acceleration_semantics": "zero_before_first_control_step",
            "robot_command_acceleration_limit_rad_s2": (
                params.max_joint_acceleration_rad_s2
            ),
            "robot_command_jerk_limit_rad_s3": params.max_joint_jerk_rad_s3,
            "finger_command_acceleration_limit_m_s2": (
                params.max_finger_acceleration_m_s2
            ),
            "finger_command_jerk_limit_m_s3": params.max_finger_jerk_m_s3,
            "robot_command_max_arm_velocity_rad_s": float(
                applied_motion_limit_audit["max_arm_joint_velocity_rad_s"]
            ),
            "robot_command_max_arm_acceleration_rad_s2": float(
                applied_motion_limit_audit["max_arm_joint_acceleration_rad_s2"]
            ),
            "robot_command_max_arm_jerk_rad_s3": float(
                applied_motion_limit_audit["max_arm_joint_jerk_rad_s3"]
            ),
            "robot_command_max_finger_velocity_m_s": float(
                applied_motion_limit_audit["max_finger_joint_velocity_m_s"]
            ),
            "robot_command_max_finger_acceleration_m_s2": float(
                applied_motion_limit_audit["max_finger_joint_acceleration_m_s2"]
            ),
            "robot_command_max_finger_jerk_m_s3": float(
                applied_motion_limit_audit["max_finger_joint_jerk_m_s3"]
            ),
            "arm_joint_velocity_limits": {
                name: float(limit)
                for name, limit in zip(
                    velocity_limits.arm_joint_names,
                    velocity_limits.arm_velocity_limits,
                    strict=True,
                )
            },
            "finger_joint_velocity_limits": {
                name: float(limit)
                for name, limit in zip(
                    velocity_limits.finger_joint_names,
                    velocity_limits.finger_velocity_limits,
                    strict=True,
                )
            },
            "arm_stiffness": params.arm_stiffness,
            "arm_damping": params.arm_damping,
            "finger_stiffness": params.finger_stiffness,
            "finger_damping": params.finger_damping,
            "robot_scene_initialization": "configured_nominal_arm_and_open_gripper",
            "robot_initial_joint_position_config_verified": True,
            "robot_initial_arm_joint_positions": list(
                params.nominal_arm_joint_positions
            ),
            "robot_initial_gripper_width_m": params.open_gripper_width_m,
            "frame_zero_control_semantics": "interpolate_configured_initial_state_to_first_command_over_one_dt",
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
                "newton_fixed_loop_joint_planned_gate_with_measured_capture"
                if params.fixed_grasp_enabled
                else "disabled"
            ),
            "grasp_activation_frame": (
                grasp_window.activation_frame if grasp_window is not None else None
            ),
            "grasp_release_frame": (
                grasp_window.release_frame if grasp_window is not None else None
            ),
            "grasp_activation_linear_velocity_tolerance_m_s": (
                params.grasp_activation_linear_velocity_tolerance_m_s
            ),
            "grasp_activation_angular_velocity_tolerance_deg_s": (
                params.grasp_activation_angular_velocity_tolerance_deg_s
            ),
            "pose_layout": "xyz_wxyz",
            "handle_pose_semantics": "configured_handle_link_body_frame",
            "collision_semantics": "cross-asset signed effective-surface clearance at_or_below_configured_margin",
            "collision_margin_m": params.collision_margin_m,
            "collision_clearance_reported": True,
            "collision_evidence_scope": "cross_asset_robot_object",
        }
        if release_transfer is not None:
            metadata.update(release_transfer.audit_metadata())
            metadata.update(
                {
                    "grasp_release_applied_motion_limit_preflight_passed": bool(
                        applied_motion_limit_audit["passed"]
                    ),
                    "grasp_release_applied_position_limit_preflight_passed": bool(
                        applied_position_limit_audit["passed"]
                    ),
                    "grasp_release_applied_reserve_preflight_passed": bool(
                        applied_reserve_audit["passed"]
                    ),
                    "grasp_release_constraint_active_during_unload_verified": bool(
                        np.all(
                            constraint_active[
                                release_transfer.release_unload_start_frame :
                                release_transfer.release_unload_end_frame + 1
                            ]
                        )
                    ),
                    "grasp_release_constraint_disabled_during_rejoin_verified": bool(
                        not np.any(
                            constraint_active[
                                release_transfer.retreat_rejoin_start_frame :
                                release_transfer.retreat_rejoin_end_frame + 1
                            ]
                        )
                    ),
                    "grasp_release_target_position_continuity_at_disable_verified": bool(
                        np.array_equal(
                            applied_arm_targets[
                                release_transfer.release_unload_end_frame
                            ],
                            applied_arm_targets[
                                release_transfer.retreat_rejoin_start_frame
                            ],
                        )
                    ),
                    "grasp_release_target_velocity_before_disable_rad_s": (
                        applied_arm_target_qd[
                            release_transfer.release_unload_end_frame
                        ].tolist()
                    ),
                    "grasp_release_target_velocity_at_disable_rad_s": (
                        applied_arm_target_qd[
                            release_transfer.retreat_rejoin_start_frame
                        ].tolist()
                    ),
                    "grasp_release_target_velocity_step_at_disable_rad_s": (
                        (
                            applied_arm_target_qd[
                                release_transfer.retreat_rejoin_start_frame
                            ]
                            - applied_arm_target_qd[
                                release_transfer.release_unload_end_frame
                            ]
                        ).tolist()
                    ),
                    "grasp_release_target_velocity_continuity_dynamics_audited": bool(
                        applied_motion_limit_audit["passed"]
                    ),
                }
            )
        else:
            metadata.update(
                {
                    "grasp_release_bumpless_transfer_enabled": False,
                    "grasp_release_bumpless_transfer_reason": (
                        "fixed_grasp_constraint_disabled"
                        if not params.fixed_grasp_enabled
                        else "release_transfer_not_constructed"
                    ),
                }
            )
        if runtime.planned_grasp_anchors is not None:
            metadata.update(
                {
                    "grasp_anchor_planned_source": (
                        "planned_handle_frame_to_tcp_relation"
                    ),
                    "grasp_anchor_planned_parent_xform_xyz_wxyz": _project_pose_from_matrix(
                        runtime.planned_grasp_anchors.parent_xform
                    ).tolist(),
                    "grasp_anchor_authored_child_xform_xyz_wxyz": _project_pose_from_matrix(
                        runtime.planned_grasp_anchors.child_xform
                    ).tolist(),
                    "remote_latch_allowed": False,
                    "remote_latch_prevention": _REMOTE_LATCH_PREVENTION,
                }
            )
        if runtime.grasp_child_anchor_readback is not None:
            metadata.update(runtime.grasp_child_anchor_readback.audit_metadata())
        if (
            runtime.grasp_initial_enabled_writeback is not None
            and activation_enabled_writeback is not None
            and release_enabled_writeback is not None
        ):
            metadata.update(
                {
                    "grasp_joint_initial_disabled_write_count": (
                        runtime.grasp_initial_enabled_writeback.runtime_model_write_count
                    ),
                    "grasp_joint_initial_disabled_readback_count": (
                        runtime.grasp_initial_enabled_writeback.runtime_model_readback_count
                    ),
                    "grasp_joint_initial_disabled_verified": bool(
                        runtime.grasp_initial_enabled_writeback.readback_verified
                        and not runtime.grasp_initial_enabled_writeback.enabled
                    ),
                    "grasp_joint_activation_enabled_write_count": (
                        activation_enabled_writeback.runtime_model_write_count
                    ),
                    "grasp_joint_activation_enabled_readback_count": (
                        activation_enabled_writeback.runtime_model_readback_count
                    ),
                    "grasp_joint_activation_enabled_verified": bool(
                        activation_enabled_writeback.readback_verified
                        and activation_enabled_writeback.enabled
                    ),
                    "grasp_joint_release_disabled_write_count": (
                        release_enabled_writeback.runtime_model_write_count
                    ),
                    "grasp_joint_release_disabled_readback_count": (
                        release_enabled_writeback.runtime_model_readback_count
                    ),
                    "grasp_joint_release_disabled_verified": bool(
                        release_enabled_writeback.readback_verified
                        and not release_enabled_writeback.enabled
                    ),
                    "grasp_joint_enabled_runtime_model_write_count": 3,
                    "grasp_joint_enabled_runtime_model_readback_count": 3,
                    "grasp_joint_enabled_write_readback_verified": True,
                }
            )
        if activation_capture is not None:
            metadata.update(activation_capture.audit_metadata())
            metadata.update(
                {
                    "grasp_anchor_post_capture_position_limit_m": (
                        _CAPTURE_POSITION_TOLERANCE_M
                    ),
                    "grasp_anchor_post_capture_orientation_limit_deg": (
                        _CAPTURE_ORIENTATION_TOLERANCE_DEG
                    ),
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
                    "grasp_activation_gate_linear_velocity_limit_m_s": (
                        params.grasp_activation_linear_velocity_tolerance_m_s
                    ),
                    "grasp_activation_gate_angular_velocity_limit_deg_s": (
                        params.grasp_activation_angular_velocity_tolerance_deg_s
                    ),
                    "grasp_activation_pose_gate_passed": None,
                    "grasp_activation_twist_gate_passed": None,
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
            applied_arm_joint_target_q=applied_arm_targets.copy(),
            applied_arm_joint_target_qd=applied_arm_target_qd.copy(),
            measured_joint_q=measured_q,
            measured_joint_qd=measured_qd,
            measured_arm_joint_q=arm_q,
            measured_arm_joint_qd=arm_qd,
            measured_finger_joint_qd=finger_qd,
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
            finger_joint_names=tuple(
                ref.configured_name for ref in runtime.finger_refs
            ),
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


def _physics_arm_motion_metric_inputs(
    metadata: Mapping[str, Any],
    arm_joint_names: Sequence[str],
    expected_initial_joint_q: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and order runtime-audited motion inputs for final metrics."""

    names = tuple(str(name) for name in arm_joint_names)
    expected_initial = np.asarray(expected_initial_joint_q, dtype=float)
    try:
        initial = np.asarray(
            metadata["robot_initial_arm_joint_positions"], dtype=float
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "physics metadata has no valid initial arm joint state",
            stage="physics_audit",
            details={"field": "robot_initial_arm_joint_positions"},
        ) from exc
    if (
        initial.shape != (len(names),)
        or not np.isfinite(initial).all()
        or expected_initial.shape != initial.shape
        or not np.isfinite(expected_initial).all()
        or not np.array_equal(initial, expected_initial)
    ):
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "physics metadata initial arm state differs from configuration",
            stage="physics_audit",
            details={
                "field": "robot_initial_arm_joint_positions",
                "joint_names": list(names),
                "metadata_value": _json_safe(initial),
                "configured_value": _json_safe(expected_initial),
            },
        )

    raw_limits = metadata.get("arm_joint_velocity_limits")
    if not isinstance(raw_limits, Mapping) or set(raw_limits) != set(names):
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "physics metadata velocity limits do not match configured arm joints",
            stage="physics_audit",
            details={
                "field": "arm_joint_velocity_limits",
                "joint_names": list(names),
                "metadata_keys": (
                    sorted(str(key) for key in raw_limits)
                    if isinstance(raw_limits, Mapping)
                    else None
                ),
            },
        )
    try:
        ordered_limits = np.asarray(
            [raw_limits[name] for name in names], dtype=float
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "physics metadata contains malformed arm joint velocity limits",
            stage="physics_audit",
            details={"field": "arm_joint_velocity_limits"},
        ) from exc
    if (
        ordered_limits.shape != (len(names),)
        or not np.isfinite(ordered_limits).all()
        or np.any(ordered_limits <= 0.0)
    ):
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "physics metadata arm joint velocity limits must be finite and positive",
            stage="physics_audit",
            details={
                "field": "arm_joint_velocity_limits",
                "ordered_value": _json_safe(ordered_limits),
            },
        )
    return initial, ordered_limits


def _physics_finger_motion_metric_inputs(
    metadata: Mapping[str, Any],
    finger_joint_names: Sequence[str],
) -> dict[str, float]:
    """Validate runtime-audited URDF limits and preserve configured name order."""

    try:
        names = tuple(finger_joint_names)
    except TypeError as exc:
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "configured finger joint names are not iterable",
            stage="physics_audit",
            details={"field": "finger_joint_names"},
        ) from exc
    if (
        not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "configured finger joint names are malformed",
            stage="physics_audit",
            details={"field": "finger_joint_names", "value": _json_safe(names)},
        )
    raw_limits = metadata.get("finger_joint_velocity_limits")
    if not isinstance(raw_limits, Mapping) or set(raw_limits) != set(names):
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "physics metadata velocity limits do not match configured finger joints",
            stage="physics_audit",
            details={
                "field": "finger_joint_velocity_limits",
                "joint_names": list(names),
                "metadata_keys": (
                    sorted(str(key) for key in raw_limits)
                    if isinstance(raw_limits, Mapping)
                    else None
                ),
            },
        )
    try:
        ordered = np.asarray([raw_limits[name] for name in names], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "physics metadata contains malformed finger joint velocity limits",
            stage="physics_audit",
            details={"field": "finger_joint_velocity_limits"},
        ) from exc
    if (
        ordered.shape != (len(names),)
        or not np.isfinite(ordered).all()
        or np.any(ordered <= 0.0)
    ):
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "physics metadata finger velocity limits must be finite and positive",
            stage="physics_audit",
            details={
                "field": "finger_joint_velocity_limits",
                "ordered_value": _json_safe(ordered),
            },
        )
    return {
        name: float(value)
        for name, value in zip(names, ordered, strict=True)
    }


def _motion_array_summary(values: np.ndarray) -> dict[str, Any]:
    """Summarize a finite frame-by-joint motion array without dropping samples."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("motion array must have shape (frames, joints)")
    if not np.isfinite(array).all():
        raise ValueError("motion array must contain only finite values")
    magnitude = np.abs(array)
    flat_index = int(np.argmax(magnitude))
    frame_index, joint_index = np.unravel_index(flat_index, magnitude.shape)
    return {
        "max": float(magnitude[frame_index, joint_index]),
        "per_joint": np.max(magnitude, axis=0).tolist(),
        "frame_index": int(frame_index),
        "joint_index": int(joint_index),
        "sample_count": int(array.shape[0]),
    }


def _with_measured_arm_motion_acceptance(
    computed: Mapping[str, Any],
    *,
    measured_arm_joint_qd: np.ndarray,
    joint_velocity_limits_rad_s: np.ndarray,
    sample_dt_s: float,
    max_velocity_limit_ratio: float,
    max_acceleration_rad_s2: float,
    max_jerk_rad_s3: float,
) -> dict[str, Any]:
    """Gate Newton endpoint velocities instead of trusting position differences.

    ``newton.eval_ik`` supplies the generalized velocity at every post-step
    endpoint.  Acceleration and jerk begin from the verified zero initial
    velocity and the explicit zero pre-control acceleration convention.
    Position finite differences remain useful trajectory diagnostics, but they
    cannot hide a dynamic overshoot or oscillation that appears in measured
    generalized velocity.
    """

    qd = np.asarray(measured_arm_joint_qd, dtype=float)
    limits = np.asarray(joint_velocity_limits_rad_s, dtype=float)
    step = _positive(sample_dt_s, "measured arm motion dt")
    ratio_limit = _positive(
        max_velocity_limit_ratio, "max measured joint velocity limit ratio"
    )
    acceleration_limit = _positive(
        max_acceleration_rad_s2, "max measured joint acceleration"
    )
    jerk_limit = _positive(max_jerk_rad_s3, "max measured joint jerk")
    if (
        qd.ndim != 2
        or qd.shape[0] < 1
        or qd.shape[1] < 1
        or limits.shape != (qd.shape[1],)
        or not np.isfinite(qd).all()
        or not np.isfinite(limits).all()
        or np.any(limits <= 0.0)
    ):
        raise ValueError("measured arm qd and URDF velocity limits are invalid")

    velocity_ratio = np.abs(qd) / limits[None, :]
    accelerations = np.diff(
        np.vstack((np.zeros((1, qd.shape[1]), dtype=float), qd)),
        axis=0,
    ) / step
    jerks = np.diff(
        np.vstack(
            (np.zeros((1, qd.shape[1]), dtype=float), accelerations)
        ),
        axis=0,
    ) / step
    velocity_summary = _motion_array_summary(qd)
    ratio_summary = _motion_array_summary(velocity_ratio)
    final_ratio_summary = _motion_array_summary(velocity_ratio[-1:, :])
    acceleration_summary = _motion_array_summary(accelerations)
    jerk_summary = _motion_array_summary(jerks)

    def gate(summary: Mapping[str, Any], threshold: float) -> dict[str, Any]:
        value = float(summary["max"])
        return {
            "value": value,
            "operator": "<=",
            "threshold": threshold,
            "passed": bool(value <= threshold),
            "source": "newton_eval_ik_post_step_joint_qd",
        }

    measured_gates = {
        "max_measured_joint_velocity_limit_ratio": gate(
            ratio_summary, ratio_limit
        ),
        "final_measured_joint_velocity_limit_ratio": gate(
            final_ratio_summary, ratio_limit
        ),
        "max_measured_joint_acceleration_rad_s2": gate(
            acceleration_summary, acceleration_limit
        ),
        "max_measured_joint_jerk_rad_s3": gate(jerk_summary, jerk_limit),
    }
    gates = dict(computed["gates"])
    gates.update(measured_gates)
    return {
        **computed,
        "measured_joint_motion_source": "newton_eval_ik_post_step_joint_qd",
        "measured_joint_motion_initial_velocity_rad_s": 0.0,
        "measured_joint_motion_initial_acceleration_rad_s2": 0.0,
        "measured_joint_velocity_sample_count": velocity_summary["sample_count"],
        "max_measured_joint_velocity_rad_s": velocity_summary["max"],
        "per_joint_max_measured_velocity_rad_s": velocity_summary["per_joint"],
        "max_measured_joint_velocity_frame_index": velocity_summary[
            "frame_index"
        ],
        "max_measured_joint_velocity_joint_index": velocity_summary[
            "joint_index"
        ],
        "max_measured_joint_velocity_limit_ratio": ratio_summary["max"],
        "per_joint_max_measured_velocity_limit_ratio": ratio_summary[
            "per_joint"
        ],
        "max_measured_joint_velocity_limit_ratio_frame_index": ratio_summary[
            "frame_index"
        ],
        "max_measured_joint_velocity_limit_ratio_joint_index": ratio_summary[
            "joint_index"
        ],
        "final_measured_joint_velocity_limit_ratio": final_ratio_summary["max"],
        "per_joint_final_measured_velocity_limit_ratio": final_ratio_summary[
            "per_joint"
        ],
        "measured_joint_acceleration_sample_count": acceleration_summary[
            "sample_count"
        ],
        "max_measured_joint_acceleration_rad_s2": acceleration_summary["max"],
        "per_joint_max_measured_acceleration_rad_s2": acceleration_summary[
            "per_joint"
        ],
        "max_measured_joint_acceleration_frame_index": acceleration_summary[
            "frame_index"
        ],
        "max_measured_joint_acceleration_joint_index": acceleration_summary[
            "joint_index"
        ],
        "measured_joint_jerk_sample_count": jerk_summary["sample_count"],
        "max_measured_joint_jerk_rad_s3": jerk_summary["max"],
        "per_joint_max_measured_jerk_rad_s3": jerk_summary["per_joint"],
        "max_measured_joint_jerk_frame_index": jerk_summary["frame_index"],
        "max_measured_joint_jerk_joint_index": jerk_summary["joint_index"],
        "final_measured_arm_joint_velocity_rad_s": qd[-1].tolist(),
        "gates": gates,
        "success": bool(computed["success"])
        and all(bool(item["passed"]) for item in measured_gates.values()),
    }


def _with_measured_finger_motion_acceptance(
    computed: Mapping[str, Any],
    *,
    measured_finger_joint_qd: np.ndarray,
    finger_joint_names: Sequence[str],
    finger_velocity_limits_m_s: Mapping[str, float],
    sample_dt_s: float,
    max_velocity_limit_ratio: float,
    max_acceleration_m_s2: float,
    max_jerk_m_s3: float,
) -> dict[str, Any]:
    """Gate measured prismatic finger velocity, acceleration, and jerk.

    Finger columns are already extracted through finalized, name-resolved
    ``ScalarJointRef`` DoF indices.  Derivatives include the verified zero
    initial velocity and explicit zero initial acceleration, so frame zero is
    an acceptance sample rather than an unobserved startup transition.
    """

    try:
        qd = np.asarray(measured_finger_joint_qd, dtype=float)
        raw_names = tuple(finger_joint_names)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "measured finger qd and names must be numeric/iterable"
        ) from exc
    if (
        qd.ndim != 2
        or qd.shape[0] < 1
        or qd.shape[1] < 1
        or len(raw_names) != qd.shape[1]
        or any(not isinstance(name, str) or not name for name in raw_names)
        or len(set(raw_names)) != len(raw_names)
        or not np.isfinite(qd).all()
    ):
        raise ValueError(
            "measured finger qd and name-resolved finger layout are invalid"
        )
    names = tuple(str(name) for name in raw_names)
    if (
        not isinstance(finger_velocity_limits_m_s, Mapping)
        or set(finger_velocity_limits_m_s) != set(names)
    ):
        raise ValueError(
            "measured finger velocity limits must match the named finger layout"
        )
    try:
        velocity_limits = np.asarray(
            [finger_velocity_limits_m_s[name] for name in names], dtype=float
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("measured finger velocity limits are malformed") from exc
    if (
        velocity_limits.shape != (len(names),)
        or not np.isfinite(velocity_limits).all()
        or np.any(velocity_limits <= 0.0)
    ):
        raise ValueError(
            "measured finger velocity limits must be finite and positive"
        )
    step = _positive(sample_dt_s, "measured finger motion dt")
    ratio_limit = _positive(
        max_velocity_limit_ratio,
        "max measured finger velocity limit ratio",
    )
    acceleration_limit = _positive(
        max_acceleration_m_s2, "max measured finger acceleration"
    )
    jerk_limit = _positive(max_jerk_m_s3, "max measured finger jerk")

    velocity_ratio = np.abs(qd) / velocity_limits[None, :]
    accelerations = np.diff(
        np.vstack((np.zeros((1, qd.shape[1]), dtype=float), qd)),
        axis=0,
    ) / step
    jerks = np.diff(
        np.vstack(
            (np.zeros((1, qd.shape[1]), dtype=float), accelerations)
        ),
        axis=0,
    ) / step
    velocity_summary = _motion_array_summary(qd)
    ratio_summary = _motion_array_summary(velocity_ratio)
    final_ratio_summary = _motion_array_summary(velocity_ratio[-1:, :])
    acceleration_summary = _motion_array_summary(accelerations)
    jerk_summary = _motion_array_summary(jerks)

    def named(values: Sequence[float]) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in zip(names, values, strict=True)
        }

    source = "newton_eval_ik_post_step_name_resolved_finger_joint_qd"

    def gate(summary: Mapping[str, Any], threshold: float) -> dict[str, Any]:
        value = float(summary["max"])
        return {
            "value": value,
            "operator": "<=",
            "threshold": threshold,
            "passed": bool(value <= threshold),
            "source": source,
        }

    measured_gates = {
        "max_measured_finger_velocity_limit_ratio": gate(
            ratio_summary, ratio_limit
        ),
        "final_measured_finger_velocity_limit_ratio": gate(
            final_ratio_summary, ratio_limit
        ),
        "max_measured_finger_acceleration_m_s2": gate(
            acceleration_summary, acceleration_limit
        ),
        "max_measured_finger_jerk_m_s3": gate(jerk_summary, jerk_limit),
    }
    gates = dict(computed["gates"])
    gates.update(measured_gates)
    velocity_peak_index = int(velocity_summary["joint_index"])
    ratio_peak_index = int(ratio_summary["joint_index"])
    acceleration_peak_index = int(acceleration_summary["joint_index"])
    jerk_peak_index = int(jerk_summary["joint_index"])
    return {
        **computed,
        "measured_finger_motion_source": source,
        "measured_finger_joint_names": list(names),
        "measured_finger_joint_velocity_limits_m_s": named(velocity_limits),
        "measured_finger_motion_initial_velocity_m_s": named(
            np.zeros(qd.shape[1], dtype=float)
        ),
        "measured_finger_motion_initial_acceleration_m_s2": named(
            np.zeros(qd.shape[1], dtype=float)
        ),
        "measured_finger_velocity_sample_count": velocity_summary[
            "sample_count"
        ],
        "max_measured_finger_velocity_m_s": velocity_summary["max"],
        "per_finger_max_measured_velocity_m_s": named(
            velocity_summary["per_joint"]
        ),
        "max_measured_finger_velocity_frame_index": velocity_summary[
            "frame_index"
        ],
        "max_measured_finger_velocity_finger_index": velocity_peak_index,
        "max_measured_finger_velocity_finger_name": names[
            velocity_peak_index
        ],
        "max_measured_finger_velocity_limit_ratio": ratio_summary["max"],
        "per_finger_max_measured_velocity_limit_ratio": named(
            ratio_summary["per_joint"]
        ),
        "max_measured_finger_velocity_limit_ratio_frame_index": (
            ratio_summary["frame_index"]
        ),
        "max_measured_finger_velocity_limit_ratio_finger_index": (
            ratio_peak_index
        ),
        "max_measured_finger_velocity_limit_ratio_finger_name": names[
            ratio_peak_index
        ],
        "final_measured_finger_velocity_limit_ratio": final_ratio_summary[
            "max"
        ],
        "per_finger_final_measured_velocity_limit_ratio": named(
            final_ratio_summary["per_joint"]
        ),
        "measured_finger_acceleration_sample_count": acceleration_summary[
            "sample_count"
        ],
        "max_measured_finger_acceleration_m_s2": acceleration_summary["max"],
        "per_finger_max_measured_acceleration_m_s2": named(
            acceleration_summary["per_joint"]
        ),
        "max_measured_finger_acceleration_frame_index": acceleration_summary[
            "frame_index"
        ],
        "max_measured_finger_acceleration_finger_index": (
            acceleration_peak_index
        ),
        "max_measured_finger_acceleration_finger_name": names[
            acceleration_peak_index
        ],
        "measured_finger_jerk_sample_count": jerk_summary["sample_count"],
        "max_measured_finger_jerk_m_s3": jerk_summary["max"],
        "per_finger_max_measured_jerk_m_s3": named(
            jerk_summary["per_joint"]
        ),
        "max_measured_finger_jerk_frame_index": jerk_summary["frame_index"],
        "max_measured_finger_jerk_finger_index": jerk_peak_index,
        "max_measured_finger_jerk_finger_name": names[jerk_peak_index],
        "final_measured_finger_velocity_m_s": named(qd[-1]),
        "final_measured_finger_acceleration_m_s2": named(accelerations[-1]),
        "final_measured_finger_jerk_m_s3": named(jerks[-1]),
        "gates": gates,
        "success": bool(computed["success"])
        and all(bool(item["passed"]) for item in measured_gates.values()),
    }


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
    *,
    expected_finger_joint_names: Sequence[str] | None = None,
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
        "applied_arm_joint_target_q": 2,
        "applied_arm_joint_target_qd": 2,
        "measured_joint_q": 2,
        "measured_joint_qd": 2,
        "measured_arm_joint_q": 2,
        "measured_arm_joint_qd": 2,
        "measured_finger_joint_qd": 2,
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

    for name in (
        "applied_arm_joint_target_q",
        "applied_arm_joint_target_qd",
    ):
        raw_value = np.asarray(getattr(rollout, name))
        if raw_value.dtype != np.dtype(np.float32):
            raise PipelineError(
                FailureCode.NUMERICAL_INSTABILITY,
                f"physics {name} is not the audited float32 controller value",
                stage="physics_audit",
                details={"dtype": str(raw_value.dtype)},
            )
        arrays[name] = raw_value.copy()

    expected_arm_shape = commands.arm_joint_targets.shape
    if arrays["applied_arm_joint_target_q"].shape != expected_arm_shape or arrays[
        "applied_arm_joint_target_qd"
    ].shape != expected_arm_shape:
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "applied arm controller targets do not match configured arm layout",
            stage="physics_result",
            details={"expected": list(expected_arm_shape)},
        )
    if arrays["measured_arm_joint_q"].shape != expected_arm_shape:
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "measured dynamic arm positions do not match configured arm layout",
            stage="physics_result",
            details={"expected": list(expected_arm_shape), "actual": list(arrays["measured_arm_joint_q"].shape)},
        )
    if arrays["measured_arm_joint_qd"].shape != expected_arm_shape:
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "measured dynamic arm velocities do not match configured arm layout",
            stage="physics_result",
            details={
                "expected": list(expected_arm_shape),
                "actual": list(arrays["measured_arm_joint_qd"].shape),
            },
        )
    try:
        finger_joint_names = tuple(
            getattr(rollout, "finger_joint_names", ())
        )
    except TypeError as exc:
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "measured dynamic finger velocities lack an iterable named layout",
            stage="physics_result",
            details={
                "finger_joint_names": _json_safe(
                    getattr(rollout, "finger_joint_names", None)
                )
            },
        ) from exc
    if (
        not finger_joint_names
        or any(
            not isinstance(name, str) or not name
            for name in finger_joint_names
        )
        or len(set(finger_joint_names)) != len(finger_joint_names)
        or arrays["measured_finger_joint_qd"].shape
        != (frame_count, len(finger_joint_names))
    ):
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "measured dynamic finger velocities lack a valid named layout",
            stage="physics_result",
            details={
                "finger_joint_names": _json_safe(finger_joint_names),
                "measured_finger_joint_qd_shape": list(
                    arrays["measured_finger_joint_qd"].shape
                ),
            },
        )
    if expected_finger_joint_names is not None:
        try:
            configured_finger_joint_names = tuple(expected_finger_joint_names)
        except TypeError as exc:
            raise PipelineError(
                FailureCode.CONFIG_INVALID,
                "configured finger joint names are not iterable",
                stage="physics_result",
            ) from exc
        if finger_joint_names != configured_finger_joint_names:
            raise PipelineError(
                FailureCode.PHYSICS_UNAVAILABLE,
                "measured finger velocity names differ from configuration",
                stage="physics_result",
                details={
                    "configured_finger_joint_names": _json_safe(
                        configured_finger_joint_names
                    ),
                    "measured_finger_joint_names": list(
                        finger_joint_names
                    ),
                },
            )
    arrays["finger_joint_names"] = np.asarray(
        finger_joint_names, dtype="U"
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
    if metadata.get("measured_finger_joint_velocity_source") != (
        "newton_eval_ik_post_step_name_resolved_joint_qd"
    ) or metadata.get("measured_finger_joint_names") != list(
        finger_joint_names
    ):
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "physics metadata does not prove name-resolved measured finger velocities",
            stage="physics_audit",
            details={
                "measured_finger_joint_velocity_source": metadata.get(
                    "measured_finger_joint_velocity_source"
                ),
                "metadata_finger_joint_names": metadata.get(
                    "measured_finger_joint_names"
                ),
                "rollout_finger_joint_names": list(finger_joint_names),
            },
        )
    if metadata.get("state_sample_timing") != _STATE_SAMPLE_TIMING:
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "physics backend did not report post-step end-of-frame state sampling",
            stage="physics_audit",
            details={"state_sample_timing": metadata.get("state_sample_timing")},
        )
    simulation_dt = metadata.get("simulation_dt_s")
    if (
        not isinstance(simulation_dt, (int, float, np.number))
        or isinstance(simulation_dt, (bool, np.bool_))
        or not math.isfinite(float(simulation_dt))
        or float(simulation_dt) <= 0.0
    ):
        raise PipelineError(
            FailureCode.PHYSICS_UNAVAILABLE,
            "physics metadata has no finite positive simulation timestep",
            stage="physics_audit",
            details={"simulation_dt_s": _json_safe(simulation_dt)},
        )
    expected_time_s = _post_step_state_sample_times(
        frame_count, float(simulation_dt)
    )
    time_tolerance = 64.0 * np.finfo(float).eps * max(
        1.0, float(expected_time_s[-1])
    )
    if not np.allclose(
        arrays["time_s"],
        expected_time_s,
        rtol=0.0,
        atol=time_tolerance,
    ):
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "physics sample timestamps are not post-step frame endpoints",
            stage="physics_audit",
            details={
                "expected_time_s": expected_time_s.tolist(),
                "actual_time_s": arrays["time_s"].tolist(),
            },
        )
    if metadata.get("constraint_backend") == (
        "newton_fixed_loop_joint_planned_gate_with_measured_capture"
    ):
        required_transfer_metadata = {
            "grasp_release_bumpless_transfer_enabled": True,
            "grasp_release_blend_profile": (
                "quintic_smoothstep_endpoint_exact"
            ),
            "grasp_release_applied_arm_target_dtype": "float32",
            "grasp_release_applied_motion_limit_preflight_passed": True,
            "grasp_release_applied_position_limit_preflight_passed": True,
            "grasp_release_applied_reserve_preflight_passed": True,
            "grasp_release_constraint_active_during_unload_verified": True,
            "grasp_release_constraint_disabled_during_rejoin_verified": True,
            "grasp_release_target_position_continuity_at_disable_verified": True,
            "grasp_release_target_velocity_continuity_dynamics_audited": True,
            "robot_applied_arm_target_position_write_readback_verified": True,
            "robot_applied_arm_target_velocity_write_readback_verified": True,
        }
        mismatches = {
            key: {"expected": expected, "actual": metadata.get(key)}
            for key, expected in required_transfer_metadata.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise PipelineError(
                FailureCode.NUMERICAL_INSTABILITY,
                "physics backend did not prove the bumpless fixed-grasp release transfer",
                stage="physics_audit",
                details={"mismatches": mismatches},
            )
        integer_fields = (
            "grasp_release_equilibrium_capture_frame",
            "grasp_release_unload_start_frame",
            "grasp_release_unload_end_frame",
            "grasp_release_constraint_disable_frame",
            "grasp_release_retreat_rejoin_start_frame",
            "grasp_release_retreat_rejoin_end_frame",
            "grasp_release_blend_frames",
        )
        if any(
            not isinstance(metadata.get(field), (int, np.integer))
            or isinstance(metadata.get(field), (bool, np.bool_))
            for field in integer_fields
        ):
            raise PipelineError(
                FailureCode.NUMERICAL_INSTABILITY,
                "bumpless release metadata has malformed frame indices",
                stage="physics_audit",
            )
        capture_frame = int(metadata["grasp_release_equilibrium_capture_frame"])
        unload_start = int(metadata["grasp_release_unload_start_frame"])
        unload_end = int(metadata["grasp_release_unload_end_frame"])
        disable_frame = int(metadata["grasp_release_constraint_disable_frame"])
        rejoin_start = int(metadata["grasp_release_retreat_rejoin_start_frame"])
        rejoin_end = int(metadata["grasp_release_retreat_rejoin_end_frame"])
        blend_frames = int(metadata["grasp_release_blend_frames"])
        try:
            captured = np.asarray(
                metadata["grasp_release_equilibrium_arm_q_float32"],
                dtype=np.float32,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineError(
                FailureCode.NUMERICAL_INSTABILITY,
                "bumpless release metadata lacks its captured arm equilibrium",
                stage="physics_audit",
            ) from exc
        if not (
            captured.shape == (expected_arm_shape[1],)
            and np.isfinite(captured).all()
            and capture_frame + 1 == unload_start
            and 0 <= capture_frame < unload_start <= unload_end < disable_frame
            and disable_frame == rejoin_start <= rejoin_end < frame_count
            and rejoin_end - rejoin_start + 1 == blend_frames
            and blend_frames >= 2
            and np.all(phases[unload_start : unload_end + 1] == "release")
            and np.all(phases[rejoin_start : rejoin_end + 1] == "retreat")
            and np.all(
                arrays["grasp_constraint_active"][unload_start : unload_end + 1]
            )
            and not np.any(
                arrays["grasp_constraint_active"][rejoin_start : rejoin_end + 1]
            )
            and np.array_equal(
                arrays["applied_arm_joint_target_q"][unload_start],
                commands.arm_joint_targets[unload_start].astype(np.float32),
            )
            and np.array_equal(
                arrays["applied_arm_joint_target_q"][unload_end], captured
            )
            and np.array_equal(
                arrays["applied_arm_joint_target_q"][rejoin_start], captured
            )
            and np.array_equal(
                arrays["applied_arm_joint_target_q"][rejoin_end],
                commands.arm_joint_targets[rejoin_end].astype(np.float32),
            )
        ):
            raise PipelineError(
                FailureCode.NUMERICAL_INSTABILITY,
                "bumpless release arrays disagree with their audited windows or endpoints",
                stage="physics_audit",
            )

        raw_arm_limits = metadata.get("arm_joint_velocity_limits")
        initial_arm = np.asarray(
            metadata.get("robot_initial_arm_joint_positions"), dtype=float
        )
        arm_names = tuple(str(name) for name in raw_arm_limits) if isinstance(
            raw_arm_limits, Mapping
        ) else ()
        if (
            len(arm_names) != expected_arm_shape[1]
            or initial_arm.shape != (expected_arm_shape[1],)
        ):
            raise PipelineError(
                FailureCode.NUMERICAL_INSTABILITY,
                "bumpless release velocity audit lacks name-aligned arm metadata",
                stage="physics_audit",
            )
        arm_limits = np.asarray(
            [raw_arm_limits[name] for name in arm_names], dtype=float
        )
        expected_applied_velocity = _bounded_float32_velocity_targets(
            np.diff(
                np.vstack(
                    (
                        initial_arm[None, :],
                        arrays["applied_arm_joint_target_q"].astype(float),
                    )
                ),
                axis=0,
            )
            / float(simulation_dt),
            arm_limits,
        )
        if not np.array_equal(
            arrays["applied_arm_joint_target_qd"], expected_applied_velocity
        ):
            mismatch = np.argwhere(
                arrays["applied_arm_joint_target_qd"]
                != expected_applied_velocity
            )[0]
            raise PipelineError(
                FailureCode.NUMERICAL_INSTABILITY,
                "stored applied arm velocity is not the finite difference actually written to Newton",
                stage="physics_audit",
                details={
                    "frame_index": int(mismatch[0]),
                    "joint_index": int(mismatch[1]),
                },
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
    if metadata.get("constraint_backend") == (
        "newton_fixed_loop_joint_planned_gate_with_measured_capture"
    ):
        expected_remote_latch_reason = _REMOTE_LATCH_PREVENTION
        required_capture_metadata = {
            "grasp_activation_gate_relation_source": (
                "planned_handle_frame_to_tcp_relation"
            ),
            "grasp_activation_twist_source": (
                "newton_state_body_qd_v_com_world_omega_world_at_"
                "runtime_capture_anchor"
            ),
            "grasp_activation_pose_gate_passed": True,
            "grasp_activation_twist_gate_passed": True,
            "grasp_activation_gate_passed": True,
            "grasp_anchor_source": (
                "runtime_measured_parent_child_body_poses_after_"
                "planned_pose_and_relative_anchor_twist_gates"
            ),
            "grasp_anchor_runtime_model_write_count": 1,
            "grasp_anchor_runtime_model_readback_count": 1,
            "grasp_anchor_runtime_model_write_readback_verified": True,
            "grasp_anchor_child_runtime_model_readback_count": 1,
            "grasp_anchor_child_runtime_model_readback_verified": True,
            "grasp_anchor_child_authored_match_verified": True,
            "grasp_joint_initial_disabled_write_count": 1,
            "grasp_joint_initial_disabled_readback_count": 1,
            "grasp_joint_initial_disabled_verified": True,
            "grasp_joint_activation_enabled_write_count": 1,
            "grasp_joint_activation_enabled_readback_count": 1,
            "grasp_joint_activation_enabled_verified": True,
            "grasp_joint_release_disabled_write_count": 1,
            "grasp_joint_release_disabled_readback_count": 1,
            "grasp_joint_release_disabled_verified": True,
            "grasp_joint_enabled_runtime_model_write_count": 3,
            "grasp_joint_enabled_runtime_model_readback_count": 3,
            "grasp_joint_enabled_write_readback_verified": True,
            "remote_latch_allowed": False,
            "remote_latch_prevention": expected_remote_latch_reason,
        }
        mismatched_capture_metadata = {
            key: {"expected": expected, "actual": metadata.get(key)}
            for key, expected in required_capture_metadata.items()
            if metadata.get(key) != expected
        }
        if mismatched_capture_metadata:
            raise PipelineError(
                FailureCode.NUMERICAL_INSTABILITY,
                "physics backend did not prove gated, verified fixed-grasp capture",
                stage="physics_audit",
                details={"mismatches": mismatched_capture_metadata},
            )
        capture_errors_and_limits = (
            (
                "grasp_anchor_post_capture_position_error_m",
                "grasp_anchor_post_capture_position_limit_m",
            ),
            (
                "grasp_anchor_post_capture_orientation_error_deg",
                "grasp_anchor_post_capture_orientation_limit_deg",
            ),
            (
                "grasp_activation_gate_relative_linear_speed_m_s",
                "grasp_activation_gate_linear_velocity_limit_m_s",
            ),
            (
                "grasp_activation_gate_relative_angular_speed_deg_s",
                "grasp_activation_gate_angular_velocity_limit_deg_s",
            ),
        )
        for error_key, limit_key in capture_errors_and_limits:
            error_value = metadata.get(error_key)
            limit_value = metadata.get(limit_key)
            if (
                not isinstance(error_value, (int, float, np.number))
                or isinstance(error_value, (bool, np.bool_))
                or not math.isfinite(float(error_value))
                or float(error_value) < 0.0
                or not isinstance(limit_value, (int, float, np.number))
                or isinstance(limit_value, (bool, np.bool_))
                or not math.isfinite(float(limit_value))
                or float(limit_value) <= 0.0
                or float(error_value) > float(limit_value)
            ):
                raise PipelineError(
                    FailureCode.NUMERICAL_INSTABILITY,
                    "physics backend reported a non-coincident captured grasp anchor",
                    stage="physics_audit",
                    details={
                        "error_key": error_key,
                        "error_value": _json_safe(error_value),
                        "limit_key": limit_key,
                        "limit_value": _json_safe(limit_value),
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
        "robot_initial_joint_velocity_zero_verified",
        "robot_initial_joint_position_config_verified",
    ):
        if metadata.get(field) is not True:
            raise PipelineError(
                FailureCode.NUMERICAL_INSTABILITY,
                "joint-PD isolation or dynamic-body assertion is missing",
                stage="physics_audit",
                details={"field": field, "value": metadata.get(field)},
            )
    if metadata.get("robot_initial_joint_acceleration_semantics") != (
        "zero_before_first_control_step"
    ):
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "physics backend did not prove zero initial joint acceleration",
            stage="physics_audit",
            details={
                "robot_initial_joint_acceleration_semantics": metadata.get(
                    "robot_initial_joint_acceleration_semantics"
                )
            },
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
    finger_joint_names: Sequence[str],
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
                "arm_joint_velocities_rad_s": {
                    name: float(value)
                    for name, value in zip(
                        arm_joint_names,
                        physics["measured_arm_joint_qd"][index],
                        strict=True,
                    )
                },
                "finger_joint_velocities_m_s": {
                    name: float(value)
                    for name, value in zip(
                        finger_joint_names,
                        physics["measured_finger_joint_qd"][index],
                        strict=True,
                    )
                },
                "command_arm_joint_positions": {
                    name: float(value)
                    for name, value in zip(
                        arm_joint_names, reference_arm_q[index], strict=True
                    )
                },
                "command_arm_joint_position_semantics": (
                    "planned_kinematic_reference_not_necessarily_applied"
                ),
                "applied_arm_joint_target_positions": {
                    name: float(value)
                    for name, value in zip(
                        arm_joint_names,
                        physics["applied_arm_joint_target_q"][index],
                        strict=True,
                    )
                },
                "applied_arm_joint_target_velocities_rad_s": {
                    name: float(value)
                    for name, value in zip(
                        arm_joint_names,
                        physics["applied_arm_joint_target_qd"][index],
                        strict=True,
                    )
                },
                "applied_arm_joint_target_semantics": (
                    "float32_position_and_finite_difference_velocity_"
                    "written_to_newton_joint_pd"
                ),
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

    arm_joint_names = tuple(
        str(name) for name in config.get("assets.robot.arm_joint_names")
    )
    joint_lower, joint_upper = _arm_joint_limits(config)
    reference_reserve_audit = audit_arm_joint_reference_reserve(
        reference_arrays["arm_joint_q"],
        arm_joint_names,
        joint_lower,
        joint_upper,
        initial_arm_q=config.get("assets.robot.default_joint_positions"),
        control_limit_margin_rad=float(
            config.get("ik.control_limit_margin_rad")
        ),
        arm_joint_tracking_reserve_rad=float(
            config.get(
                "simulation.robot_control.arm_joint_tracking_reserve_rad"
            )
        ),
        phase_names=plan.phase_names,
    )
    log.add(
        "physics_reference_arm_joint_reserve_audited",
        audit=reference_reserve_audit,
    )
    if not bool(reference_reserve_audit["passed"]):
        raise PipelineError(
            FailureCode.IK_UNREACHABLE,
            "initial arm state or kinematic reference lacks the reserve required for physics tracking",
            stage="ik_motion_limits",
            details={
                "feasibility_scope": (
                    "configured_initial_state_and_entire_kinematic_reference_"
                    "before_physics"
                ),
                **reference_reserve_audit,
            },
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
        _validate_physics_rollout(
            rollout,
            commands,
            expected_finger_joint_names=tuple(
                str(name)
                for name in config.get("assets.robot.finger_joint_names")
            ),
        )
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
    finger_joint_names = tuple(
        str(name) for name in config.get("assets.robot.finger_joint_names")
    )
    initial_arm_joint_q, arm_joint_velocity_limits = (
        _physics_arm_motion_metric_inputs(
            metadata,
            arm_joint_names,
            config.get("assets.robot.default_joint_positions"),
        )
    )
    finger_joint_velocity_limits = _physics_finger_motion_metric_inputs(
        metadata, finger_joint_names
    )
    goal_angle_rad = math.radians(float(config.get("task.goal_angle_deg")))
    try:
        metric_thresholds = MetricThresholds.from_mapping(
            config.get("thresholds")
        )
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
            thresholds=metric_thresholds,
            joint_limit_tolerance_rad=float(
                config.get("ik.joint_limit_tolerance_rad")
            ),
            sample_dt_s=float(config.get("simulation.dt")),
            initial_joint_q=initial_arm_joint_q,
            joint_velocity_limits_rad_s=arm_joint_velocity_limits,
            include_position_difference_acceleration_jerk_gates=False,
        )
        computed["joint_motion_joint_q_source"] = (
            "newton_eval_ik_post_step_joint_q"
        )
        if (
            metric_thresholds.max_joint_velocity_limit_ratio is None
            or metric_thresholds.max_joint_acceleration_rad_s2 is None
            or metric_thresholds.max_joint_jerk_rad_s3 is None
        ):
            raise MetricsInputError(
                "PHYSICS_MOTION_THRESHOLD_REQUIRED",
                "physics acceptance requires velocity, acceleration, and jerk thresholds",
            )
        computed = _with_measured_arm_motion_acceptance(
            computed,
            measured_arm_joint_qd=physics["measured_arm_joint_qd"],
            joint_velocity_limits_rad_s=arm_joint_velocity_limits,
            sample_dt_s=float(config.get("simulation.dt")),
            max_velocity_limit_ratio=(
                metric_thresholds.max_joint_velocity_limit_ratio
            ),
            max_acceleration_rad_s2=(
                metric_thresholds.max_joint_acceleration_rad_s2
            ),
            max_jerk_rad_s3=metric_thresholds.max_joint_jerk_rad_s3,
        )
        computed = _with_measured_finger_motion_acceptance(
            computed,
            measured_finger_joint_qd=physics[
                "measured_finger_joint_qd"
            ],
            finger_joint_names=finger_joint_names,
            finger_velocity_limits_m_s=finger_joint_velocity_limits,
            sample_dt_s=float(config.get("simulation.dt")),
            max_velocity_limit_ratio=(
                metric_thresholds.max_joint_velocity_limit_ratio
            ),
            max_acceleration_m_s2=float(
                config.get("thresholds.max_finger_acceleration_m_s2")
            ),
            max_jerk_m_s3=float(
                config.get("thresholds.max_finger_jerk_m_s3")
            ),
        )
    except (MetricsInputError, ValueError) as exc:
        details = (
            exc.to_dict()
            if isinstance(exc, MetricsInputError)
            else {"code": "MEASURED_JOINT_MOTION_INVALID", "message": str(exc)}
        )
        raise PipelineError(
            FailureCode.NUMERICAL_INSTABILITY,
            "physics rollout metrics could not be computed",
            stage="metrics",
            details=details,
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
        "reference_arm_joint_reserve_audit": reference_reserve_audit,
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
        "reference_command_arm_joint_q": reference_arrays["arm_joint_q"],
        "arm_joint_q": physics["measured_arm_joint_q"],
        "arm_joint_names": np.asarray(arm_joint_names, dtype="U"),
        "initial_arm_joint_q": initial_arm_joint_q,
        "arm_joint_velocity_limits_rad_s": arm_joint_velocity_limits,
        "initial_gripper_width_m": np.asarray(
            float(config.get("assets.robot.open_gripper_width_m")), dtype=float
        ),
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
                finger_joint_names=finger_joint_names,
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
    "BumplessGraspReleaseTransfer",
    "FixedGraspAnchorWriteback",
    "FixedGraspActivationGate",
    "FixedGraspChildAnchorReadback",
    "FixedGraspJointEnabledWriteback",
    "FixedGraspRuntimeCapture",
    "GraspActivationWindow",
    "NewtonPhysicsAssistedSimulator",
    "PhysicsCommandTrajectory",
    "PhysicsParameters",
    "PhysicsRollout",
    "RobotJointVelocityLimits",
    "ScalarJointRef",
    "audit_arm_joint_reference_reserve",
    "audit_bumpless_grasp_release_transfer",
    "build_bumpless_grasp_release_transfer",
    "build_robot_position_targets",
    "build_robot_velocity_targets",
    "captured_fixed_grasp_parent_anchor",
    "evaluate_fixed_grasp_activation_gate",
    "fixed_grasp_activation_window",
    "handle_frame_world_from_link_poses",
    "kinematic_body_twists",
    "load_robot_joint_velocity_limits",
    "load_robot_arm_joint_position_limits",
    "named_robot_body_poses",
    "plan_massless_fixed_joint_collapse",
    "planned_fixed_grasp_anchors",
    "read_fixed_grasp_child_anchor",
    "require_newton_v13",
    "resolve_named_scalar_joints",
    "run_physics_assisted",
    "simulate_physics_assisted",
    "validate_robot_command_motion_limits",
    "validate_applied_arm_target_position_limits",
    "write_fixed_grasp_parent_anchor",
    "write_fixed_grasp_joint_enabled",
]
