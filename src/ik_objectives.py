"""Project-specific Newton IK objectives.

The module deliberately keeps its layout helpers usable when Newton/Warp are
not installed.  The remote runtime owns those optional dependencies, while
local unit tests can still validate the coordinate/DoF mapping contract.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


try:  # Optional locally; required only when constructing the Newton objective.
    import newton
    import warp as wp
except (ImportError, OSError) as exc:  # Native library loading may raise OSError.
    newton = None  # type: ignore[assignment]
    wp = None  # type: ignore[assignment]
    _NEWTON_IMPORT_ERROR: BaseException | None = exc
else:
    _NEWTON_IMPORT_ERROR = None


class ScalarJointLayoutError(ValueError):
    """Raised when a model cannot be represented by scalar DoF coordinates."""


class NewtonObjectiveUnavailableError(RuntimeError):
    """Raised when the optional Newton objective is used without Newton/Warp."""


def _integer_vector(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ScalarJointLayoutError(f"{name} must be one-dimensional, got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        try:
            converted = array.astype(np.int64)
        except (TypeError, ValueError) as exc:
            raise ScalarJointLayoutError(f"{name} must contain integers") from exc
        if not np.array_equal(array, converted):
            raise ScalarJointLayoutError(f"{name} must contain integers")
        array = converted
    return np.asarray(array, dtype=np.int64)


def build_scalar_dof_to_coord_map(
    joint_q_start: Sequence[int] | np.ndarray,
    joint_qd_start: Sequence[int] | np.ndarray,
    joint_dof_dim: Sequence[Sequence[int]] | np.ndarray,
) -> np.ndarray:
    """Build the Newton DoF-to-coordinate map for scalar movable joints.

    Newton stores generalized positions and velocities in distinct layouts.
    Revolute and prismatic joints have one coordinate per DoF, while ball and
    free joints contain quaternion coordinates and therefore cannot use the
    diagonal nominal-posture Jacobian implemented below.  Fixed joints are
    accepted and contribute no entries.

    Args:
        joint_q_start: Coordinate start offsets with the closing sentinel.
        joint_qd_start: DoF start offsets with the closing sentinel.
        joint_dof_dim: Per-joint ``(linear_dofs, angular_dofs)`` values.

    Returns:
        An ``int32`` array of length ``joint_dof_count`` whose values index the
        matching scalar generalized coordinates.
    """

    q_start = _integer_vector(joint_q_start, "joint_q_start")
    qd_start = _integer_vector(joint_qd_start, "joint_qd_start")
    dof_dim = np.asarray(joint_dof_dim)

    if dof_dim.ndim != 2 or dof_dim.shape[1] != 2:
        raise ScalarJointLayoutError(
            f"joint_dof_dim must have shape (joint_count, 2), got {dof_dim.shape}"
        )
    if not np.issubdtype(dof_dim.dtype, np.integer):
        try:
            converted = dof_dim.astype(np.int64)
        except (TypeError, ValueError) as exc:
            raise ScalarJointLayoutError("joint_dof_dim must contain integers") from exc
        if not np.array_equal(dof_dim, converted):
            raise ScalarJointLayoutError("joint_dof_dim must contain integers")
        dof_dim = converted
    dof_dim = np.asarray(dof_dim, dtype=np.int64)

    joint_count = dof_dim.shape[0]
    expected_start_length = joint_count + 1
    if len(q_start) != expected_start_length or len(qd_start) != expected_start_length:
        raise ScalarJointLayoutError(
            "joint_q_start and joint_qd_start must each contain joint_count + 1 entries"
        )
    if q_start[0] != 0 or qd_start[0] != 0:
        raise ScalarJointLayoutError("joint start arrays must begin at zero")
    if np.any(np.diff(q_start) < 0) or np.any(np.diff(qd_start) < 0):
        raise ScalarJointLayoutError("joint start arrays must be monotonic")
    if np.any(dof_dim < 0):
        raise ScalarJointLayoutError("joint_dof_dim cannot contain negative values")

    dof_count = int(qd_start[-1])
    mapping = np.full(dof_count, -1, dtype=np.int32)
    for joint_index in range(joint_count):
        coord_begin = int(q_start[joint_index])
        coord_end = int(q_start[joint_index + 1])
        dof_begin = int(qd_start[joint_index])
        dof_end = int(qd_start[joint_index + 1])
        declared_dofs = int(dof_dim[joint_index, 0] + dof_dim[joint_index, 1])
        actual_dofs = dof_end - dof_begin
        coord_count = coord_end - coord_begin

        if declared_dofs != actual_dofs:
            raise ScalarJointLayoutError(
                f"joint {joint_index} declares {declared_dofs} DoFs but its start offsets span "
                f"{actual_dofs}"
            )
        if actual_dofs == 0:
            if coord_count != 0:
                raise ScalarJointLayoutError(
                    f"fixed joint {joint_index} unexpectedly owns {coord_count} coordinates"
                )
            continue
        if coord_count != actual_dofs:
            raise ScalarJointLayoutError(
                f"joint {joint_index} has {coord_count} coordinates for {actual_dofs} DoFs; "
                "nominal posture supports scalar-coordinate joints only"
            )
        mapping[dof_begin:dof_end] = np.arange(coord_begin, coord_end, dtype=np.int32)

    if np.any(mapping < 0):
        missing = np.flatnonzero(mapping < 0).tolist()
        raise ScalarJointLayoutError(f"DoF-to-coordinate map has unmapped entries: {missing}")
    return mapping


def newton_objective_available() -> bool:
    """Return whether the Newton/Warp-backed objective can be constructed."""

    return _NEWTON_IMPORT_ERROR is None


if _NEWTON_IMPORT_ERROR is None:

    @wp.kernel
    def _nominal_posture_residuals(
        joint_q: wp.array2d[wp.float32],
        nominal_q: wp.array[wp.float32],
        dof_to_coord: wp.array[wp.int32],
        active_mask: wp.array[wp.float32],
        residual_scale: float,
        start_idx: int,
        residuals: wp.array2d[wp.float32],
    ):
        problem_idx, dof_idx = wp.tid()
        coord_idx = dof_to_coord[dof_idx]
        residuals[problem_idx, start_idx + dof_idx] = (
            residual_scale
            * active_mask[dof_idx]
            * (joint_q[problem_idx, coord_idx] - nominal_q[coord_idx])
        )


    @wp.kernel
    def _nominal_posture_jacobian(
        active_mask: wp.array[wp.float32],
        residual_scale: float,
        start_idx: int,
        jacobian: wp.array3d[wp.float32],
    ):
        problem_idx, dof_idx = wp.tid()
        jacobian[problem_idx, start_idx + dof_idx, dof_idx] = (
            residual_scale * active_mask[dof_idx]
        )


    class JointNominalObjective(newton.ik.IKObjective):
        """Analytic quadratic preference for a configured nominal posture.

        ``cost_weight`` is a cost coefficient, so the residual and Jacobian
        use ``sqrt(cost_weight)``.  ``active_mask`` selects which scalar DoFs
        participate without relying on robot-specific indices.
        """

        def __init__(
            self,
            nominal_q: Sequence[float] | np.ndarray,
            dof_to_coord: Sequence[int] | np.ndarray,
            active_mask: Sequence[float] | np.ndarray,
            *,
            cost_weight: float,
        ) -> None:
            super().__init__()
            nominal = np.asarray(nominal_q, dtype=np.float32)
            mapping = _integer_vector(dof_to_coord, "dof_to_coord").astype(np.int32)
            mask = np.asarray(active_mask, dtype=np.float32)

            if nominal.ndim != 1 or not np.isfinite(nominal).all():
                raise ValueError("nominal_q must be a finite one-dimensional vector")
            if mask.shape != mapping.shape or not np.isfinite(mask).all():
                raise ValueError("active_mask must be finite and match dof_to_coord")
            if np.any((mask != 0.0) & (mask != 1.0)):
                raise ValueError("active_mask entries must be exactly zero or one")
            if mapping.size == 0:
                raise ValueError("dof_to_coord cannot be empty")
            if np.any(mapping < 0) or np.any(mapping >= nominal.size):
                raise ValueError("dof_to_coord contains an out-of-range coordinate index")
            weight = float(cost_weight)
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError("cost_weight must be finite and non-negative")

            self._nominal_q_np = nominal
            self._dof_to_coord_np = mapping
            self._active_mask_np = mask
            self._residual_scale = math.sqrt(weight)
            self.n_dofs = int(mapping.size)
            self.nominal_q = None
            self.dof_to_coord = None
            self.active_mask = None

        def residual_dim(self) -> int:
            return self.n_dofs

        def supports_analytic(self) -> bool:
            return True

        def init_buffers(self, model: Any, jacobian_mode: Any) -> None:
            self._require_batch_layout()
            if model.joint_coord_count != len(self._nominal_q_np):
                raise ValueError(
                    "nominal_q length does not match model.joint_coord_count: "
                    f"{len(self._nominal_q_np)} != {model.joint_coord_count}"
                )
            if model.joint_dof_count != self.n_dofs:
                raise ValueError(
                    "dof_to_coord length does not match model.joint_dof_count: "
                    f"{self.n_dofs} != {model.joint_dof_count}"
                )
            self.nominal_q = wp.array(
                self._nominal_q_np, dtype=wp.float32, device=self.device
            )
            self.dof_to_coord = wp.array(
                self._dof_to_coord_np, dtype=wp.int32, device=self.device
            )
            self.active_mask = wp.array(
                self._active_mask_np, dtype=wp.float32, device=self.device
            )

        def compute_residuals(
            self,
            body_q: Any,
            joint_q: Any,
            model: Any,
            residuals: Any,
            start_idx: int,
            problem_idx: Any,
        ) -> None:
            wp.launch(
                _nominal_posture_residuals,
                dim=[joint_q.shape[0], self.n_dofs],
                inputs=[
                    joint_q,
                    self.nominal_q,
                    self.dof_to_coord,
                    self.active_mask,
                    self._residual_scale,
                    start_idx,
                ],
                outputs=[residuals],
                device=self.device,
            )

        def compute_jacobian_analytic(
            self,
            body_q: Any,
            joint_q: Any,
            model: Any,
            jacobian: Any,
            joint_S_s: Any,
            start_idx: int,
        ) -> None:
            wp.launch(
                _nominal_posture_jacobian,
                dim=[joint_q.shape[0], self.n_dofs],
                inputs=[self.active_mask, self._residual_scale, start_idx],
                outputs=[jacobian],
                device=self.device,
            )


else:

    class JointNominalObjective:  # type: ignore[no-redef]
        """Import-safe placeholder used when Newton/Warp are unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NewtonObjectiveUnavailableError(
                "JointNominalObjective requires newton and warp; optional dependency import "
                f"failed with {type(_NEWTON_IMPORT_ERROR).__name__}: {_NEWTON_IMPORT_ERROR}"
            ) from _NEWTON_IMPORT_ERROR


__all__ = [
    "JointNominalObjective",
    "NewtonObjectiveUnavailableError",
    "ScalarJointLayoutError",
    "build_scalar_dof_to_coord_map",
    "newton_objective_available",
]
