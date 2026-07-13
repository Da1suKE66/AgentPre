"""Deterministic rigid-transform helpers used by the project.

Project configuration and all public APIs in this module use quaternions in
``wxyz`` order.  Backend adapters (for example, Newton bindings that expect
``xyzw``) must perform that conversion explicitly at their boundary.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


_EPS = 1.0e-12


class TransformError(ValueError):
    """Raised when a vector, quaternion, or transform is malformed."""


def _finite_vector(values: Iterable[float], size: int, name: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TransformError(f"{name} must contain {size} numeric values") from exc
    if vector.shape != (size,):
        raise TransformError(f"{name} must have shape ({size},), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise TransformError(f"{name} must contain only finite values")
    return vector


def _rotation_matrix(matrix: np.ndarray, name: str = "rotation") -> np.ndarray:
    try:
        rotation = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TransformError(f"{name} must be a numeric 3x3 matrix") from exc
    if rotation.shape != (3, 3):
        raise TransformError(f"{name} must have shape (3, 3), got {rotation.shape}")
    if not np.isfinite(rotation).all():
        raise TransformError(f"{name} must contain only finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-7, rtol=0.0):
        raise TransformError(f"{name} must be orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, abs_tol=1.0e-7):
        raise TransformError(f"{name} must have determinant +1, got {determinant}")
    return rotation


def _rigid_transform(matrix: np.ndarray, name: str = "transform") -> np.ndarray:
    try:
        transform = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TransformError(f"{name} must be a numeric 4x4 matrix") from exc
    if transform.shape != (4, 4):
        raise TransformError(f"{name} must have shape (4, 4), got {transform.shape}")
    if not np.isfinite(transform).all():
        raise TransformError(f"{name} must contain only finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-9, rtol=0.0):
        raise TransformError(f"{name} must have homogeneous last row [0, 0, 0, 1]")
    _rotation_matrix(transform[:3, :3], f"{name} rotation")
    return transform


def normalize_quaternion(quaternion_wxyz: Iterable[float]) -> np.ndarray:
    """Return a unit quaternion in ``wxyz`` order.

    A zero quaternion has no defined rotation and is rejected instead of being
    silently converted to identity.
    """

    quaternion = _finite_vector(quaternion_wxyz, 4, "quaternion_wxyz")
    norm = float(np.linalg.norm(quaternion))
    if norm <= _EPS:
        raise TransformError("quaternion_wxyz must have non-zero norm")
    return quaternion / norm


def quaternion_conjugate(quaternion_wxyz: Iterable[float]) -> np.ndarray:
    """Return the conjugate of a unit quaternion in ``wxyz`` order."""

    w, x, y, z = normalize_quaternion(quaternion_wxyz)
    return np.asarray([w, -x, -y, -z], dtype=float)


def quaternion_multiply(
    lhs_wxyz: Iterable[float], rhs_wxyz: Iterable[float]
) -> np.ndarray:
    """Compose two rotations as ``lhs * rhs`` and return a unit ``wxyz`` quaternion."""

    w1, x1, y1, z1 = normalize_quaternion(lhs_wxyz)
    w2, x2, y2, z2 = normalize_quaternion(rhs_wxyz)
    result = np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )
    return normalize_quaternion(result)


def quaternion_to_matrix(quaternion_wxyz: Iterable[float]) -> np.ndarray:
    """Convert a quaternion in ``wxyz`` order to a 3x3 rotation matrix."""

    w, x, y, z = normalize_quaternion(quaternion_wxyz)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a canonical ``wxyz`` quaternion."""

    matrix = _rotation_matrix(rotation)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(max(0.0, trace + 1.0))
        quaternion = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    elif matrix[0, 0] >= matrix[1, 1] and matrix[0, 0] >= matrix[2, 2]:
        scale = 2.0 * math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]))
        quaternion = np.asarray(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            ]
        )
    elif matrix[1, 1] >= matrix[2, 2]:
        scale = 2.0 * math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]))
        quaternion = np.asarray(
            [
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            ]
        )
    else:
        scale = 2.0 * math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]))
        quaternion = np.asarray(
            [
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            ]
        )
    quaternion = normalize_quaternion(quaternion)
    # q and -q encode the same rotation.  A non-negative scalar term makes
    # serialization and test comparisons deterministic in the common case.
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return quaternion


def rpy_to_matrix(rpy: Iterable[float]) -> np.ndarray:
    """Convert URDF fixed-axis roll, pitch, yaw angles to a rotation matrix.

    URDF uses rotations about fixed X, Y, then Z axes, represented by
    ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    """

    roll, pitch, yaw = _finite_vector(rpy, 3, "rpy")
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def pose_matrix(position: Iterable[float], orientation_wxyz: Iterable[float]) -> np.ndarray:
    """Create a 4x4 local-to-parent transform from position and ``wxyz`` quaternion."""

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = quaternion_to_matrix(orientation_wxyz)
    transform[:3, 3] = _finite_vector(position, 3, "position")
    return transform


def decompose_pose(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(position, orientation_wxyz)`` from a rigid 4x4 transform."""

    matrix = _rigid_transform(transform)
    return matrix[:3, 3].copy(), matrix_to_quaternion(matrix[:3, :3])


def compose_transforms(*transforms: np.ndarray) -> np.ndarray:
    """Compose rigid transforms from left to right.

    ``compose_transforms(T_world_a, T_a_b)`` returns ``T_world_b``.  With no
    arguments the function returns identity.
    """

    result = np.eye(4, dtype=float)
    for index, transform in enumerate(transforms):
        result = result @ _rigid_transform(transform, f"transforms[{index}]")
    return result


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform without a general matrix inverse."""

    matrix = _rigid_transform(transform)
    rotation_t = matrix[:3, :3].T
    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = rotation_t
    inverse[:3, 3] = -(rotation_t @ matrix[:3, 3])
    return inverse


def transform_point(transform: np.ndarray, point: Iterable[float]) -> np.ndarray:
    """Apply a rigid transform to one 3D point."""

    matrix = _rigid_transform(transform)
    value = _finite_vector(point, 3, "point")
    return matrix[:3, :3] @ value + matrix[:3, 3]


def orientation_error_degrees(
    actual_wxyz: Iterable[float], target_wxyz: Iterable[float]
) -> float:
    """Return the shortest angular distance between two ``wxyz`` quaternions."""

    actual = normalize_quaternion(actual_wxyz)
    target = normalize_quaternion(target_wxyz)
    dot = min(1.0, max(-1.0, abs(float(np.dot(actual, target)))))
    return math.degrees(2.0 * math.acos(dot))
