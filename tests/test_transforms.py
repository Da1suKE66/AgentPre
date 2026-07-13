from __future__ import annotations

import math
import unittest

import numpy as np

from src.transforms import (
    TransformError,
    compose_transforms,
    decompose_pose,
    invert_transform,
    matrix_to_quaternion,
    normalize_quaternion,
    orientation_error_degrees,
    pose_matrix,
    quaternion_multiply,
    quaternion_to_matrix,
    rpy_to_matrix,
    transform_point,
)


class TransformTests(unittest.TestCase):
    def test_quaternion_matrix_round_trip_uses_wxyz(self) -> None:
        quaternion_wxyz = normalize_quaternion([0.7, -0.2, 0.3, 0.6])
        rotation = quaternion_to_matrix(quaternion_wxyz)
        recovered_wxyz = matrix_to_quaternion(rotation)
        self.assertAlmostEqual(abs(float(np.dot(quaternion_wxyz, recovered_wxyz))), 1.0, places=12)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)

    def test_rpy_follows_urdf_fixed_axis_convention(self) -> None:
        rotation = rpy_to_matrix([0.0, 0.0, math.pi / 2.0])
        np.testing.assert_allclose(
            rotation @ np.asarray([1.0, 0.0, 0.0]),
            [0.0, 1.0, 0.0],
            atol=1.0e-12,
        )

    def test_pose_compose_inverse_and_point(self) -> None:
        world_a = pose_matrix(
            [1.0, 2.0, 3.0],
            [math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)],
        )
        a_b = pose_matrix([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        world_b = compose_transforms(world_a, a_b)
        np.testing.assert_allclose(world_b[:3, 3], [1.0, 3.0, 3.0], atol=1.0e-12)
        np.testing.assert_allclose(
            compose_transforms(world_b, invert_transform(world_b)),
            np.eye(4),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            transform_point(world_a, [1.0, 0.0, 0.0]),
            [1.0, 3.0, 3.0],
            atol=1.0e-12,
        )
        position, quaternion_wxyz = decompose_pose(world_b)
        np.testing.assert_allclose(position, [1.0, 3.0, 3.0], atol=1.0e-12)
        np.testing.assert_allclose(
            quaternion_to_matrix(quaternion_wxyz), world_b[:3, :3], atol=1.0e-12
        )

    def test_quaternion_composition(self) -> None:
        quarter_turn_z_wxyz = [math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)]
        half_turn_z_wxyz = quaternion_multiply(quarter_turn_z_wxyz, quarter_turn_z_wxyz)
        np.testing.assert_allclose(
            quaternion_to_matrix(half_turn_z_wxyz) @ [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            atol=1.0e-12,
        )

    def test_orientation_error_uses_shortest_rotation(self) -> None:
        self.assertAlmostEqual(
            orientation_error_degrees([1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]),
            0.0,
        )
        self.assertAlmostEqual(
            orientation_error_degrees([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]),
            180.0,
        )

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(TransformError, "non-zero norm"):
            normalize_quaternion([0.0, 0.0, 0.0, 0.0])
        with self.assertRaisesRegex(TransformError, "finite"):
            pose_matrix([0.0, float("nan"), 0.0], [1.0, 0.0, 0.0, 0.0])
        with self.assertRaisesRegex(TransformError, "orthonormal"):
            matrix_to_quaternion(np.ones((3, 3)))
        invalid_homogeneous = np.eye(4)
        invalid_homogeneous[3, 3] = 2.0
        with self.assertRaisesRegex(TransformError, "homogeneous last row"):
            invert_transform(invalid_homogeneous)


if __name__ == "__main__":
    unittest.main()
