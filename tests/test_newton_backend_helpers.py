from __future__ import annotations

import math
import unittest

import numpy as np

from src.errors import FailureCode, PipelineError
from src.ik_objectives import (
    JointNominalObjective,
    NewtonObjectiveUnavailableError,
    ScalarJointLayoutError,
    build_scalar_dof_to_coord_map,
    newton_objective_available,
)
from src.newton_backend import (
    compute_pose_error,
    joint_limit_violations,
    newton_backend_available,
    quaternion_angle_rad_xyzw,
    quaternion_wxyz_to_xyzw,
    quaternion_xyzw_to_wxyz,
    require_newton_backend,
    resolve_unique_label,
    validate_waypoint_solution,
)


class QuaternionBoundaryTests(unittest.TestCase):
    def test_wxyz_xyzw_round_trip_is_explicit(self) -> None:
        project_quaternion = np.asarray([0.5, 0.1, -0.2, 0.8])
        backend_quaternion = quaternion_wxyz_to_xyzw(project_quaternion)
        np.testing.assert_array_equal(backend_quaternion, [0.1, -0.2, 0.8, 0.5])
        np.testing.assert_array_equal(
            quaternion_xyzw_to_wxyz(backend_quaternion), project_quaternion
        )

    def test_orientation_distance_uses_shortest_equivalent_quaternion(self) -> None:
        identity_xyzw = [0.0, 0.0, 0.0, 1.0]
        self.assertAlmostEqual(
            quaternion_angle_rad_xyzw(identity_xyzw, [0.0, 0.0, 0.0, -1.0]),
            0.0,
        )
        self.assertAlmostEqual(
            quaternion_angle_rad_xyzw(identity_xyzw, [1.0, 0.0, 0.0, 0.0]),
            math.pi,
        )

    def test_pose_error_is_measured_from_realized_values(self) -> None:
        error = compute_pose_error(
            [0.03, 0.04, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )
        self.assertAlmostEqual(error.position_m, 0.05)
        self.assertAlmostEqual(error.orientation_rad, 0.0)


class NameResolutionTests(unittest.TestCase):
    def test_exact_or_prefixed_suffix_must_resolve_uniquely(self) -> None:
        exact = resolve_unique_label(["base", "tool"], "tool", kind="body")
        self.assertEqual((exact.index, exact.label), (1, "tool"))
        prefixed = resolve_unique_label(
            ["franka/base", "franka/tool"], "tool", kind="body"
        )
        self.assertEqual((prefixed.index, prefixed.label), (1, "franka/tool"))

    def test_ambiguous_suffix_is_rejected_instead_of_taking_first(self) -> None:
        with self.assertRaises(PipelineError) as context:
            resolve_unique_label(
                ["robot_a/tool", "robot_b/tool"], "tool", kind="body"
            )
        self.assertEqual(context.exception.code, FailureCode.NAME_NOT_UNIQUE)
        self.assertEqual(len(context.exception.details["matches"]), 2)

    def test_missing_label_is_structured(self) -> None:
        with self.assertRaises(PipelineError) as context:
            resolve_unique_label(["robot/base"], "tool", kind="body")
        self.assertEqual(context.exception.code, FailureCode.FRAME_MISSING)


class ScalarLayoutTests(unittest.TestCase):
    def test_fixed_revolute_prismatic_layout_maps_dofs_to_coordinates(self) -> None:
        mapping = build_scalar_dof_to_coord_map(
            joint_q_start=[0, 0, 1, 2],
            joint_qd_start=[0, 0, 1, 2],
            joint_dof_dim=[[0, 0], [0, 1], [1, 0]],
        )
        np.testing.assert_array_equal(mapping, np.asarray([0, 1], dtype=np.int32))

    def test_quaternion_coordinate_joint_is_rejected(self) -> None:
        with self.assertRaisesRegex(ScalarJointLayoutError, "scalar-coordinate"):
            build_scalar_dof_to_coord_map(
                joint_q_start=[0, 7],
                joint_qd_start=[0, 6],
                joint_dof_dim=[[3, 3]],
            )


class TrueValidationTests(unittest.TestCase):
    def test_joint_limit_check_uses_dof_to_coord_map_and_explicit_tolerance(self) -> None:
        violations = joint_limit_violations(
            joint_q=[99.0, 1.12, -0.02],
            joint_limit_lower=[-1.0, -0.01],
            joint_limit_upper=[1.0, 0.01],
            dof_to_coord=[1, 2],
            tolerance=0.05,
        )
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].coord_index, 1)
        self.assertAlmostEqual(violations[0].magnitude, 0.12)

    def test_waypoint_acceptance_requires_pose_and_hard_limits(self) -> None:
        validation = validate_waypoint_solution(
            joint_q=[0.25],
            actual_position=[0.0, 0.0, 0.0],
            actual_orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
            target_position=[0.0, 0.0, 0.0],
            target_orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
            joint_limit_lower=[-1.0],
            joint_limit_upper=[1.0],
            dof_to_coord=[0],
            position_tolerance_m=0.01,
            orientation_tolerance_rad=0.1,
            joint_limit_tolerance=0.0,
        )
        self.assertTrue(validation.success)
        self.assertEqual(validation.failed_checks, ())

    def test_pose_cost_is_not_treated_as_success(self) -> None:
        validation = validate_waypoint_solution(
            joint_q=[0.0],
            actual_position=[0.2, 0.0, 0.0],
            actual_orientation_xyzw=[1.0, 0.0, 0.0, 0.0],
            target_position=[0.0, 0.0, 0.0],
            target_orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
            joint_limit_lower=[-1.0],
            joint_limit_upper=[1.0],
            dof_to_coord=[0],
            position_tolerance_m=0.01,
            orientation_tolerance_rad=0.1,
            joint_limit_tolerance=0.0,
        )
        self.assertFalse(validation.success)
        self.assertEqual(
            validation.failed_checks, ("position_error", "orientation_error")
        )
        self.assertEqual(validation.failure_codes, (FailureCode.IK_UNREACHABLE,))

    def test_joint_limit_and_nan_are_explicit_failures(self) -> None:
        limited = validate_waypoint_solution(
            joint_q=[1.2],
            actual_position=[0.0, 0.0, 0.0],
            actual_orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
            target_position=[0.0, 0.0, 0.0],
            target_orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
            joint_limit_lower=[-1.0],
            joint_limit_upper=[1.0],
            dof_to_coord=[0],
            position_tolerance_m=0.01,
            orientation_tolerance_rad=0.1,
            joint_limit_tolerance=0.0,
        )
        self.assertEqual(limited.failure_codes, (FailureCode.JOINT_LIMIT,))

        nonfinite = validate_waypoint_solution(
            joint_q=[math.nan],
            actual_position=[math.nan, 0.0, 0.0],
            actual_orientation_xyzw=[math.nan, 0.0, 0.0, 1.0],
            target_position=[0.0, 0.0, 0.0],
            target_orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
            joint_limit_lower=[-1.0],
            joint_limit_upper=[1.0],
            dof_to_coord=[0],
            position_tolerance_m=0.01,
            orientation_tolerance_rad=0.1,
            joint_limit_tolerance=0.0,
        )
        self.assertTrue(nonfinite.has_nonfinite)
        self.assertIn(FailureCode.NUMERICAL_INSTABILITY, nonfinite.failure_codes)


class OptionalDependencyTests(unittest.TestCase):
    def test_import_is_safe_without_newton(self) -> None:
        self.assertEqual(newton_backend_available(), newton_objective_available())
        if newton_backend_available():
            return
        with self.assertRaises(PipelineError) as context:
            require_newton_backend()
        self.assertEqual(context.exception.code, FailureCode.PHYSICS_UNAVAILABLE)
        with self.assertRaises(NewtonObjectiveUnavailableError):
            JointNominalObjective([], [], [], cost_weight=0.0)


if __name__ == "__main__":
    unittest.main()
