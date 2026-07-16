from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from src.errors import FailureCode, PipelineError
from src.ik_objectives import (
    JointNominalObjective,
    JointReferenceObjective,
    NewtonObjectiveUnavailableError,
    ScalarJointLayoutError,
    build_scalar_dof_to_coord_map,
    newton_objective_available,
)
from src.newton_backend import (
    JointMotionInfeasibleError,
    NewtonFrankaIKBackend,
    compute_pose_error,
    joint_limit_violations,
    joint_velocity_violations,
    newton_backend_available,
    quaternion_angle_rad_xyzw,
    quaternion_wxyz_to_xyzw,
    project_scalar_joint_limits,
    project_scalar_joint_motion_limits,
    project_scalar_joint_velocity_limits,
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
    def test_hard_limit_projection_uses_dof_to_coordinate_mapping(self) -> None:
        projected = project_scalar_joint_limits(
            joint_q=[9.0, -1.000006, 2.000004],
            joint_limit_lower=[-1.0, -2.0],
            joint_limit_upper=[1.0, 2.0],
            dof_to_coord=[1, 2],
        )
        np.testing.assert_array_equal(projected, [9.0, -1.0, 2.0])
        preserved_nan = project_scalar_joint_limits(
            [math.nan], [-1.0], [1.0], [0]
        )
        self.assertTrue(math.isnan(float(preserved_nan[0])))

    def test_velocity_projection_hard_limits_each_mapped_scalar_dof(self) -> None:
        projected = project_scalar_joint_velocity_limits(
            candidate_joint_q=[99.0, 1.0, -1.0],
            previous_joint_q=[99.0, 0.0, 0.0],
            joint_velocity_limits=[2.0, 4.0],
            dof_to_coord=[1, 2],
            dt_s=0.1,
        )
        np.testing.assert_allclose(projected, [99.0, 0.2, -0.4])

    def test_velocity_projection_only_changes_active_dofs(self) -> None:
        projected = project_scalar_joint_velocity_limits(
            candidate_joint_q=[1.0, -1.0],
            previous_joint_q=[0.0, 0.0],
            joint_velocity_limits=[2.0, 4.0],
            dof_to_coord=[0, 1],
            dt_s=0.1,
            active_mask=[1.0, 0.0],
        )
        np.testing.assert_allclose(projected, [0.2, -1.0])

    def test_raw_velocity_violation_reports_limit_ratio(self) -> None:
        violations = joint_velocity_violations(
            candidate_joint_q=[0.1, -0.3],
            previous_joint_q=[0.0, 0.0],
            joint_velocity_limits=[2.0, 1.0],
            dof_to_coord=[0, 1],
            dt_s=0.1,
        )
        self.assertEqual(len(violations), 1)
        violation = violations[0]
        self.assertEqual((violation.dof_index, violation.coord_index), (1, 1))
        self.assertAlmostEqual(violation.max_delta, 0.1)
        self.assertAlmostEqual(violation.requested_velocity, 3.0)
        self.assertAlmostEqual(violation.limit_ratio, 3.0)

    def test_velocity_projection_rejects_invalid_timing_or_limits(self) -> None:
        with self.assertRaisesRegex(ValueError, "dt_s"):
            project_scalar_joint_velocity_limits(
                [0.0], [0.0], [1.0], [0], dt_s=0.0
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            project_scalar_joint_velocity_limits(
                [0.0], [0.0], [-1.0], [0], dt_s=0.1
            )

    def test_motion_projection_enforces_acceleration_and_jerk_from_rest(self) -> None:
        projection = project_scalar_joint_motion_limits(
            candidate_joint_q=[99.0, 1.0],
            previous_joint_q=[99.0, 0.0],
            previous_joint_velocity=[0.0],
            previous_joint_acceleration=[0.0],
            joint_position_lower=[-2.0],
            joint_position_upper=[2.0],
            joint_velocity_limits=[5.0],
            dof_to_coord=[1],
            dt_s=0.1,
            max_acceleration_rad_s2=2.0,
            max_jerk_rad_s3=5.0,
        )
        self.assertEqual(projection.projected_dof_indices, (0,))
        self.assertAlmostEqual(projection.joint_q[0], 99.0)
        self.assertLessEqual(abs(projection.joint_velocity[0]), 5.0)
        self.assertLessEqual(abs(projection.joint_acceleration[0]), 2.0)
        self.assertLessEqual(abs(projection.joint_jerk[0]), 5.0)
        # Starting at zero acceleration, jerk is the active first-step bound:
        # a <= 5 * 0.1, v <= a * 0.1, q <= v * 0.1.
        self.assertAlmostEqual(projection.joint_q[1], 0.005, places=8)

    def test_motion_projection_state_is_continuous_across_steps(self) -> None:
        dt = 0.1
        q = np.asarray([0.0])
        velocity = np.asarray([0.0])
        acceleration = np.asarray([0.0])
        q_history = [float(q[0])]
        returned_velocity: list[float] = []
        returned_acceleration: list[float] = []
        returned_jerk: list[float] = []

        for _ in range(5):
            projection = project_scalar_joint_motion_limits(
                candidate_joint_q=[10.0],
                previous_joint_q=q,
                previous_joint_velocity=velocity,
                previous_joint_acceleration=acceleration,
                joint_position_lower=[-20.0],
                joint_position_upper=[20.0],
                joint_velocity_limits=[3.0],
                dof_to_coord=[0],
                dt_s=dt,
                max_acceleration_rad_s2=2.0,
                max_jerk_rad_s3=5.0,
            )
            q = projection.joint_q
            velocity = projection.joint_velocity
            acceleration = projection.joint_acceleration
            q_history.append(float(q[0]))
            returned_velocity.append(float(velocity[0]))
            returned_acceleration.append(float(acceleration[0]))
            returned_jerk.append(float(projection.joint_jerk[0]))

        finite_difference_velocity = np.diff(q_history) / dt
        finite_difference_acceleration = np.diff(
            np.concatenate(([0.0], finite_difference_velocity))
        ) / dt
        finite_difference_jerk = np.diff(
            np.concatenate(([0.0], finite_difference_acceleration))
        ) / dt
        np.testing.assert_allclose(
            returned_velocity, finite_difference_velocity, atol=1.0e-12
        )
        np.testing.assert_allclose(
            returned_acceleration, finite_difference_acceleration, atol=1.0e-12
        )
        np.testing.assert_allclose(
            returned_jerk, finite_difference_jerk, atol=1.0e-10
        )
        self.assertLessEqual(np.max(np.abs(finite_difference_velocity)), 3.0)
        self.assertLessEqual(np.max(np.abs(finite_difference_acceleration)), 2.0)
        self.assertLessEqual(np.max(np.abs(finite_difference_jerk)), 5.0)

    def test_motion_projection_float32_round_trip_respects_strict_limits(self) -> None:
        dt = 0.0166666667
        acceleration_limit = 7.5
        jerk_limit = 450.0
        projection = project_scalar_joint_motion_limits(
            candidate_joint_q=[10.0],
            previous_joint_q=[0.0],
            previous_joint_velocity=[0.0],
            previous_joint_acceleration=[0.0],
            joint_position_lower=[-20.0],
            joint_position_upper=[20.0],
            joint_velocity_limits=[2.175],
            dof_to_coord=[0],
            dt_s=dt,
            max_acceleration_rad_s2=acceleration_limit,
            max_jerk_rad_s3=jerk_limit,
        )

        realized_q = float(projection.joint_q[0])
        self.assertEqual(realized_q, float(np.float32(realized_q)))
        realized_velocity = realized_q / dt
        realized_acceleration = realized_velocity / dt
        realized_jerk = realized_acceleration / dt
        self.assertLessEqual(abs(realized_velocity), 2.175)
        self.assertLessEqual(abs(realized_acceleration), acceleration_limit)
        self.assertLessEqual(abs(realized_jerk), jerk_limit)
        self.assertAlmostEqual(
            float(projection.joint_velocity[0]), realized_velocity
        )
        self.assertAlmostEqual(
            float(projection.joint_acceleration[0]), realized_acceleration
        )
        self.assertAlmostEqual(float(projection.joint_jerk[0]), realized_jerk)
        diagnostic = projection.diagnostics[0]
        self.assertIn("float32_quantization", diagnostic.trigger_reasons)
        self.assertEqual(diagnostic.projected_q, realized_q)

    def test_motion_projection_multi_step_state_uses_float32_realized_q(self) -> None:
        dt = 0.0166666667
        q = np.asarray([0.0])
        velocity = np.asarray([0.0])
        acceleration = np.asarray([0.0])

        for step in range(60):
            target = 0.5 if (step // 10) % 2 == 0 else -0.5
            previous_q = q.copy()
            previous_velocity = velocity.copy()
            previous_acceleration = acceleration.copy()
            projection = project_scalar_joint_motion_limits(
                candidate_joint_q=[target],
                previous_joint_q=previous_q,
                previous_joint_velocity=previous_velocity,
                previous_joint_acceleration=previous_acceleration,
                joint_position_lower=[-20.0],
                joint_position_upper=[20.0],
                joint_velocity_limits=[2.175],
                dof_to_coord=[0],
                dt_s=dt,
                max_acceleration_rad_s2=7.5,
                max_jerk_rad_s3=450.0,
            )
            q = projection.joint_q
            velocity = projection.joint_velocity
            acceleration = projection.joint_acceleration
            self.assertEqual(float(q[0]), float(np.float32(q[0])))
            expected_velocity = (q - previous_q) / dt
            expected_acceleration = (expected_velocity - previous_velocity) / dt
            expected_jerk = (expected_acceleration - previous_acceleration) / dt
            np.testing.assert_array_equal(velocity, expected_velocity)
            np.testing.assert_array_equal(acceleration, expected_acceleration)
            np.testing.assert_array_equal(projection.joint_jerk, expected_jerk)
            self.assertLessEqual(abs(float(velocity[0])), 2.175)
            self.assertLessEqual(abs(float(acceleration[0])), 7.5)
            self.assertLessEqual(abs(float(projection.joint_jerk[0])), 450.0)

    def test_motion_projection_reports_acceleration_and_jerk_only_reason(self) -> None:
        projection = project_scalar_joint_motion_limits(
            candidate_joint_q=[0.05],
            previous_joint_q=[0.0],
            previous_joint_velocity=[0.0],
            previous_joint_acceleration=[0.0],
            joint_position_lower=[-1.0],
            joint_position_upper=[1.0],
            joint_velocity_limits=[1.0],
            dof_to_coord=[0],
            dt_s=0.1,
            max_acceleration_rad_s2=2.0,
            max_jerk_rad_s3=5.0,
        )

        diagnostic = projection.diagnostics[0]
        self.assertNotIn("velocity_limit", diagnostic.trigger_reasons)
        self.assertIn("acceleration_limit", diagnostic.trigger_reasons)
        self.assertIn("jerk_limit", diagnostic.trigger_reasons)
        self.assertAlmostEqual(diagnostic.raw_requested_velocity_rad_s, 0.5)
        self.assertAlmostEqual(diagnostic.raw_requested_acceleration_rad_s2, 5.0)
        self.assertAlmostEqual(diagnostic.raw_requested_jerk_rad_s3, 50.0)
        self.assertAlmostEqual(
            diagnostic.q_correction_rad,
            diagnostic.projected_q - diagnostic.raw_candidate_q,
        )

    def test_motion_projection_rejects_interval_without_float32_point(self) -> None:
        lower = 1.0 + 2.0e-8
        upper = 1.0 + 4.0e-8
        with self.assertRaises(JointMotionInfeasibleError) as context:
            project_scalar_joint_motion_limits(
                candidate_joint_q=[1.0 + 3.0e-8],
                previous_joint_q=[0.0],
                previous_joint_velocity=[0.0],
                previous_joint_acceleration=[0.0],
                joint_position_lower=[lower],
                joint_position_upper=[upper],
                joint_velocity_limits=[10.0],
                dof_to_coord=[0],
                dt_s=1.0,
                max_acceleration_rad_s2=10.0,
                max_jerk_rad_s3=10.0,
            )
        self.assertEqual(
            context.exception.details["constraint"], "float32_representability"
        )
        self.assertEqual(
            context.exception.details["feasibility_scope"],
            "one_step_from_previous_state",
        )

    def test_motion_projection_rejects_empty_constraint_intersection(self) -> None:
        with self.assertRaises(JointMotionInfeasibleError) as context:
            project_scalar_joint_motion_limits(
                candidate_joint_q=[2.0],
                previous_joint_q=[0.0],
                previous_joint_velocity=[1.0],
                previous_joint_acceleration=[1.0],
                joint_position_lower=[-10.0],
                joint_position_upper=[10.0],
                joint_velocity_limits=[1.0],
                dof_to_coord=[0],
                dt_s=0.1,
                max_acceleration_rad_s2=2.0,
                max_jerk_rad_s3=1.0,
            )
        self.assertEqual(
            context.exception.details["constraint"],
            "velocity_acceleration_jerk",
        )
        self.assertEqual(context.exception.details["dof_index"], 0)

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


class OrderedWaypointHoldTests(unittest.TestCase):
    def test_repeated_target_brakes_then_holds_without_null_space_drift(self) -> None:
        class FakeArray:
            def __init__(self, values: object) -> None:
                self.values = np.asarray(values, dtype=np.float32).copy()

            def numpy(self) -> np.ndarray:
                return self.values.copy()

        class FakeObjective:
            def set_target_position(self, _index: int, _value: object) -> None:
                return None

            def set_target_rotation(self, _index: int, _value: object) -> None:
                return None

            def set_active_mask(self, _value: object) -> None:
                return None

            def set_reference_q(self, _value: object) -> None:
                return None

        class DriftingSolver:
            def __init__(self) -> None:
                self.step_count = 0
                self.costs = FakeArray([12.5])

            def step(
                self,
                working_input: FakeArray,
                working_output: FakeArray,
                *,
                iterations: int,
                step_size: float,
            ) -> None:
                del iterations, step_size
                self.step_count += 1
                # Models the LM nominal-posture null-space drift that would
                # accumulate if every identical waypoint were optimized.
                working_output.values[...] = working_input.values + 0.1

        fake_wp = SimpleNamespace(
            float32=np.float32,
            array=lambda values, **_kwargs: FakeArray(values),
            vec3=lambda *values: tuple(values),
            vec4=lambda *values: tuple(values),
        )
        solver = DriftingSolver()
        backend = object.__new__(NewtonFrankaIKBackend)
        backend.parameters = SimpleNamespace(
            iterations=8,
            step_size=0.5,
            waypoint_dt_s=0.1,
            max_joint_acceleration_rad_s2=2.0,
            max_joint_jerk_rad_s3=5.0,
            position_tolerance_m=1.0e-6,
            orientation_tolerance_rad=1.0e-6,
            joint_limit_tolerance=0.0,
        )
        backend.model = SimpleNamespace(joint_coord_count=1, joint_dof_count=1)
        backend.device = "cpu"
        backend.nominal_joint_q = np.asarray([0.0], dtype=np.float32)
        backend._velocity_active_mask = np.asarray([1.0])
        backend._joint_velocity_limits = np.asarray([3.0])
        backend._control_limit_lower = np.asarray([-1.0])
        backend._control_limit_upper = np.asarray([1.0])
        backend._joint_limit_lower = np.asarray([-1.0])
        backend._joint_limit_upper = np.asarray([1.0])
        backend.dof_to_coord = np.asarray([0])
        backend.arm_coord_indices = (0,)
        backend.position_objective = FakeObjective()
        backend.rotation_objective = FakeObjective()
        backend.continuity_objective = FakeObjective()
        backend.solver = solver
        fk_calls: list[float] = []

        def fake_fk(joint_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            fk_calls.append(float(joint_q[0]))
            return np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])

        backend._forward_kinematics_tcp = fake_fk
        position_offsets = [0.0, 5.0e-11, -4.0e-11, 0.0, 0.0, 0.0, 0.0]
        positions = [[offset, 0.0, 0.0] for offset in position_offsets]
        orientations = [
            [1.0, 0.0, 0.0, 0.0]
            if index % 2 == 0
            else [-1.0, 0.0, 0.0, 0.0]
            for index in range(len(positions))
        ]

        with mock.patch("src.newton_backend.wp", fake_wp):
            trajectory = backend.solve_waypoints(positions, orientations)

        # Only the first target invokes LM.  Numerically equivalent positions
        # and sign-equivalent quaternions take the stationary-hold path.
        self.assertEqual(solver.step_count, 1)
        self.assertEqual(len(fk_calls), len(positions))
        self.assertTrue(trajectory.all_successful)
        self.assertTrue(
            all(
                math.isfinite(waypoint.objective_cost)
                and waypoint.objective_cost == 12.5
                for waypoint in trajectory.waypoints
            )
        )

        q = np.asarray(
            [0.0]
            + [waypoint.joint_positions[0] for waypoint in trajectory.waypoints]
        )
        velocity = np.diff(q) / backend.parameters.waypoint_dt_s
        acceleration = np.diff(np.concatenate(([0.0], velocity))) / (
            backend.parameters.waypoint_dt_s
        )
        jerk = np.diff(np.concatenate(([0.0], acceleration))) / (
            backend.parameters.waypoint_dt_s
        )
        self.assertLessEqual(np.max(np.abs(velocity)), 3.0)
        self.assertLessEqual(
            np.max(np.abs(acceleration)),
            backend.parameters.max_joint_acceleration_rad_s2,
        )
        self.assertLessEqual(
            np.max(np.abs(jerk)),
            backend.parameters.max_joint_jerk_rad_s3,
        )
        # The projector is allowed two constrained braking steps, then both
        # v/a reach zero and the float32 joint coordinate is held exactly.
        self.assertEqual(float(velocity[-1]), 0.0)
        self.assertEqual(float(acceleration[-1]), 0.0)
        self.assertEqual(len(set(q[-4:].tolist())), 1)


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
        with self.assertRaises(NewtonObjectiveUnavailableError):
            JointReferenceObjective([], [], [], cost_weight=0.0)


if __name__ == "__main__":
    unittest.main()
