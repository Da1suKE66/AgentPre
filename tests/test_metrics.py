from __future__ import annotations

import json
import math
import unittest

import numpy as np

from src.metrics import MetricThresholds, MetricsInputError, compute_metrics
from src.transforms import compose_transforms, pose_matrix


def pose(
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    yaw_deg: float = 0.0,
) -> np.ndarray:
    half = math.radians(yaw_deg) / 2.0
    return pose_matrix(position, [math.cos(half), 0.0, 0.0, math.sin(half)])


def thresholds(**overrides):
    result = {
        "min_ik_success_rate": 0.95,
        "position_error_m": 0.02,
        "orientation_error_deg": 10.0,
        "final_door_angle_deg": 3.0,
        "grasp_position_drift_m": 0.01,
        "grasp_orientation_drift_deg": 5.0,
        "max_joint_limit_violations": 0,
        "max_joint_limit_violation_frame_ratio": 0.0,
        "max_collision_frame_ratio": 0.0,
        "require_nan_free": True,
    }
    result.update(overrides)
    return result


def common_inputs(frame_count: int) -> dict:
    identities = np.repeat(np.eye(4)[None, :, :], frame_count, axis=0)
    return {
        "phase_names": np.asarray(["actuate"] * frame_count),
        "door_angle_rad": np.linspace(0.0, 1.0, frame_count),
        "handle_world": identities.copy(),
        "target_gripper_world": identities.copy(),
        "achieved_gripper_world": identities.copy(),
        "joint_q": np.zeros((frame_count, 2)),
        "joint_lower": np.asarray([-1.0, -1.0]),
        "joint_upper": np.asarray([1.0, 1.0]),
        "collision_flags": np.zeros(frame_count, dtype=bool),
        "ik_success_flags": np.ones(frame_count, dtype=bool),
        "target_door_angle_rad": 1.0,
        "thresholds": thresholds(),
    }


class MetricsTests(unittest.TestCase):
    def test_perfect_rollout_passes_all_configured_gates(self) -> None:
        phase_names = np.asarray(
            [
                "pregrasp",
                "approach",
                "close",
                "actuate",
                "actuate",
                "release",
                "retreat",
            ]
        )
        door_angle = np.asarray([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        handle_world = np.stack(
            [pose((0.1 * index, 0.0, 0.5), math.degrees(angle)) for index, angle in enumerate(door_angle)]
        )
        handle_to_gripper = pose((0.05, -0.01, 0.0), 0.0)
        target = np.stack(
            [compose_transforms(handle, handle_to_gripper) for handle in handle_world]
        )
        metrics = compute_metrics(
            phase_names=phase_names,
            door_angle_rad=door_angle,
            handle_world=handle_world,
            target_gripper_world=target,
            achieved_gripper_world=target.copy(),
            joint_q=np.zeros((len(phase_names), 3)),
            joint_lower=np.asarray([-1.0, -1.0, -1.0]),
            joint_upper=np.asarray([1.0, 1.0, 1.0]),
            collision_flags=np.zeros(len(phase_names), dtype=bool),
            ik_success_flags=np.ones(len(phase_names), dtype=bool),
            target_door_angle_rad=1.0,
            thresholds=thresholds(),
        )
        self.assertTrue(metrics["success"])
        self.assertEqual(metrics["ik_waypoint_success_rate"], 1.0)
        self.assertEqual(metrics["median_ee_position_error_m"], 0.0)
        self.assertEqual(metrics["max_ee_orientation_error_deg"], 0.0)
        self.assertEqual(metrics["joint_limit_violation_count"], 0)
        self.assertEqual(metrics["collision_frame_ratio"], 0.0)
        self.assertAlmostEqual(metrics["final_door_angle_error_deg"], 0.0)
        self.assertAlmostEqual(metrics["max_handle_gripper_position_drift_m"], 0.0)
        self.assertAlmostEqual(metrics["max_handle_gripper_orientation_drift_deg"], 0.0)
        self.assertFalse(metrics["has_nan"])
        self.assertTrue(all(gate["passed"] for gate in metrics["gates"].values()))
        json.dumps(metrics, allow_nan=False)

    def test_error_statistics_limit_counts_and_failed_gates(self) -> None:
        data = common_inputs(3)
        data["phase_names"] = np.asarray(["close", "actuate", "retreat"])
        data["door_angle_rad"] = np.radians([0.0, 20.0, 50.0])
        data["target_door_angle_rad"] = math.radians(60.0)
        data["achieved_gripper_world"] = np.stack(
            [pose((0.0, 0.0, 0.0), 0.0), pose((0.01, 0.0, 0.0), 10.0), pose((0.03, 0.0, 0.0), 20.0)]
        )
        data["joint_q"] = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, -2.0]])
        data["collision_flags"] = np.asarray([False, True, False])
        data["ik_success_flags"] = np.asarray([True, False, False])
        data["thresholds"] = thresholds()
        metrics = compute_metrics(**data)

        self.assertFalse(metrics["success"])
        self.assertAlmostEqual(metrics["ik_waypoint_success_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["median_ee_position_error_m"], 0.01)
        self.assertAlmostEqual(metrics["max_ee_position_error_m"], 0.03)
        self.assertAlmostEqual(metrics["median_ee_orientation_error_deg"], 10.0)
        self.assertAlmostEqual(metrics["max_ee_orientation_error_deg"], 20.0)
        self.assertEqual(metrics["joint_limit_violation_count"], 3)
        self.assertEqual(metrics["joint_limit_violation_frame_count"], 2)
        self.assertAlmostEqual(metrics["joint_limit_violation_frame_ratio"], 2.0 / 3.0)
        self.assertEqual(metrics["collision_frame_count"], 1)
        self.assertAlmostEqual(metrics["collision_frame_ratio"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["final_door_angle_deg"], 50.0)
        self.assertAlmostEqual(metrics["target_door_angle_deg"], 60.0)
        self.assertAlmostEqual(metrics["final_door_angle_error_deg"], 10.0)
        self.assertFalse(metrics["gates"]["ik_waypoint_success_rate"]["passed"])
        self.assertFalse(metrics["gates"]["joint_limit_violation_count"]["passed"])
        self.assertFalse(metrics["gates"]["collision_frame_ratio"]["passed"])
        self.assertFalse(metrics["gates"]["final_door_angle_error_deg"]["passed"])

    def test_grasp_drift_uses_close_actuate_and_release_only(self) -> None:
        phase_names = np.asarray(
            ["pregrasp", "close", "actuate", "release", "retreat"]
        )
        handle_world = np.stack(
            [pose(), pose(), pose((0.2, 0.1, 0.0), 30.0), pose(), pose()]
        )
        nominal_relative = pose((0.1, 0.0, 0.0), 0.0)
        target = np.stack(
            [compose_transforms(handle, nominal_relative) for handle in handle_world]
        )
        achieved = target.copy()
        achieved[0] = pose((5.0, 0.0, 0.0), 90.0)  # excluded from grasp drift
        achieved[2] = compose_transforms(
            handle_world[2], pose((0.12, 0.0, 0.0), 5.0)
        )
        achieved[3] = compose_transforms(
            handle_world[3], pose((0.13, 0.0, 0.0), 6.0)
        )
        achieved[4] = pose((-5.0, 0.0, 0.0), -90.0)  # excluded from grasp drift
        metrics = compute_metrics(
            phase_names=phase_names,
            door_angle_rad=np.asarray([0.0, 0.0, 1.0, 1.0, 1.0]),
            handle_world=handle_world,
            target_gripper_world=target,
            achieved_gripper_world=achieved,
            joint_q=np.zeros((5, 1)),
            joint_lower=np.asarray([-1.0]),
            joint_upper=np.asarray([1.0]),
            collision_flags=np.zeros(5, dtype=bool),
            ik_success_flags=np.ones(5, dtype=bool),
            target_door_angle_rad=1.0,
            thresholds=thresholds(
                position_error_m=10.0,
                orientation_error_deg=180.0,
                grasp_position_drift_m=0.01,
                grasp_orientation_drift_deg=4.0,
            ),
        )
        self.assertEqual(metrics["grasp_drift_phase_frame_count"], 3)
        self.assertEqual(metrics["grasp_drift_valid_frame_count"], 3)
        self.assertAlmostEqual(metrics["max_handle_gripper_position_drift_m"], 0.03)
        self.assertAlmostEqual(metrics["max_handle_gripper_orientation_drift_deg"], 6.0)
        self.assertFalse(metrics["gates"]["max_handle_gripper_position_drift_m"]["passed"])
        self.assertFalse(metrics["gates"]["max_handle_gripper_orientation_drift_deg"]["passed"])

    def test_nan_is_reported_without_putting_nan_in_json(self) -> None:
        data = common_inputs(2)
        data["phase_names"] = np.asarray(["close", "actuate"])
        data["achieved_gripper_world"][1, 1, 3] = np.nan
        data["joint_q"][1, 0] = np.nan
        metrics = compute_metrics(**data)
        self.assertTrue(metrics["has_nan"])
        self.assertFalse(metrics["success"])
        self.assertFalse(metrics["gates"]["finite_free"]["passed"])
        self.assertEqual(metrics["ee_error_valid_frame_count"], 1)
        self.assertEqual(metrics["grasp_drift_valid_frame_count"], 1)
        json.dumps(metrics, allow_nan=False)

    def test_infinity_always_fails_required_finite_gate(self) -> None:
        data = common_inputs(20)
        data["achieved_gripper_world"][0, 0, 3] = np.inf
        data["ik_success_flags"][0] = False
        metrics = compute_metrics(**data)
        self.assertEqual(metrics["ik_waypoint_success_rate"], 0.95)
        self.assertFalse(metrics["has_nan"])
        self.assertTrue(metrics["has_infinite"])
        self.assertFalse(metrics["gates"]["finite_free"]["passed"])
        self.assertFalse(metrics["success"])
        json.dumps(metrics, allow_nan=False)

    def test_all_thresholds_are_required_and_validated(self) -> None:
        missing = thresholds()
        missing.pop("max_collision_frame_ratio")
        with self.assertRaises(MetricsInputError) as raised:
            MetricThresholds.from_mapping(missing)
        self.assertEqual(raised.exception.code, "THRESHOLD_MISSING")

        with self.assertRaises(MetricsInputError) as raised:
            MetricThresholds.from_mapping(thresholds(max_collision_frame_ratio=1.1))
        self.assertEqual(raised.exception.code, "THRESHOLD_INVALID")

        with self.assertRaises(MetricsInputError):
            MetricThresholds.from_mapping(thresholds(max_joint_limit_violations=0.0))
        with self.assertRaises(MetricsInputError):
            MetricThresholds.from_mapping(thresholds(require_nan_free=1))

    def test_shape_and_joint_limit_order_errors_are_structured(self) -> None:
        data = common_inputs(2)
        data["collision_flags"] = np.asarray([False])
        with self.assertRaises(MetricsInputError) as raised:
            compute_metrics(**data)
        self.assertEqual(raised.exception.code, "ARRAY_SHAPE_INVALID")

        data = common_inputs(2)
        data["joint_lower"] = np.asarray([2.0, -1.0])
        with self.assertRaises(MetricsInputError) as raised:
            compute_metrics(**data)
        self.assertEqual(raised.exception.code, "JOINT_LIMIT_ORDER_INVALID")

    def test_joint_limit_tolerance_is_explicit_and_reports_raw_overshoot(self) -> None:
        data = common_inputs(2)
        data["joint_q"][0, 0] = 1.0 + 5.0e-8
        data["joint_q"][1, 0] = 1.0 + 2.0e-7
        data["joint_limit_tolerance_rad"] = 1.0e-7
        metrics = compute_metrics(**data)
        self.assertEqual(metrics["joint_limit_violation_count"], 1)
        self.assertEqual(metrics["joint_limit_violation_frame_count"], 1)
        self.assertAlmostEqual(metrics["joint_limit_tolerance_rad"], 1.0e-7)
        self.assertAlmostEqual(metrics["max_joint_limit_raw_overshoot_rad"], 2.0e-7)
        self.assertFalse(metrics["success"])

        data["joint_limit_tolerance_rad"] = -1.0
        with self.assertRaises(MetricsInputError) as raised:
            compute_metrics(**data)
        self.assertEqual(raised.exception.code, "JOINT_LIMIT_TOLERANCE_INVALID")

    def test_joint_motion_metrics_and_configured_gates(self) -> None:
        data = common_inputs(4)
        data["joint_q"] = np.asarray(
            [
                [0.0, 0.0],
                [1.0, -2.0],
                [4.0, -4.0],
                [9.0, -6.0],
            ]
        )
        data["joint_lower"] = np.asarray([-10.0, -10.0])
        data["joint_upper"] = np.asarray([10.0, 10.0])
        data["sample_dt_s"] = 1.0
        data["initial_joint_q"] = np.asarray([0.0, 0.0])
        data["joint_velocity_limits_rad_s"] = np.asarray([2.5, 1.0])
        data["thresholds"] = thresholds(
            max_joint_velocity_rad_s=4.0,
            max_joint_velocity_limit_ratio=1.0,
            max_joint_acceleration_rad_s2=2.0,
            max_joint_jerk_rad_s3=2.1,
        )
        metrics = compute_metrics(**data)

        self.assertEqual(metrics["trajectory_sample_dt_s"], 1.0)
        self.assertEqual(metrics["max_joint_step_rad"], 5.0)
        self.assertEqual(metrics["per_joint_max_step_rad"], [5.0, 2.0])
        self.assertEqual(metrics["max_joint_step_frame_index"], 3)
        self.assertEqual(metrics["max_joint_step_joint_index"], 0)
        self.assertEqual(metrics["max_joint_velocity_rad_s"], 5.0)
        self.assertEqual(metrics["per_joint_max_velocity_rad_s"], [5.0, 2.0])
        self.assertEqual(metrics["max_joint_velocity_frame_index"], 3)
        self.assertEqual(metrics["max_joint_velocity_joint_index"], 0)
        self.assertEqual(metrics["max_joint_velocity_limit_ratio"], 2.0)
        self.assertEqual(metrics["per_joint_max_velocity_limit_ratio"], [2.0, 2.0])
        self.assertEqual(metrics["max_joint_velocity_limit_ratio_frame_index"], 1)
        self.assertEqual(metrics["max_joint_velocity_limit_ratio_joint_index"], 1)
        self.assertEqual(metrics["max_joint_acceleration_rad_s2"], 2.0)
        self.assertEqual(metrics["per_joint_max_acceleration_rad_s2"], [2.0, 2.0])
        self.assertEqual(metrics["max_joint_acceleration_frame_index"], 1)
        self.assertEqual(metrics["max_joint_jerk_rad_s3"], 2.0)
        self.assertEqual(metrics["per_joint_max_jerk_rad_s3"], [1.0, 2.0])
        self.assertEqual(metrics["joint_velocity_sample_count"], 4)
        self.assertEqual(metrics["joint_acceleration_sample_count"], 4)
        self.assertEqual(metrics["joint_jerk_sample_count"], 4)
        self.assertFalse(metrics["gates"]["max_joint_velocity_rad_s"]["passed"])
        self.assertFalse(
            metrics["gates"]["max_joint_velocity_limit_ratio"]["passed"]
        )
        self.assertTrue(
            metrics["gates"]["max_joint_acceleration_rad_s2"]["passed"]
        )
        self.assertTrue(metrics["gates"]["max_joint_jerk_rad_s3"]["passed"])
        self.assertFalse(metrics["success"])
        json.dumps(metrics, allow_nan=False)

    def test_joint_velocity_gate_catches_one_frame_branch_jump(self) -> None:
        data = common_inputs(5)
        data["joint_q"] = np.zeros((5, 2))
        data["joint_q"][3:, 0] = 1.416
        data["joint_lower"] = np.asarray([-2.0, -2.0])
        data["joint_upper"] = np.asarray([2.0, 2.0])
        data["sample_dt_s"] = 1.0 / 60.0
        data["initial_joint_q"] = np.zeros(2)
        data["joint_velocity_limits_rad_s"] = np.asarray([2.61, 2.61])
        data["thresholds"] = thresholds(max_joint_velocity_limit_ratio=1.0)
        metrics = compute_metrics(**data)

        self.assertAlmostEqual(metrics["max_joint_velocity_rad_s"], 84.96)
        self.assertEqual(metrics["max_joint_velocity_frame_index"], 3)
        self.assertEqual(metrics["max_joint_velocity_joint_index"], 0)
        self.assertAlmostEqual(
            metrics["max_joint_velocity_limit_ratio"], 84.96 / 2.61
        )
        self.assertFalse(
            metrics["gates"]["max_joint_velocity_limit_ratio"]["passed"]
        )
        self.assertFalse(metrics["success"])

    def test_motion_thresholds_require_valid_sample_dt(self) -> None:
        data = common_inputs(4)
        data["thresholds"] = thresholds(max_joint_velocity_rad_s=1.0)
        with self.assertRaises(MetricsInputError) as raised:
            compute_metrics(**data)
        self.assertEqual(raised.exception.code, "TRAJECTORY_DT_REQUIRED")

        data["sample_dt_s"] = 0.0
        data["initial_joint_q"] = np.zeros(2)
        with self.assertRaises(MetricsInputError) as raised:
            compute_metrics(**data)
        self.assertEqual(raised.exception.code, "TRAJECTORY_DT_INVALID")

        for field, value in (
            ("max_joint_velocity_rad_s", 0.0),
            ("max_joint_velocity_limit_ratio", 0.0),
            ("max_joint_acceleration_rad_s2", -1.0),
            ("max_joint_jerk_rad_s3", 0.0),
        ):
            with self.subTest(field=field), self.assertRaises(
                MetricsInputError
            ) as raised:
                MetricThresholds.from_mapping(thresholds(**{field: value}))
            self.assertEqual(raised.exception.code, "THRESHOLD_INVALID")

    def test_initial_to_frame_zero_is_included_in_motion_differences(self) -> None:
        data = common_inputs(3)
        data["joint_q"] = np.zeros((3, 2))
        data["initial_joint_q"] = np.asarray([-3.0, 0.0])
        data["sample_dt_s"] = 1.0
        data["thresholds"] = thresholds(max_joint_velocity_rad_s=2.0)
        metrics = compute_metrics(**data)

        self.assertEqual(metrics["max_joint_step_rad"], 3.0)
        self.assertEqual(metrics["max_joint_step_frame_index"], 0)
        self.assertEqual(metrics["max_joint_velocity_rad_s"], 3.0)
        self.assertEqual(metrics["max_joint_velocity_frame_index"], 0)
        self.assertEqual(metrics["joint_velocity_sample_count"], 3)
        self.assertFalse(metrics["gates"]["max_joint_velocity_rad_s"]["passed"])

    def test_stationary_initial_velocity_and_acceleration_gate_frame_zero(self) -> None:
        data = common_inputs(3)
        data["joint_q"] = np.asarray(
            [
                [0.0, 0.0],
                [3.0, 0.0],
                [6.0, 0.0],
            ]
        )
        data["joint_lower"] = np.asarray([-10.0, -10.0])
        data["joint_upper"] = np.asarray([10.0, 10.0])
        data["initial_joint_q"] = np.asarray([-3.0, 0.0])
        data["sample_dt_s"] = 1.0
        data["thresholds"] = thresholds(
            max_joint_acceleration_rad_s2=2.0,
            max_joint_jerk_rad_s3=2.0,
        )
        metrics = compute_metrics(**data)

        self.assertEqual(metrics["per_joint_max_velocity_rad_s"], [3.0, 0.0])
        self.assertEqual(metrics["max_joint_acceleration_rad_s2"], 3.0)
        self.assertEqual(metrics["max_joint_acceleration_frame_index"], 0)
        self.assertEqual(metrics["max_joint_jerk_rad_s3"], 3.0)
        self.assertEqual(metrics["max_joint_jerk_frame_index"], 0)
        self.assertEqual(metrics["joint_velocity_sample_count"], 3)
        self.assertEqual(metrics["joint_acceleration_sample_count"], 3)
        self.assertEqual(metrics["joint_jerk_sample_count"], 3)
        self.assertFalse(
            metrics["gates"]["max_joint_acceleration_rad_s2"]["passed"]
        )
        self.assertFalse(metrics["gates"]["max_joint_jerk_rad_s3"]["passed"])
        self.assertEqual(
            metrics["gates"]["max_joint_acceleration_rad_s2"]["source"],
            "finite_difference_of_joint_q",
        )
        self.assertEqual(
            metrics["gates"]["max_joint_jerk_rad_s3"]["source"],
            "finite_difference_of_joint_q",
        )
        self.assertTrue(
            metrics["position_difference_acceleration_jerk_gates_enabled"]
        )
        self.assertEqual(
            metrics["position_difference_acceleration_acceptance_role"],
            "acceptance_gate",
        )
        self.assertEqual(
            metrics["position_difference_jerk_acceptance_role"],
            "acceptance_gate",
        )
        self.assertFalse(metrics["success"])

    def test_position_difference_acceleration_and_jerk_can_be_diagnostic_only(
        self,
    ) -> None:
        data = common_inputs(3)
        data["joint_q"] = np.asarray(
            [
                [0.0, 0.0],
                [3.0, 0.0],
                [6.0, 0.0],
            ]
        )
        data["joint_lower"] = np.asarray([-10.0, -10.0])
        data["joint_upper"] = np.asarray([10.0, 10.0])
        data["initial_joint_q"] = np.asarray([-3.0, 0.0])
        data["sample_dt_s"] = 1.0
        data["thresholds"] = thresholds(
            max_joint_velocity_rad_s=4.0,
            max_joint_acceleration_rad_s2=2.0,
            max_joint_jerk_rad_s3=2.0,
        )
        data[
            "include_position_difference_acceleration_jerk_gates"
        ] = False
        metrics = compute_metrics(**data)

        self.assertEqual(metrics["joint_motion_source"], "finite_difference_of_joint_q")
        self.assertEqual(metrics["max_joint_acceleration_rad_s2"], 3.0)
        self.assertEqual(metrics["max_joint_acceleration_frame_index"], 0)
        self.assertEqual(metrics["max_joint_jerk_rad_s3"], 3.0)
        self.assertEqual(metrics["max_joint_jerk_frame_index"], 0)
        self.assertEqual(metrics["joint_acceleration_sample_count"], 3)
        self.assertEqual(metrics["joint_jerk_sample_count"], 3)
        self.assertEqual(
            metrics["thresholds"]["max_joint_acceleration_rad_s2"], 2.0
        )
        self.assertEqual(metrics["thresholds"]["max_joint_jerk_rad_s3"], 2.0)
        self.assertNotIn("max_joint_acceleration_rad_s2", metrics["gates"])
        self.assertNotIn("max_joint_jerk_rad_s3", metrics["gates"])
        self.assertIn("max_joint_velocity_rad_s", metrics["gates"])
        self.assertTrue(metrics["gates"]["max_joint_velocity_rad_s"]["passed"])
        self.assertFalse(
            metrics["position_difference_acceleration_jerk_gates_enabled"]
        )
        self.assertEqual(
            metrics["position_difference_acceleration_acceptance_role"],
            "diagnostic_only",
        )
        self.assertEqual(
            metrics["position_difference_jerk_acceptance_role"],
            "diagnostic_only",
        )
        self.assertTrue(metrics["success"])

        data["joint_upper"] = np.asarray([5.0, 10.0])
        limited = compute_metrics(**data)
        self.assertEqual(limited["joint_limit_violation_count"], 1)
        self.assertFalse(limited["gates"]["joint_limit_violation_count"]["passed"])
        self.assertFalse(
            limited["gates"]["joint_limit_violation_frame_ratio"]["passed"]
        )
        self.assertNotIn("max_joint_acceleration_rad_s2", limited["gates"])
        self.assertNotIn("max_joint_jerk_rad_s3", limited["gates"])
        self.assertFalse(limited["success"])

    def test_motion_gate_fails_when_any_expected_value_is_nonfinite(self) -> None:
        cases = (
            ("max_joint_velocity_rad_s", "joint_velocity"),
            ("max_joint_velocity_limit_ratio", "joint_velocity_limit_ratio"),
            ("max_joint_acceleration_rad_s2", "joint_acceleration"),
            ("max_joint_jerk_rad_s3", "joint_jerk"),
        )
        for gate_name, validity_prefix in cases:
            with self.subTest(gate=gate_name):
                data = common_inputs(4)
                data["joint_q"][2, 0] = np.nan
                data["initial_joint_q"] = np.zeros(2)
                data["joint_velocity_limits_rad_s"] = np.ones(2)
                data["sample_dt_s"] = 1.0
                data["thresholds"] = thresholds(
                    require_nan_free=False,
                    **{gate_name: 100.0},
                )
                metrics = compute_metrics(**data)

                gate = metrics["gates"][gate_name]
                self.assertEqual(metrics[gate_name], 0.0)
                self.assertFalse(metrics[f"{validity_prefix}_all_finite"])
                self.assertLess(
                    metrics[f"{validity_prefix}_valid_value_count"],
                    metrics[f"{validity_prefix}_expected_value_count"],
                )
                self.assertFalse(gate["all_expected_values_finite"])
                self.assertFalse(gate["passed"])
                self.assertFalse(metrics["success"])
                json.dumps(metrics, allow_nan=False)

    def test_motion_inputs_are_required_and_velocity_limits_validated(self) -> None:
        data = common_inputs(3)
        data["sample_dt_s"] = 1.0
        data["thresholds"] = thresholds(max_joint_velocity_limit_ratio=1.0)
        data["joint_velocity_limits_rad_s"] = np.ones(2)
        with self.assertRaises(MetricsInputError) as raised:
            compute_metrics(**data)
        self.assertEqual(raised.exception.code, "INITIAL_JOINT_Q_REQUIRED")

        data["initial_joint_q"] = np.zeros(2)
        data.pop("joint_velocity_limits_rad_s")
        with self.assertRaises(MetricsInputError) as raised:
            compute_metrics(**data)
        self.assertEqual(raised.exception.code, "JOINT_VELOCITY_LIMITS_REQUIRED")

        data["joint_velocity_limits_rad_s"] = np.asarray([2.0, 0.0])
        with self.assertRaises(MetricsInputError) as raised:
            compute_metrics(**data)
        self.assertEqual(raised.exception.code, "JOINT_VELOCITY_LIMITS_INVALID")

    def test_old_callers_may_omit_motion_thresholds_and_sample_dt(self) -> None:
        metrics = compute_metrics(**common_inputs(2))
        self.assertNotIn("max_joint_velocity_rad_s", metrics["gates"])
        self.assertIsNone(metrics["trajectory_sample_dt_s"])
        self.assertFalse(metrics["initial_joint_q_provided"])
        self.assertIsNone(metrics["max_joint_velocity_rad_s"])
        self.assertEqual(metrics["per_joint_max_velocity_rad_s"], [None, None])
        self.assertTrue(metrics["success"])


if __name__ == "__main__":
    unittest.main()
