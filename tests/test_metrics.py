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
            ["pregrasp", "approach", "close", "actuate", "actuate", "retreat"]
        )
        door_angle = np.asarray([0.0, 0.0, 0.0, 0.5, 1.0, 1.0])
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

    def test_grasp_drift_uses_only_close_and_actuate(self) -> None:
        phase_names = np.asarray(["pregrasp", "close", "actuate", "retreat"])
        handle_world = np.stack([pose(), pose(), pose((0.2, 0.1, 0.0), 30.0), pose()])
        nominal_relative = pose((0.1, 0.0, 0.0), 0.0)
        target = np.stack(
            [compose_transforms(handle, nominal_relative) for handle in handle_world]
        )
        achieved = target.copy()
        achieved[0] = pose((5.0, 0.0, 0.0), 90.0)  # excluded from grasp drift
        achieved[2] = compose_transforms(
            handle_world[2], pose((0.12, 0.0, 0.0), 5.0)
        )
        achieved[3] = pose((-5.0, 0.0, 0.0), -90.0)  # excluded from grasp drift
        metrics = compute_metrics(
            phase_names=phase_names,
            door_angle_rad=np.asarray([0.0, 0.0, 1.0, 1.0]),
            handle_world=handle_world,
            target_gripper_world=target,
            achieved_gripper_world=achieved,
            joint_q=np.zeros((4, 1)),
            joint_lower=np.asarray([-1.0]),
            joint_upper=np.asarray([1.0]),
            collision_flags=np.zeros(4, dtype=bool),
            ik_success_flags=np.ones(4, dtype=bool),
            target_door_angle_rad=1.0,
            thresholds=thresholds(
                position_error_m=10.0,
                orientation_error_deg=180.0,
                grasp_position_drift_m=0.01,
                grasp_orientation_drift_deg=4.0,
            ),
        )
        self.assertEqual(metrics["grasp_drift_phase_frame_count"], 2)
        self.assertEqual(metrics["grasp_drift_valid_frame_count"], 2)
        self.assertAlmostEqual(metrics["max_handle_gripper_position_drift_m"], 0.02)
        self.assertAlmostEqual(metrics["max_handle_gripper_orientation_drift_deg"], 5.0)
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
        self.assertFalse(metrics["gates"]["nan_free"]["passed"])
        self.assertEqual(metrics["ee_error_valid_frame_count"], 1)
        self.assertEqual(metrics["grasp_drift_valid_frame_count"], 1)
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


if __name__ == "__main__":
    unittest.main()
