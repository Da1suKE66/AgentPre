from __future__ import annotations

import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from src.config import ProjectConfig, load_config
from src.errors import FailureCode, PipelineError
from src.physics_assisted import (
    FixedGraspRuntimeCapture,
    NewtonPhysicsAssistedSimulator,
    PlannedFixedGraspAnchors,
    PhysicsCommandTrajectory,
    PhysicsParameters,
    PhysicsRollout,
    RobotJointVelocityLimits,
    ScalarJointRef,
    audit_bumpless_grasp_release_transfer,
    build_robot_position_targets,
    build_robot_velocity_targets,
    build_bumpless_grasp_release_transfer,
    captured_fixed_grasp_parent_anchor,
    fixed_grasp_activation_window,
    evaluate_fixed_grasp_activation_gate,
    handle_frame_world_from_link_poses,
    kinematic_body_twists,
    load_robot_joint_velocity_limits,
    planned_fixed_grasp_anchors,
    plan_massless_fixed_joint_collapse,
    read_fixed_grasp_child_anchor,
    require_newton_v13,
    resolve_named_scalar_joints,
    validate_robot_command_motion_limits,
    write_fixed_grasp_parent_anchor,
    write_fixed_grasp_joint_enabled,
)
from src.run import main
from src.transforms import compose_transforms, decompose_pose, pose_matrix


ROOT = Path(__file__).resolve().parents[1]


class _FakeTransformArray:
    def __init__(self, rows: np.ndarray, *, corrupt_writes: bool = False) -> None:
        self.rows = np.asarray(rows, dtype=np.float32).copy()
        self.corrupt_writes = corrupt_writes
        self.assign_count = 0

    def numpy(self) -> np.ndarray:
        return self.rows.copy()

    def assign(self, values: np.ndarray) -> None:
        self.assign_count += 1
        self.rows = np.asarray(values, dtype=np.float32).copy()
        if self.corrupt_writes:
            self.rows[1, 0] += np.float32(0.01)


class _FakeWarp:
    transform = object()

    @staticmethod
    def array(
        values: np.ndarray, *, dtype: object, device: object
    ) -> np.ndarray:
        if dtype is not _FakeWarp.transform:
            raise AssertionError("fixed-grasp writer used the wrong Warp dtype")
        if device != "cpu":
            raise AssertionError("fixed-grasp writer used the wrong device")
        return np.asarray(values, dtype=np.float32).copy()


class _FakeBoolArray:
    def __init__(self, values: np.ndarray, *, corrupt_writes: bool = False) -> None:
        self.values = np.asarray(values, dtype=bool).copy()
        self.corrupt_writes = corrupt_writes
        self.assign_count = 0

    def numpy(self) -> np.ndarray:
        return self.values.copy()

    def assign(self, values: np.ndarray) -> None:
        self.assign_count += 1
        self.values = np.asarray(values, dtype=bool).copy()
        if self.corrupt_writes:
            self.values[1] = not bool(self.values[1])


class _FakeBoolWarp:
    @staticmethod
    def array(values: np.ndarray, *, dtype: object, device: object) -> np.ndarray:
        if dtype is not bool:
            raise AssertionError("joint-enabled writer used the wrong Warp dtype")
        if device != "cpu":
            raise AssertionError("joint-enabled writer used the wrong device")
        return np.asarray(values, dtype=bool).copy()


def settled_gate_kwargs() -> dict[str, object]:
    return {
        "parent_body_qd": np.zeros(6),
        "child_body_qd": np.zeros(6),
        "parent_body_com": np.zeros(3),
        "child_body_com": np.zeros(3),
        "parent_body_name": "robot/panda_hand",
        "child_body_name": "object/handle",
        "linear_velocity_limit_m_s": 0.02,
        "angular_velocity_limit_deg_s": 10.0,
    }


def ref(name: str, coord: int, dof: int) -> ScalarJointRef:
    return ScalarJointRef(
        configured_name=name,
        label=f"robot/{name}",
        joint_index=coord,
        coord_index=coord,
        dof_index=dof,
    )


class PhysicsTrajectoryTests(unittest.TestCase):
    def test_post_step_state_times_use_frame_endpoints(self) -> None:
        import src.physics_assisted as module

        actual = module._post_step_state_sample_times(4, 0.02)
        expected = (np.arange(4, dtype=float) + 1.0) * 0.02
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
        self.assertEqual(actual[0], 0.02)
        self.assertEqual(actual[-1], 0.08)
        audit = module._physics_safety_audit_metadata()
        self.assertEqual(audit["state_sample_timing"], "post_step_end_of_frame")
        self.assertIs(audit["grasp_parent_child_collision_filtered"], False)

    def test_commands_are_validated_copied_and_read_only(self) -> None:
        phases = np.asarray(["pregrasp", "close", "actuate", "retreat"])
        arm = np.zeros((4, 2))
        width = np.asarray([0.08, 0.03, 0.03, 0.08])
        door = np.linspace(0.0, 1.0, 4)
        commands = PhysicsCommandTrajectory(phases, arm, width, door)

        arm[0, 0] = 99.0
        door[0] = 99.0
        self.assertEqual(commands.arm_joint_targets[0, 0], 0.0)
        self.assertEqual(commands.door_reference_rad[0], 0.0)
        self.assertFalse(commands.arm_joint_targets.flags.writeable)
        self.assertFalse(commands.door_reference_rad.flags.writeable)
        self.assertEqual(commands.frame_count, 4)

    def test_invalid_commands_fail_with_structured_reason(self) -> None:
        with self.assertRaises(PipelineError) as caught:
            PhysicsCommandTrajectory(
                np.asarray(["close", "unknown"]),
                np.zeros((2, 1)),
                np.asarray([0.03, 0.03]),
            )
        self.assertEqual(caught.exception.code, FailureCode.CONFIG_INVALID)
        self.assertEqual(caught.exception.stage, "physics_trajectory")

        with self.assertRaises(PipelineError) as caught:
            PhysicsCommandTrajectory(
                np.asarray(["close", "actuate"]),
                np.zeros((2, 1)),
                np.asarray([0.03, 0.03]),
                np.asarray([0.0]),
            )
        self.assertEqual(caught.exception.code, FailureCode.CONFIG_INVALID)

    def test_fixed_grasp_stays_active_through_release_and_stops_at_retreat(self) -> None:
        phases = np.asarray(
            [
                "pregrasp",
                "approach",
                "close",
                "close",
                "actuate",
                "actuate",
                "release",
                "retreat",
            ]
        )
        window = fixed_grasp_activation_window(phases, "close")
        self.assertEqual(window.activation_frame, 4)
        self.assertEqual(window.release_frame, 7)
        self.assertFalse(window.is_active(3))
        self.assertTrue(window.is_active(4))
        self.assertTrue(window.is_active(5))
        self.assertTrue(window.is_active(6))
        self.assertFalse(window.is_active(7))

    @staticmethod
    def _bumpless_release_fixture() -> tuple[
        np.ndarray, np.ndarray, np.ndarray, object
    ]:
        phases = np.asarray(
            [
                "pregrasp",
                "approach",
                "close",
                "close",
                "actuate",
                "actuate",
                "release",
                "release",
                "release",
                "release",
                "release",
                "retreat",
                "retreat",
                "retreat",
                "retreat",
                "retreat",
                "retreat",
            ]
        )
        planned = np.zeros((len(phases), 2), dtype=float)
        planned[4:11] = [0.2, -0.2]
        planned[11:, 0] = np.linspace(0.21, 0.4, len(phases) - 11)
        planned[11:, 1] = np.linspace(-0.19, -0.1, len(phases) - 11)
        captured = np.asarray([0.24, -0.18], dtype=np.float32)
        window = fixed_grasp_activation_window(phases, "close")
        return phases, planned, captured, window

    def test_bumpless_release_has_bit_exact_endpoints_and_correct_windows(
        self,
    ) -> None:
        phases, planned, captured, window = self._bumpless_release_fixture()
        transfer = build_bumpless_grasp_release_transfer(
            phases, planned, captured, window, 4
        )
        applied = transfer.applied_arm_joint_targets

        self.assertEqual(applied.dtype, np.float32)
        self.assertFalse(applied.flags.writeable)
        self.assertEqual(transfer.equilibrium_capture_frame, 5)
        self.assertEqual(transfer.release_unload_start_frame, 6)
        self.assertEqual(transfer.release_unload_end_frame, 10)
        self.assertEqual(transfer.retreat_rejoin_start_frame, 11)
        self.assertEqual(transfer.retreat_rejoin_end_frame, 14)
        self.assertTrue(window.is_active(transfer.release_unload_start_frame))
        self.assertTrue(window.is_active(transfer.release_unload_end_frame))
        self.assertFalse(window.is_active(transfer.retreat_rejoin_start_frame))
        self.assertFalse(window.is_active(transfer.retreat_rejoin_end_frame))
        np.testing.assert_array_equal(applied[6], planned[6].astype(np.float32))
        np.testing.assert_array_equal(applied[10], captured)
        np.testing.assert_array_equal(applied[11], captured)
        np.testing.assert_array_equal(applied[14], planned[14].astype(np.float32))
        np.testing.assert_array_equal(
            applied[:6], planned[:6].astype(np.float32)
        )
        np.testing.assert_array_equal(
            applied[15:], planned[15:].astype(np.float32)
        )
        metadata = transfer.audit_metadata()
        self.assertTrue(metadata["grasp_release_bumpless_transfer_enabled"])
        self.assertEqual(metadata["grasp_release_blend_frames"], 4)
        self.assertEqual(
            metadata["grasp_release_constraint_disable_frame"],
            window.release_frame,
        )

    def test_bumpless_release_exact_applied_targets_pass_motion_and_reserve_audits(
        self,
    ) -> None:
        phases, planned, captured, window = self._bumpless_release_fixture()
        transfer = build_bumpless_grasp_release_transfer(
            phases, planned, captured, window, 4
        )
        commands = PhysicsCommandTrajectory(
            phases,
            planned,
            np.full(len(phases), 0.04),
        )
        limits = RobotJointVelocityLimits(
            source_urdf=Path("robot.urdf"),
            arm_joint_names=("a", "b"),
            arm_joint_types=("revolute", "revolute"),
            arm_velocity_limits=(10.0, 10.0),
            finger_joint_names=("left", "right"),
            finger_joint_types=("prismatic", "prismatic"),
            finger_velocity_limits=(10.0, 10.0),
        )
        audit = audit_bumpless_grasp_release_transfer(
            transfer,
            commands,
            limits,
            np.asarray([-1.0, -1.0]),
            np.asarray([1.0, 1.0]),
            initial_arm_joint_positions=np.zeros(2),
            initial_gripper_width_m=0.04,
            control_limit_margin_rad=0.02,
            arm_joint_tracking_reserve_rad=0.05,
            dt=1.0,
            max_joint_acceleration_rad_s2=10.0,
            max_joint_jerk_rad_s3=10.0,
            max_finger_acceleration_m_s2=10.0,
            max_finger_jerk_m_s3=10.0,
        )
        self.assertTrue(audit["passed"])
        self.assertTrue(audit["position_limits"]["passed"])
        self.assertTrue(audit["tracking_reserve"]["passed"])
        self.assertTrue(audit["motion_limits"]["passed"])

    def test_bumpless_release_fails_closed_on_measured_reserve_or_control_touch(
        self,
    ) -> None:
        import src.physics_assisted as module

        phases, planned, _, window = self._bumpless_release_fixture()
        commands = PhysicsCommandTrajectory(
            phases, planned, np.full(len(phases), 0.04)
        )
        limits = RobotJointVelocityLimits(
            source_urdf=Path("robot.urdf"),
            arm_joint_names=("a", "b"),
            arm_joint_types=("revolute", "revolute"),
            arm_velocity_limits=(10.0, 10.0),
            finger_joint_names=("left", "right"),
            finger_joint_types=("prismatic", "prismatic"),
            finger_velocity_limits=(10.0, 10.0),
        )

        for label, captured, reserve, expected_check in (
            (
                "hard_reserve",
                np.asarray([0.96, 0.0], dtype=np.float32),
                0.05,
                "hard_limit_clearance",
            ),
            (
                "control_touch",
                np.asarray(
                    [
                        module._inward_float32_position_bounds(
                            np.asarray([-0.9]), np.asarray([0.9])
                        )[1][0],
                        0.0,
                    ],
                    dtype=np.float32,
                ),
                0.01,
                "control_bound_touch",
            ),
        ):
            with self.subTest(label=label):
                transfer = build_bumpless_grasp_release_transfer(
                    phases, planned, captured, window, 4
                )
                with self.assertRaises(PipelineError) as caught:
                    audit_bumpless_grasp_release_transfer(
                        transfer,
                        commands,
                        limits,
                        np.asarray([-1.0, -1.0]),
                        np.asarray([1.0, 1.0]),
                        initial_arm_joint_positions=np.zeros(2),
                        initial_gripper_width_m=0.04,
                        control_limit_margin_rad=0.1,
                        arm_joint_tracking_reserve_rad=reserve,
                        dt=1.0,
                        max_joint_acceleration_rad_s2=10.0,
                        max_joint_jerk_rad_s3=10.0,
                        max_finger_acceleration_m_s2=10.0,
                        max_finger_jerk_m_s3=10.0,
                    )
                self.assertEqual(caught.exception.stage, "ik_motion_limits")
                self.assertIn(expected_check, caught.exception.details["failed_checks"])

    def test_planned_fixed_grasp_anchors_reproduce_authored_relation(self) -> None:
        hand_to_tcp = pose_matrix([0.0, 0.0, 0.1], [1.0, 0.0, 0.0, 0.0])
        link_to_handle = pose_matrix([0.0, 0.0, 0.05], [1.0, 0.0, 0.0, 0.0])
        handle_to_tcp = pose_matrix([0.0, -0.01, 0.0], [1.0, 0.0, 0.0, 0.0])
        anchors = planned_fixed_grasp_anchors(
            hand_to_tcp, link_to_handle, handle_to_tcp
        )
        handle_world = pose_matrix([0.5, 0.0, 0.8], [1.0, 0.0, 0.0, 0.0])
        tcp_world = compose_transforms(handle_world, link_to_handle, handle_to_tcp)
        hand_world = compose_transforms(tcp_world, np.linalg.inv(hand_to_tcp))
        np.testing.assert_allclose(
            compose_transforms(hand_world, anchors.parent_xform),
            compose_transforms(handle_world, anchors.child_xform),
            atol=1.0e-12,
        )

    def test_measured_parent_anchor_is_exactly_coincident_with_authored_child(self) -> None:
        root_half = np.sqrt(0.5)
        parent_world = pose_matrix(
            [0.31, -0.22, 0.64], [root_half, 0.0, 0.0, root_half]
        )
        child_world = pose_matrix(
            [-0.12, 0.41, 0.77], [0.9238795325, 0.3826834324, 0.0, 0.0]
        )
        authored_child = pose_matrix(
            [0.07, -0.03, 0.11], [0.9659258263, 0.0, 0.2588190451, 0.0]
        )

        captured_parent = captured_fixed_grasp_parent_anchor(
            parent_world, child_world, authored_child
        )

        np.testing.assert_allclose(
            compose_transforms(parent_world, captured_parent),
            compose_transforms(child_world, authored_child),
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_measured_parent_anchor_rejects_malformed_transform(self) -> None:
        with self.assertRaises(PipelineError) as caught:
            captured_fixed_grasp_parent_anchor(
                np.eye(3), np.eye(4), np.eye(4)
            )
        self.assertEqual(caught.exception.code, FailureCode.NUMERICAL_INSTABILITY)
        self.assertEqual(caught.exception.stage, "physics_constraint")

    def test_fixed_grasp_parent_anchor_write_is_targeted_and_read_back(self) -> None:
        identity_row = np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
        )
        array = _FakeTransformArray(np.repeat(identity_row[None, :], 3, axis=0))
        model = SimpleNamespace(joint_X_p=array, device="cpu")
        target = pose_matrix(
            [0.14, -0.03, 0.28], [0.9238795325, 0.0, 0.3826834324, 0.0]
        )

        result = write_fixed_grasp_parent_anchor(
            model, 1, target, warp_module=_FakeWarp
        )

        self.assertEqual(array.assign_count, 1)
        self.assertEqual(result.runtime_model_write_count, 1)
        self.assertEqual(result.runtime_model_readback_count, 1)
        self.assertTrue(result.readback_verified)
        np.testing.assert_array_equal(array.rows[0], identity_row)
        np.testing.assert_array_equal(array.rows[2], identity_row)
        np.testing.assert_allclose(result.parent_xform, target, atol=1.0e-7)

        capture = FixedGraspRuntimeCapture(
            frame_index=216,
            parent_xform=result.parent_xform,
            child_xform=np.eye(4),
            runtime_model_write_count=result.runtime_model_write_count,
            runtime_model_readback_count=result.runtime_model_readback_count,
            readback_verified=result.readback_verified,
            post_capture_position_error_m=2.0e-8,
            post_capture_orientation_error_deg=3.0e-6,
        )
        metadata = capture.audit_metadata()
        self.assertIn("runtime_measured", metadata["grasp_anchor_source"])
        self.assertEqual(metadata["grasp_anchor_runtime_model_write_count"], 1)
        self.assertEqual(metadata["grasp_anchor_runtime_model_readback_count"], 1)
        self.assertTrue(
            metadata["grasp_anchor_runtime_model_write_readback_verified"]
        )
        self.assertEqual(
            metadata["grasp_anchor_post_capture_position_error_m"], 2.0e-8
        )
        self.assertIs(metadata["remote_latch_allowed"], False)

    def test_fixed_grasp_parent_anchor_write_fails_on_bad_readback(self) -> None:
        identity_row = np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
        )
        array = _FakeTransformArray(
            np.repeat(identity_row[None, :], 3, axis=0), corrupt_writes=True
        )
        model = SimpleNamespace(joint_X_p=array, device="cpu")

        with self.assertRaises(PipelineError) as caught:
            write_fixed_grasp_parent_anchor(
                model, 1, np.eye(4), warp_module=_FakeWarp
            )

        self.assertEqual(caught.exception.code, FailureCode.NUMERICAL_INSTABILITY)
        self.assertEqual(caught.exception.stage, "physics_constraint")
        self.assertEqual(
            caught.exception.details[
                "grasp_anchor_runtime_model_write_count"
            ],
            1,
        )
        self.assertIs(
            caught.exception.details[
                "grasp_anchor_runtime_model_write_readback_verified"
            ],
            False,
        )
        self.assertGreaterEqual(array.assign_count, 2)

    def test_finalized_child_anchor_and_enabled_transitions_are_verified(self) -> None:
        child_target = pose_matrix(
            [0.04, -0.02, 0.09], [0.9238795325, 0.0, 0.3826834324, 0.0]
        )
        child_position, child_quaternion_wxyz = decompose_pose(child_target)
        child_row = np.concatenate(
            (
                child_position,
                child_quaternion_wxyz[1:],
                child_quaternion_wxyz[:1],
            )
        ).astype(np.float32)
        identity_row = np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
        )
        child_rows = np.repeat(identity_row[None, :], 3, axis=0)
        child_rows[1] = child_row
        enabled = _FakeBoolArray(np.asarray([False, False, True]))
        model = SimpleNamespace(
            joint_X_c=_FakeTransformArray(child_rows),
            joint_enabled=enabled,
            device="cpu",
        )

        child_readback = read_fixed_grasp_child_anchor(model, 1, child_target)
        np.testing.assert_allclose(
            child_readback.child_xform, child_target, rtol=0.0, atol=1.0e-7
        )
        self.assertTrue(child_readback.readback_verified)
        self.assertTrue(child_readback.authored_match_verified)

        initial = write_fixed_grasp_joint_enabled(
            model, 1, False, warp_module=_FakeBoolWarp
        )
        activated = write_fixed_grasp_joint_enabled(
            model, 1, True, warp_module=_FakeBoolWarp
        )
        released = write_fixed_grasp_joint_enabled(
            model, 1, False, warp_module=_FakeBoolWarp
        )
        self.assertFalse(initial.enabled)
        self.assertTrue(activated.enabled)
        self.assertFalse(released.enabled)
        self.assertTrue(initial.readback_verified)
        self.assertTrue(activated.readback_verified)
        self.assertTrue(released.readback_verified)
        self.assertEqual(enabled.assign_count, 3)
        np.testing.assert_array_equal(enabled.values, [False, False, True])

    def test_child_anchor_mismatch_and_enabled_bad_readback_fail_closed(self) -> None:
        identity_row = np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
        )
        model = SimpleNamespace(
            joint_X_c=_FakeTransformArray(
                np.repeat(identity_row[None, :], 2, axis=0)
            ),
            joint_enabled=_FakeBoolArray(
                np.asarray([False, False]), corrupt_writes=True
            ),
            device="cpu",
        )
        with self.assertRaises(PipelineError) as child_error:
            read_fixed_grasp_child_anchor(
                model,
                1,
                pose_matrix([0.01, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            )
        self.assertEqual(child_error.exception.code, FailureCode.NUMERICAL_INSTABILITY)

        with self.assertRaises(PipelineError) as enabled_error:
            write_fixed_grasp_joint_enabled(
                model, 1, True, warp_module=_FakeBoolWarp
            )
        self.assertEqual(
            enabled_error.exception.code, FailureCode.NUMERICAL_INSTABILITY
        )
        self.assertFalse(
            enabled_error.exception.details[
                "runtime_model_write_readback_verified"
            ]
        )

    def test_articraft_same_link_handle_anchor_matches_authored_tcp(self) -> None:
        config = json.loads(
            (ROOT / "configs/articraft_microwave_franka.json").read_text()
        )
        affordances = json.loads(
            (
                ROOT
                / "assets/articraft"
                / "rec_microwave_oven_5e86f3429e954dcd9ab6c9d3a94db707"
                / "affordances.json"
            ).read_text()
        )
        object_config = config["assets"]["object"]
        frame = affordances["frames"][object_config["handle_frame"]]
        self.assertEqual(object_config["handle_link"], object_config["door_link"])
        self.assertEqual(frame["link"], object_config["door_link"])

        robot_offset = config["assets"]["robot"]["end_effector_offset"]
        grasp_offset = config["task"]["grasp_offset"]
        hand_to_tcp = pose_matrix(
            robot_offset["position"], robot_offset["orientation_wxyz"]
        )
        door_to_handle = pose_matrix(frame["position"], frame["quaternion_wxyz"])
        handle_to_tcp = pose_matrix(
            grasp_offset["position"], grasp_offset["orientation_wxyz"]
        )
        anchors = planned_fixed_grasp_anchors(
            hand_to_tcp, door_to_handle, handle_to_tcp
        )

        door_world = pose_matrix(
            [0.14, -0.21, 0.52],
            [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        )
        expected_tcp_world = compose_transforms(
            door_world, door_to_handle, handle_to_tcp
        )
        hand_world = compose_transforms(expected_tcp_world, np.linalg.inv(hand_to_tcp))
        gate = evaluate_fixed_grasp_activation_gate(
            hand_world,
            door_world,
            anchors,
            **settled_gate_kwargs(),
            frame_index=0,
            position_limit_m=1.0e-9,
            orientation_limit_deg=1.0e-7,
        )
        self.assertTrue(gate.passed)
        self.assertLess(gate.position_error_m, 1.0e-12)
        self.assertLess(gate.orientation_error_deg, 1.0e-9)

    def test_planned_grasp_anchors_reject_remote_latch(self) -> None:
        hand_to_tcp = pose_matrix([0.0, 0.0, 0.1], [1.0, 0.0, 0.0, 0.0])
        link_to_handle = pose_matrix([0.0, 0.0, 0.05], [1.0, 0.0, 0.0, 0.0])
        handle_to_tcp = pose_matrix([0.0, -0.01, 0.0], [1.0, 0.0, 0.0, 0.0])
        anchors = planned_fixed_grasp_anchors(
            hand_to_tcp, link_to_handle, handle_to_tcp
        )
        handle_world = pose_matrix([0.5, 0.0, 0.8], [1.0, 0.0, 0.0, 0.0])
        expected_tcp_world = compose_transforms(handle_world, link_to_handle, handle_to_tcp)
        hand_world = compose_transforms(expected_tcp_world, np.linalg.inv(hand_to_tcp))
        passing = evaluate_fixed_grasp_activation_gate(
            hand_world,
            handle_world,
            anchors,
            **settled_gate_kwargs(),
            frame_index=4,
            position_limit_m=0.015,
            orientation_limit_deg=7.5,
        )
        self.assertTrue(passing.passed)
        gate_metadata = passing.audit_metadata()
        self.assertEqual(
            gate_metadata["grasp_activation_gate_relation_source"],
            "planned_handle_frame_to_tcp_relation",
        )
        self.assertNotIn("grasp_anchor_runtime_model_write_count", gate_metadata)
        self.assertIs(gate_metadata["remote_latch_allowed"], False)
        self.assertIn(
            "pose_and_relative_anchor_twist_gates",
            gate_metadata["remote_latch_prevention"],
        )

        remote_hand = hand_world.copy()
        remote_hand[0, 3] += 0.02
        rejected = evaluate_fixed_grasp_activation_gate(
            remote_hand,
            handle_world,
            anchors,
            **settled_gate_kwargs(),
            frame_index=4,
            position_limit_m=0.015,
            orientation_limit_deg=7.5,
        )
        self.assertFalse(rejected.passed)
        self.assertGreater(rejected.position_error_m, 0.015)

    def test_fixed_grasp_rejects_nonzero_relative_anchor_twist(self) -> None:
        hand_to_tcp = pose_matrix([0.0, 0.0, 0.1], [1.0, 0.0, 0.0, 0.0])
        link_to_handle = pose_matrix([0.0, 0.0, 0.05], [1.0, 0.0, 0.0, 0.0])
        handle_to_tcp = pose_matrix([0.0, -0.01, 0.0], [1.0, 0.0, 0.0, 0.0])
        anchors = planned_fixed_grasp_anchors(
            hand_to_tcp, link_to_handle, handle_to_tcp
        )
        child_world = pose_matrix([0.5, 0.0, 0.8], [1.0, 0.0, 0.0, 0.0])
        tcp_world = compose_transforms(
            child_world, link_to_handle, handle_to_tcp
        )
        parent_world = compose_transforms(tcp_world, np.linalg.inv(hand_to_tcp))
        gate_kwargs = settled_gate_kwargs()
        gate_kwargs["child_body_qd"] = np.asarray(
            [0.03, 0.0, 0.0, 0.0, 0.0, np.deg2rad(20.0)]
        )

        gate = evaluate_fixed_grasp_activation_gate(
            parent_world,
            child_world,
            anchors,
            **gate_kwargs,
            frame_index=12,
            position_limit_m=0.015,
            orientation_limit_deg=7.5,
        )

        self.assertTrue(gate.pose_passed)
        self.assertFalse(gate.twist_passed)
        self.assertFalse(gate.passed)
        self.assertGreater(gate.relative_linear_speed_m_s, 0.02)
        self.assertGreater(gate.relative_angular_speed_deg_s, 10.0)
        metadata = gate.audit_metadata()
        self.assertEqual(
            metadata["grasp_activation_parent_body"], "robot/panda_hand"
        )
        self.assertEqual(
            metadata["grasp_activation_child_body"], "object/handle"
        )
        self.assertFalse(metadata["grasp_activation_twist_gate_passed"])

    def test_fixed_grasp_twist_gate_compares_anchor_not_com_velocity(self) -> None:
        anchor = pose_matrix(
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        )
        anchors = PlannedFixedGraspAnchors(
            parent_xform=anchor,
            child_xform=anchor,
        )
        # Both bodies rotate at 1 rad/s about +Z and the common anchor moves at
        # [0, 1, 0] m/s.  Their reported COM linear velocities differ because
        # the child COM is offset by 0.5 m in +X.
        gate = evaluate_fixed_grasp_activation_gate(
            np.eye(4),
            np.eye(4),
            anchors,
            parent_body_qd=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            child_body_qd=np.asarray([0.0, 0.5, 0.0, 0.0, 0.0, 1.0]),
            parent_body_com=np.asarray([0.0, 0.0, 0.0]),
            child_body_com=np.asarray([0.5, 0.0, 0.0]),
            parent_body_name="parent",
            child_body_name="child",
            frame_index=3,
            position_limit_m=1.0e-8,
            orientation_limit_deg=1.0e-8,
            linear_velocity_limit_m_s=1.0e-8,
            angular_velocity_limit_deg_s=1.0e-8,
        )

        self.assertTrue(gate.pose_passed)
        self.assertTrue(gate.twist_passed)
        self.assertTrue(gate.passed)
        self.assertAlmostEqual(gate.relative_linear_speed_m_s, 0.0)
        self.assertAlmostEqual(gate.relative_angular_speed_deg_s, 0.0)

    def test_fixed_grasp_joint_preserves_parent_child_collision(self) -> None:
        builder = mock.Mock()
        builder.add_joint_fixed.return_value = 17
        parent_xform = object()
        child_xform = object()

        joint_index = NewtonPhysicsAssistedSimulator._add_disabled_fixed_grasp_joint(
            builder,
            parent=3,
            child=8,
            parent_xform=parent_xform,
            child_xform=child_xform,
        )

        self.assertEqual(joint_index, 17)
        builder.add_joint_fixed.assert_called_once_with(
            parent=3,
            child=8,
            parent_xform=parent_xform,
            child_xform=child_xform,
            label="agentpre_fixed_grasp",
            collision_filter_parent=False,
            enabled=False,
        )
        self.assertIs(
            builder.add_joint_fixed.call_args.kwargs["collision_filter_parent"],
            False,
        )

    def test_kinematic_twist_uses_com_and_world_angular_velocity(self) -> None:
        before = np.eye(4)[None, :, :]
        after = pose_matrix(
            [0.1, 0.0, 0.0], [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
        )[None, :, :]
        twist = kinematic_body_twists(
            before, after, np.asarray([[1.0, 0.0, 0.0]]), 0.5
        )
        np.testing.assert_allclose(twist[0, :3], [-1.8, 2.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(twist[0, 3:], [0.0, 0.0, np.pi], atol=1e-12)


class NamedLayoutTests(unittest.TestCase):
    def test_exact_and_prefixed_scalar_names_resolve_without_order_assumptions(self) -> None:
        refs = resolve_named_scalar_joints(
            ["robot/joint_a", "object/door", "robot/joint_b"],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [[0, 1], [0, 1], [0, 1]],
            ["joint_b", "joint_a"],
        )
        self.assertEqual([item.joint_index for item in refs], [2, 0])
        self.assertEqual([item.coord_index for item in refs], [2, 0])

    def test_builder_layout_without_sentinel_uses_explicit_totals(self) -> None:
        refs = resolve_named_scalar_joints(
            ["joint_a", "fixed"],
            [0, 1],
            [0, 1],
            [[0, 1], [0, 0]],
            ["joint_a"],
            joint_coord_count=1,
            joint_dof_count=1,
        )
        self.assertEqual(refs[0].coord_index, 0)

    def test_ambiguous_or_non_scalar_joint_fails_closed(self) -> None:
        with self.assertRaises(PipelineError) as caught:
            resolve_named_scalar_joints(
                ["one/door", "two/door"],
                [0, 1, 2],
                [0, 1, 2],
                [[0, 1], [0, 1]],
                ["door"],
            )
        self.assertEqual(caught.exception.code, FailureCode.NAME_NOT_UNIQUE)

        with self.assertRaises(PipelineError) as caught:
            resolve_named_scalar_joints(
                ["ball"],
                [0, 4],
                [0, 3],
                [[0, 3]],
                ["ball"],
            )
        self.assertEqual(caught.exception.code, FailureCode.CONFIG_INVALID)

    def test_only_nonroot_massless_fixed_child_is_selected_for_collapse(self) -> None:
        collapse, keep = plan_massless_fixed_joint_collapse(
            ["fixed_base", "joint8", "hand_joint", "arm_joint"],
            [3, 3, 3, 1],
            [-1, 7, 8, 0],
            [0, 8, 9, 1],
            [2.8, 2.3, 2.4, 2.6, 2.7, 3.0, 1.1, 0.4, 0.0, 0.55],
            fixed_joint_type=3,
        )
        self.assertEqual(collapse, ("joint8",))
        self.assertEqual(keep, ("fixed_base", "hand_joint"))


class RobotCoordinateMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arm_refs = (ref("arm", 0, 0),)
        self.finger_refs = (ref("left_finger", 1, 1), ref("right_finger", 3, 3))

    def test_target_builder_changes_only_named_robot_coordinates(self) -> None:
        base = np.asarray([0.0, 0.04, 0.73, 0.04, -0.2])
        target = build_robot_position_targets(
            base,
            self.arm_refs,
            self.finger_refs,
            [0.5],
            0.06,
            protected_coord_indices=(2,),
        )
        np.testing.assert_allclose(target, [0.5, 0.03, 0.73, 0.03, -0.2])
        np.testing.assert_allclose(base, [0.0, 0.04, 0.73, 0.04, -0.2])

    def test_target_builder_rejects_nonfinite_robot_commands(self) -> None:
        with self.assertRaises(ValueError):
            build_robot_position_targets(
                np.zeros(5),
                self.arm_refs,
                self.finger_refs,
                [np.nan],
                0.06,
            )
        with self.assertRaises(ValueError):
            build_robot_position_targets(
                np.zeros(5),
                self.arm_refs,
                self.finger_refs,
                [0.0],
                np.inf,
            )

    def test_overlapping_door_mapping_is_rejected(self) -> None:
        with self.assertRaises(PipelineError) as caught:
            build_robot_position_targets(
                np.zeros(2),
                (ref("arm", 0, 0),),
                (),
                [0.1],
                0.0,
                protected_coord_indices=(0,),
            )
        self.assertEqual(caught.exception.code, FailureCode.CONFIG_INVALID)


class RobotVelocityLimitTests(unittest.TestCase):
    @staticmethod
    def _write_robot_urdf(
        directory: Path, *, missing_velocity_joint: str | None = None
    ) -> Path:
        joints = (
            ("arm_a", "revolute", "base", "arm_a_link", "2.0"),
            ("arm_b", "revolute", "arm_a_link", "arm_b_link", "1.0"),
            ("finger_left", "prismatic", "arm_b_link", "left_link", "0.2"),
            ("finger_right", "prismatic", "arm_b_link", "right_link", "0.2"),
        )
        joint_xml: list[str] = []
        for name, joint_type, parent, child, velocity in joints:
            velocity_attribute = (
                "" if name == missing_velocity_joint else f' velocity="{velocity}"'
            )
            joint_xml.append(
                f"""
  <joint name="{name}" type="{joint_type}">
    <parent link="{parent}"/><child link="{child}"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="10"{velocity_attribute}/>
  </joint>"""
            )
        path = directory / "robot.urdf"
        path.write_text(
            """<robot name="velocity_test">
  <link name="base"/><link name="arm_a_link"/><link name="arm_b_link"/>
  <link name="left_link"/><link name="right_link"/>
"""
            + "".join(joint_xml)
            + "\n</robot>\n",
            encoding="utf-8",
        )
        return path

    def test_urdf_limits_are_name_aligned_and_missing_limit_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            robot = self._write_robot_urdf(root)
            limits = load_robot_joint_velocity_limits(
                robot,
                ("arm_b", "arm_a"),
                ("finger_right", "finger_left"),
            )
            self.assertEqual(limits.arm_velocity_limits, (1.0, 2.0))
            self.assertEqual(limits.finger_velocity_limits, (0.2, 0.2))
            self.assertEqual(
                limits.controlled_joint_names,
                ("arm_b", "arm_a", "finger_right", "finger_left"),
            )

            missing = self._write_robot_urdf(
                root, missing_velocity_joint="arm_b"
            )
            with self.assertRaises(PipelineError) as caught:
                load_robot_joint_velocity_limits(
                    missing,
                    ("arm_a", "arm_b"),
                    ("finger_left", "finger_right"),
                )
            self.assertEqual(caught.exception.code, FailureCode.ASSET_INVALID)
            self.assertEqual(caught.exception.stage, "physics_model")
            self.assertEqual(caught.exception.details["joint"], "arm_b")

    def test_whole_trajectory_preflight_reports_the_bad_frame_and_joint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            robot = self._write_robot_urdf(Path(temporary))
            limits = load_robot_joint_velocity_limits(
                robot,
                ("arm_a", "arm_b"),
                ("finger_left", "finger_right"),
            )
            valid = PhysicsCommandTrajectory(
                np.asarray(["pregrasp", "approach", "actuate"]),
                np.asarray([[0.0, 0.0], [0.1, 0.05], [0.2, 0.1]]),
                np.asarray([0.08, 0.076, 0.072]),
            )
            audit = validate_robot_command_motion_limits(
                valid,
                limits,
                0.1,
                initial_arm_joint_positions=(0.0, 0.0),
                initial_gripper_width_m=0.08,
                max_joint_acceleration_rad_s2=20.0,
                max_joint_jerk_rad_s3=200.0,
                max_finger_acceleration_m_s2=20.0,
                max_finger_jerk_m_s3=200.0,
            )
            self.assertTrue(audit["passed"])
            self.assertTrue(audit["initial_state_included"])
            self.assertEqual(audit["transition_count"], 3)
            self.assertEqual(audit["acceleration_sample_count"], 5)
            self.assertEqual(audit["jerk_sample_count"], 5)
            self.assertEqual(audit["terminal_hold_sample_count"], 2)
            self.assertEqual(audit["initial_joint_velocity_rad_s"], 0.0)
            self.assertEqual(audit["initial_joint_acceleration_rad_s2"], 0.0)
            self.assertAlmostEqual(audit["max_requested_to_limit_ratio"], 0.5)
            self.assertAlmostEqual(audit["max_arm_joint_velocity_rad_s"], 1.0)
            self.assertAlmostEqual(
                audit["max_arm_joint_acceleration_rad_s2"], 10.0
            )
            self.assertAlmostEqual(audit["max_arm_joint_jerk_rad_s3"], 100.0)

            commands = PhysicsCommandTrajectory(
                np.asarray(["pregrasp", "approach", "actuate"]),
                np.asarray([[0.0, 0.0], [0.1, 0.05], [0.11, 0.25]]),
                np.asarray([0.08, 0.076, 0.072]),
            )
            with self.assertRaises(PipelineError) as caught:
                validate_robot_command_motion_limits(
                    commands,
                    limits,
                    0.1,
                    initial_arm_joint_positions=(0.0, 0.0),
                    initial_gripper_width_m=0.08,
                    max_joint_acceleration_rad_s2=100.0,
                    max_joint_jerk_rad_s3=1000.0,
                    max_finger_acceleration_m_s2=100.0,
                    max_finger_jerk_m_s3=1000.0,
                )

            error = caught.exception
            self.assertEqual(error.code, FailureCode.JOINT_LIMIT)
            self.assertEqual(error.stage, "physics_trajectory")
            self.assertEqual(error.details["violation_count"], 1)
            violation = error.details["violations"][0]
            self.assertEqual(violation["previous_frame_index"], 1)
            self.assertEqual(violation["frame_index"], 2)
            self.assertEqual(violation["joint"], "arm_b")
            self.assertAlmostEqual(violation["requested_speed"], 2.0)
            self.assertAlmostEqual(violation["velocity_limit"], 1.0)
            self.assertAlmostEqual(violation["required_transition_dt_s"], 0.2)

    def test_preflight_includes_configured_initial_to_frame_zero_transition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            robot = self._write_robot_urdf(Path(temporary))
            limits = load_robot_joint_velocity_limits(
                robot,
                ("arm_a", "arm_b"),
                ("finger_left", "finger_right"),
            )
            commands = PhysicsCommandTrajectory(
                np.asarray(["pregrasp"]),
                np.asarray([[0.3, 0.0]]),
                np.asarray([0.08]),
            )
            with self.assertRaises(PipelineError) as caught:
                validate_robot_command_motion_limits(
                    commands,
                    limits,
                    0.1,
                    initial_arm_joint_positions=(0.0, 0.0),
                    initial_gripper_width_m=0.08,
                    max_joint_acceleration_rad_s2=100.0,
                    max_joint_jerk_rad_s3=1000.0,
                    max_finger_acceleration_m_s2=100.0,
                    max_finger_jerk_m_s3=1000.0,
                )
            self.assertEqual(caught.exception.details["limit_kind"], "velocity")
            violation = caught.exception.details["violations"][0]
            self.assertEqual(violation["previous_frame_index"], -1)
            self.assertEqual(violation["frame_index"], 0)
            self.assertEqual(
                violation["previous_phase"], "configured_initial_state"
            )
            self.assertEqual(violation["joint"], "arm_a")
            self.assertAlmostEqual(violation["requested_speed"], 3.0)

    def test_acceleration_and_jerk_preflights_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            robot = self._write_robot_urdf(Path(temporary))
            limits = load_robot_joint_velocity_limits(
                robot,
                ("arm_a", "arm_b"),
                ("finger_left", "finger_right"),
            )
            initial_derivative = PhysicsCommandTrajectory(
                np.asarray(["pregrasp"]),
                np.asarray([[0.01, 0.0]]),
                np.asarray([0.08]),
            )
            with self.assertRaises(PipelineError) as caught:
                validate_robot_command_motion_limits(
                    initial_derivative,
                    limits,
                    0.1,
                    initial_arm_joint_positions=(0.0, 0.0),
                    initial_gripper_width_m=0.08,
                    max_joint_acceleration_rad_s2=0.5,
                    max_joint_jerk_rad_s3=1000.0,
                    max_finger_acceleration_m_s2=100.0,
                    max_finger_jerk_m_s3=1000.0,
                )
            self.assertEqual(caught.exception.code, FailureCode.JOINT_LIMIT)
            self.assertEqual(
                caught.exception.details["limit_kind"], "acceleration"
            )
            acceleration = caught.exception.details["violations"][0]
            self.assertEqual(acceleration["previous_frame_index"], -1)
            self.assertEqual(acceleration["frame_index"], 0)
            self.assertEqual(acceleration["joint"], "arm_a")
            self.assertAlmostEqual(acceleration["acceleration_magnitude"], 1.0)

            with self.assertRaises(PipelineError) as caught:
                validate_robot_command_motion_limits(
                    initial_derivative,
                    limits,
                    0.1,
                    initial_arm_joint_positions=(0.0, 0.0),
                    initial_gripper_width_m=0.08,
                    max_joint_acceleration_rad_s2=2.1,
                    max_joint_jerk_rad_s3=5.0,
                    max_finger_acceleration_m_s2=100.0,
                    max_finger_jerk_m_s3=1000.0,
                )
            self.assertEqual(caught.exception.code, FailureCode.JOINT_LIMIT)
            self.assertEqual(caught.exception.details["limit_kind"], "jerk")
            jerk = caught.exception.details["violations"][0]
            self.assertEqual(jerk["previous_frame_index"], 0)
            self.assertEqual(jerk["frame_index"], 1)
            self.assertEqual(jerk["frame_kind"], "virtual_terminal_hold")
            self.assertEqual(jerk["joint"], "arm_a")
            self.assertAlmostEqual(jerk["jerk_magnitude"], 20.0, places=5)

    def test_finger_acceleration_and_jerk_use_prismatic_si_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            robot = self._write_robot_urdf(Path(temporary))
            limits = load_robot_joint_velocity_limits(
                robot,
                ("arm_a", "arm_b"),
                ("finger_left", "finger_right"),
            )
            commands = PhysicsCommandTrajectory(
                np.asarray(["close"]),
                np.asarray([[0.0, 0.0]]),
                np.asarray([0.10]),
            )
            with self.assertRaises(PipelineError) as caught:
                validate_robot_command_motion_limits(
                    commands,
                    limits,
                    0.1,
                    initial_arm_joint_positions=(0.0, 0.0),
                    initial_gripper_width_m=0.08,
                    max_joint_acceleration_rad_s2=100.0,
                    max_joint_jerk_rad_s3=1000.0,
                    max_finger_acceleration_m_s2=0.5,
                    max_finger_jerk_m_s3=1000.0,
                )
            self.assertEqual(caught.exception.details["limit_kind"], "acceleration")
            self.assertEqual(caught.exception.details["scope"], "finger_joints")
            self.assertEqual(
                caught.exception.details["source"],
                "thresholds.max_finger_acceleration_m_s2",
            )
            acceleration = caught.exception.details["violations"][0]
            self.assertEqual(acceleration["joint_kind"], "finger")
            self.assertEqual(acceleration["joint_type"], "prismatic")
            self.assertEqual(acceleration["acceleration_unit"], "m/s^2")
            self.assertAlmostEqual(acceleration["acceleration_magnitude"], 1.0)

            with self.assertRaises(PipelineError) as caught:
                validate_robot_command_motion_limits(
                    commands,
                    limits,
                    0.1,
                    initial_arm_joint_positions=(0.0, 0.0),
                    initial_gripper_width_m=0.08,
                    max_joint_acceleration_rad_s2=100.0,
                    max_joint_jerk_rad_s3=1000.0,
                    max_finger_acceleration_m_s2=2.0,
                    max_finger_jerk_m_s3=5.0,
                )
            self.assertEqual(caught.exception.details["limit_kind"], "jerk")
            self.assertEqual(
                caught.exception.details["source"],
                "thresholds.max_finger_jerk_m_s3",
            )
            jerk = caught.exception.details["violations"][0]
            self.assertEqual(jerk["jerk_unit"], "m/s^3")
            self.assertAlmostEqual(jerk["jerk_magnitude"], 20.0, places=5)

    def test_frame_zero_velocity_writer_is_real_and_never_exceeds_limit(self) -> None:
        limits = RobotJointVelocityLimits(
            source_urdf=Path("robot.urdf"),
            arm_joint_names=("arm",),
            arm_joint_types=("revolute",),
            arm_velocity_limits=(0.1,),
            finger_joint_names=(),
            finger_joint_types=(),
            finger_velocity_limits=(),
        )
        self.assertGreater(float(np.float32(0.1)), 0.1)
        target = build_robot_velocity_targets(
            [0.0], [0.02], 0.0, 0.0, limits, 0.2, frame_index=0
        )
        self.assertEqual(target.dtype, np.float32)
        self.assertGreater(float(target[0]), 0.0)
        self.assertLessEqual(abs(float(target[0])), 0.1)

        with self.assertRaises(PipelineError) as caught:
            build_robot_velocity_targets(
                [0.0], [0.021], 0.0, 0.0, limits, 0.2, frame_index=0
            )
        self.assertEqual(caught.exception.code, FailureCode.JOINT_LIMIT)
        self.assertEqual(caught.exception.stage, "physics_control")

    def test_preflight_rechecks_float32_target_velocity_derivatives(self) -> None:
        dt = 0.0166666667
        acceleration_limit = 7.5
        velocity_increment = acceleration_limit * dt - 2.0e-8
        requested_velocity = np.concatenate(
            (
                np.arange(1, 18, dtype=float),
                np.arange(16, -1, -1, dtype=float),
            )
        ) * velocity_increment
        positions = np.cumsum(requested_velocity * dt)
        limits = RobotJointVelocityLimits(
            source_urdf=Path("robot.urdf"),
            arm_joint_names=("arm",),
            arm_joint_types=("revolute",),
            arm_velocity_limits=(2.61,),
            finger_joint_names=(),
            finger_joint_types=(),
            finger_velocity_limits=(),
        )
        commands = PhysicsCommandTrajectory(
            np.asarray(["pregrasp"] * len(positions)),
            positions[:, None],
            np.zeros(len(positions)),
        )
        with self.assertRaises(PipelineError) as caught:
            validate_robot_command_motion_limits(
                commands,
                limits,
                dt,
                initial_arm_joint_positions=(0.0,),
                initial_gripper_width_m=0.0,
                max_joint_acceleration_rad_s2=acceleration_limit,
                max_joint_jerk_rad_s3=10000.0,
                max_finger_acceleration_m_s2=1.0,
                max_finger_jerk_m_s3=1.0,
            )
        self.assertEqual(caught.exception.details["limit_kind"], "acceleration")
        violation = caught.exception.details["violations"][0]
        self.assertGreater(
            violation["acceleration_magnitude"], acceleration_limit
        )

    def test_simulator_rejects_bad_velocity_before_newton_runtime_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            robot = self._write_robot_urdf(Path(temporary))
            base = PhysicsParameters.from_project_config(
                load_config(ROOT / "configs/microwave_franka.json")
            )
            params = replace(
                base,
                robot_urdf=robot,
                arm_joint_names=("arm_a", "arm_b"),
                finger_joint_names=("finger_left", "finger_right"),
                nominal_arm_joint_positions=(0.0, 0.0),
            )
            commands = PhysicsCommandTrajectory(
                np.asarray(["close", "actuate"]),
                np.asarray([[0.0, 0.0], [1.0, 0.0]]),
                np.asarray([0.04, 0.04]),
            )
            simulator = NewtonPhysicsAssistedSimulator(params)
            with mock.patch.object(simulator, "_build_runtime") as runtime_build:
                with self.assertRaises(PipelineError) as caught:
                    simulator.run(commands)
            runtime_build.assert_not_called()
            self.assertEqual(caught.exception.code, FailureCode.JOINT_LIMIT)
            self.assertEqual(caught.exception.stage, "physics_trajectory")


class PhysicsContactEvidenceTests(unittest.TestCase):
    @staticmethod
    def _array(values: object) -> SimpleNamespace:
        data = np.asarray(values)
        return SimpleNamespace(numpy=lambda: data)

    def test_speculative_contact_is_ignored_but_penetration_is_reported(self) -> None:
        identity_pose_xyzw = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        contacts = SimpleNamespace(
            rigid_contact_count=self._array([2]),
            rigid_contact_shape0=self._array([0, 0]),
            rigid_contact_shape1=self._array([1, 1]),
            rigid_contact_point0=self._array([[0.0, 0.0, 0.0]] * 2),
            # With 0.02 m margins on both sides, 0.10 m is an inactive
            # candidate while 0.03 m is a 0.01 m penetration.
            rigid_contact_point1=self._array(
                [[0.10, 0.0, 0.0], [0.03, 0.0, 0.0]]
            ),
            rigid_contact_normal=self._array([[1.0, 0.0, 0.0]] * 2),
            rigid_contact_margin0=self._array([0.02, 0.02]),
            rigid_contact_margin1=self._array([0.02, 0.02]),
        )
        runtime = SimpleNamespace(
            contacts=contacts,
            model=SimpleNamespace(
                shape_body=self._array([0, 1]),
                body_label=("robot/link", "object/link"),
            ),
            state_in=SimpleNamespace(
                body_q=self._array([identity_pose_xyzw, identity_pose_xyzw])
            ),
            allowed_contact_body_indices=frozenset(),
            robot_body_indices=frozenset({0}),
            object_body_indices=frozenset({1}),
            collision_margin_m=0.003,
        )

        pairs = NewtonPhysicsAssistedSimulator._forbidden_contact_pairs(runtime)
        self.assertEqual(pairs, (("object/link", "robot/link"),))


class PhysicsConfigurationTests(unittest.TestCase):
    def test_broad_phase_is_case_normalized_and_strictly_validated(self) -> None:
        import src.physics_assisted as module

        config_path = ROOT / "configs/microwave_franka.json"
        config = ProjectConfig(
            path=config_path,
            project_root=ROOT,
            data=json.loads(config_path.read_text()),
        )
        params = PhysicsParameters.from_project_config(config)
        self.assertEqual(
            replace(params, collision_broad_phase="  SaP  ").collision_broad_phase,
            "sap",
        )
        self.assertEqual(module._normalize_newton_broad_phase("  SaP  "), "sap")
        self.assertEqual(module._normalize_newton_broad_phase("NXN"), "nxn")
        self.assertEqual(
            module._normalize_newton_broad_phase("Explicit"), "explicit"
        )
        with self.assertRaisesRegex(ValueError, "expected sap, nxn, or explicit"):
            module._normalize_newton_broad_phase("gpu_grid")
        with self.assertRaisesRegex(ValueError, "must be a string"):
            module._normalize_newton_broad_phase(None)

    def test_physics_scene_starts_at_configured_nominal_robot_state(self) -> None:
        base = PhysicsParameters.from_project_config(
            load_config("configs/microwave_franka.json")
        )
        params = replace(
            base,
            arm_joint_names=("arm",),
            finger_joint_names=("left", "right"),
            nominal_arm_joint_positions=(0.7,),
            open_gripper_width_m=0.06,
        )
        simulator = NewtonPhysicsAssistedSimulator(params)
        builder = SimpleNamespace(joint_q=[-9.0, -9.0, -9.0, 0.0])
        simulator._set_builder_initial_coordinates(
            builder,
            (ref("arm", 0, 0),),
            (ref("left", 1, 1), ref("right", 2, 2)),
            ref("door", 3, 3),
        )
        np.testing.assert_allclose(
            builder.joint_q,
            [0.7, 0.03, 0.03, 0.0],
        )

    def test_checked_in_config_maps_all_physics_controls_explicitly(self) -> None:
        config = load_config("configs/microwave_franka.json")
        params = PhysicsParameters.from_project_config(config)
        self.assertEqual(params.device, "cpu")
        self.assertEqual(params.solver, "xpbd")
        self.assertEqual(params.arm_joint_names[0], "panda_joint1")
        self.assertEqual(params.finger_joint_names, ("panda_finger_joint1", "panda_finger_joint2"))
        self.assertEqual(params.door_joint, "door_hinge")
        self.assertTrue(params.fixed_grasp_enabled)
        self.assertEqual(params.fixed_grasp_activate_after_phase, "close")
        self.assertEqual(params.robot_control_backend, "joint_pd")
        self.assertEqual(
            params.robot_control_implementation, "newton_xpbd_joint_targets"
        )
        self.assertEqual(params.target_velocity_mode, "finite_difference")
        self.assertEqual(params.arm_joint_tracking_reserve_rad, 0.05)
        self.assertEqual(params.control_limit_margin_rad, 0.02)
        self.assertEqual(params.grasp_release_blend_frames, 32)
        with self.assertRaisesRegex(
            ValueError, "arm_joint_tracking_reserve_rad"
        ):
            replace(params, arm_joint_tracking_reserve_rad=0.0)
        with self.assertRaisesRegex(ValueError, "control_limit_margin_rad"):
            replace(params, control_limit_margin_rad=0.0)
        with self.assertRaisesRegex(ValueError, "grasp_release_blend_frames"):
            replace(params, grasp_release_blend_frames=1)
        self.assertEqual(
            params.max_joint_acceleration_rad_s2,
            float(config.get("thresholds.max_joint_acceleration_rad_s2")),
        )
        self.assertEqual(
            params.max_joint_jerk_rad_s3,
            float(config.get("thresholds.max_joint_jerk_rad_s3")),
        )
        self.assertEqual(
            params.max_finger_acceleration_m_s2,
            float(config.get("thresholds.max_finger_acceleration_m_s2")),
        )
        self.assertEqual(
            params.max_finger_jerk_m_s3,
            float(config.get("thresholds.max_finger_jerk_m_s3")),
        )
        with self.assertRaisesRegex(ValueError, "max_joint_acceleration_rad_s2"):
            replace(params, max_joint_acceleration_rad_s2=0.0)
        with self.assertRaisesRegex(ValueError, "max_joint_jerk_rad_s3"):
            replace(params, max_joint_jerk_rad_s3=0.0)
        with self.assertRaisesRegex(ValueError, "max_finger_acceleration_m_s2"):
            replace(params, max_finger_acceleration_m_s2=0.0)
        with self.assertRaisesRegex(ValueError, "max_finger_jerk_m_s3"):
            replace(params, max_finger_jerk_m_s3=0.0)
        self.assertEqual(
            params.grasp_activation_linear_velocity_tolerance_m_s,
            float(
                config.get(
                    "simulation.fixed_grasp_constraint.activation_linear_velocity_tolerance_m_s"
                )
            ),
        )
        self.assertEqual(
            params.grasp_activation_angular_velocity_tolerance_deg_s,
            float(
                config.get(
                    "simulation.fixed_grasp_constraint.activation_angular_velocity_tolerance_deg_s"
                )
            ),
        )
        with self.assertRaisesRegex(
            ValueError, "grasp_activation_linear_velocity_tolerance_m_s"
        ):
            replace(params, grasp_activation_linear_velocity_tolerance_m_s=0.0)
        with self.assertRaisesRegex(
            ValueError, "grasp_activation_angular_velocity_tolerance_deg_s"
        ):
            replace(params, grasp_activation_angular_velocity_tolerance_deg_s=0.0)
        self.assertGreater(params.arm_stiffness, 0.0)
        self.assertGreater(params.arm_damping, 0.0)
        self.assertAlmostEqual(params.grasp_activation_position_tolerance_m, 0.015)

    def test_reference_reserve_audit_uses_strict_hard_limit_clearance(self) -> None:
        import src.physics_assisted as module

        equal = module.audit_arm_joint_reference_reserve(
            np.asarray([[0.75]]),
            ("arm",),
            np.asarray([-1.0]),
            np.asarray([1.0]),
            initial_arm_q=np.asarray([0.0]),
            control_limit_margin_rad=0.1,
            arm_joint_tracking_reserve_rad=0.25,
            phase_names=("actuate",),
        )
        self.assertTrue(equal["passed"])
        self.assertEqual(equal["min_hard_limit_clearance_rad"], 0.25)
        self.assertEqual(
            equal["min_hard_limit_clearance_sample_scope"],
            "trajectory_frame",
        )
        self.assertEqual(equal["min_hard_limit_clearance_sample_index"], 1)
        self.assertEqual(equal["min_hard_limit_clearance_frame_index"], 0)
        self.assertEqual(equal["min_hard_limit_clearance_joint_name"], "arm")
        self.assertEqual(equal["min_hard_limit_clearance_side"], "upper")
        self.assertEqual(equal["hard_limit_reserve_violation_count"], 0)
        self.assertEqual(equal["control_bound_touch_count"], 0)

        below_reserve = np.nextafter(
            np.float32(0.75), np.float32(math.inf)
        )
        failed = module.audit_arm_joint_reference_reserve(
            np.asarray([[below_reserve]], dtype=np.float32),
            ("arm",),
            np.asarray([-1.0]),
            np.asarray([1.0]),
            initial_arm_q=np.asarray([0.0]),
            control_limit_margin_rad=0.1,
            arm_joint_tracking_reserve_rad=0.25,
            phase_names=("actuate",),
        )
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["failed_checks"], ["hard_limit_clearance"])
        self.assertLess(failed["min_hard_limit_clearance_rad"], 0.25)
        self.assertEqual(failed["hard_limit_reserve_violation_count"], 1)
        self.assertEqual(failed["control_bound_touch_count"], 0)

    def test_reference_reserve_audit_detects_exact_float32_control_endpoint(
        self,
    ) -> None:
        import src.physics_assisted as module

        hard_lower = np.asarray([-3.0718])
        hard_upper = np.asarray([-0.0698])
        model_lower = hard_lower.astype(np.float32).astype(float)
        continuous_lower = model_lower + 0.02
        endpoint, _ = module._inward_float32_position_bounds(
            continuous_lower,
            hard_upper.astype(np.float32).astype(float) - 0.02,
        )
        self.assertEqual(endpoint[0], -3.051799774169922)

        touched = module.audit_arm_joint_reference_reserve(
            endpoint[None, :],
            ("panda_joint4",),
            hard_lower,
            hard_upper,
            initial_arm_q=np.asarray([-1.5]),
            control_limit_margin_rad=0.02,
            arm_joint_tracking_reserve_rad=0.01,
            phase_names=("actuate",),
        )
        self.assertFalse(touched["passed"])
        self.assertEqual(touched["failed_checks"], ["control_bound_touch"])
        self.assertEqual(touched["control_bound_touch_count"], 1)
        self.assertEqual(touched["control_bound_touch_sample_count"], 1)
        self.assertEqual(touched["control_bound_touch_frame_count"], 1)
        self.assertEqual(
            touched["first_control_bound_touch_sample_scope"],
            "trajectory_frame",
        )
        self.assertEqual(touched["first_control_bound_touch_sample_index"], 1)
        self.assertEqual(touched["first_control_bound_touch_frame_index"], 0)
        self.assertEqual(touched["first_control_bound_touch_phase"], "actuate")
        self.assertEqual(
            touched["first_control_bound_touch_joint_name"], "panda_joint4"
        )
        self.assertEqual(touched["first_control_bound_touch_side"], "lower")
        self.assertEqual(
            touched["first_control_bound_touch_bound_rad"], endpoint[0]
        )

        one_ulp_inside = np.nextafter(
            np.float32(endpoint[0]), np.float32(math.inf)
        )
        clear = module.audit_arm_joint_reference_reserve(
            np.asarray([[one_ulp_inside]], dtype=np.float32),
            ("panda_joint4",),
            hard_lower,
            hard_upper,
            initial_arm_q=np.asarray([-1.5]),
            control_limit_margin_rad=0.02,
            arm_joint_tracking_reserve_rad=0.01,
        )
        self.assertTrue(clear["passed"])
        self.assertEqual(clear["control_bound_touch_count"], 0)

    def test_reference_reserve_audits_initial_state_before_frame_zero(self) -> None:
        import src.physics_assisted as module

        initial_hard_failure = module.audit_arm_joint_reference_reserve(
            np.asarray([[0.0]]),
            ("arm",),
            np.asarray([-1.0]),
            np.asarray([1.0]),
            initial_arm_q=np.asarray([0.8]),
            control_limit_margin_rad=0.1,
            arm_joint_tracking_reserve_rad=0.25,
            phase_names=("pregrasp",),
        )
        self.assertFalse(initial_hard_failure["passed"])
        self.assertEqual(
            initial_hard_failure["failed_checks"],
            ["hard_limit_clearance"],
        )
        self.assertEqual(
            initial_hard_failure["min_hard_limit_clearance_sample_scope"],
            "initial_state",
        )
        self.assertEqual(
            initial_hard_failure["min_hard_limit_clearance_sample_index"], 0
        )
        self.assertIsNone(
            initial_hard_failure["min_hard_limit_clearance_frame_index"]
        )
        self.assertIsNone(
            initial_hard_failure["min_hard_limit_clearance_phase"]
        )
        self.assertEqual(
            initial_hard_failure[
                "initial_state_hard_limit_reserve_violation_count"
            ],
            1,
        )
        self.assertEqual(
            initial_hard_failure[
                "trajectory_hard_limit_reserve_violation_count"
            ],
            0,
        )

        _, upper_endpoint = module._inward_float32_position_bounds(
            np.asarray([-0.9]), np.asarray([0.9])
        )
        initial_control_failure = module.audit_arm_joint_reference_reserve(
            np.asarray([[0.0]]),
            ("arm",),
            np.asarray([-1.0]),
            np.asarray([1.0]),
            initial_arm_q=upper_endpoint,
            control_limit_margin_rad=0.1,
            arm_joint_tracking_reserve_rad=0.01,
            phase_names=("pregrasp",),
        )
        self.assertFalse(initial_control_failure["passed"])
        self.assertEqual(
            initial_control_failure["failed_checks"],
            ["control_bound_touch"],
        )
        self.assertEqual(
            initial_control_failure["first_control_bound_touch_sample_scope"],
            "initial_state",
        )
        self.assertEqual(
            initial_control_failure["first_control_bound_touch_sample_index"],
            0,
        )
        self.assertIsNone(
            initial_control_failure["first_control_bound_touch_frame_index"]
        )
        self.assertIsNone(
            initial_control_failure["first_control_bound_touch_phase"]
        )
        self.assertEqual(
            initial_control_failure["initial_state_control_bound_touch_count"],
            1,
        )
        self.assertEqual(
            initial_control_failure["trajectory_control_bound_touch_count"],
            0,
        )
        self.assertEqual(
            initial_control_failure["control_bound_touch_sample_count"], 1
        )
        self.assertEqual(
            initial_control_failure["control_bound_touch_frame_count"], 0
        )

    def test_checked_in_nominal_arm_state_passes_reference_reserve(self) -> None:
        import src.physics_assisted as module

        config = load_config(ROOT / "configs/microwave_franka.json")
        nominal = np.asarray(
            config.get("assets.robot.default_joint_positions"), dtype=float
        )
        hard_lower = np.asarray(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        )
        hard_upper = np.asarray(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        )
        audit = module.audit_arm_joint_reference_reserve(
            nominal[None, :],
            tuple(config.get("assets.robot.arm_joint_names")),
            hard_lower,
            hard_upper,
            initial_arm_q=nominal,
            control_limit_margin_rad=float(
                config.get("ik.control_limit_margin_rad")
            ),
            arm_joint_tracking_reserve_rad=float(
                config.get(
                    "simulation.robot_control.arm_joint_tracking_reserve_rad"
                )
            ),
            phase_names=("pregrasp",),
        )
        self.assertTrue(audit["passed"])
        self.assertTrue(audit["initial_state_included"])
        self.assertEqual(audit["audited_sample_count"], 2)
        self.assertEqual(audit["reference_frame_count"], 1)
        self.assertEqual(audit["control_bound_touch_count"], 0)
        self.assertEqual(audit["hard_limit_reserve_violation_count"], 0)

    def test_reference_reserve_audit_rejects_malformed_reference(self) -> None:
        import src.physics_assisted as module

        common = {
            "arm_joint_names": ("a", "b"),
            "hard_lower_rad": np.asarray([-1.0, -1.0]),
            "hard_upper_rad": np.asarray([1.0, 1.0]),
            "initial_arm_q": np.asarray([0.0, 0.0]),
            "control_limit_margin_rad": 0.1,
            "arm_joint_tracking_reserve_rad": 0.2,
        }
        with self.assertRaises(PipelineError) as shape_error:
            module.audit_arm_joint_reference_reserve(
                np.zeros((2, 1)),
                **common,
            )
        self.assertEqual(shape_error.exception.stage, "ik_motion_limits")
        self.assertEqual(shape_error.exception.code, FailureCode.CONFIG_INVALID)

        with self.assertRaises(PipelineError) as finite_error:
            module.audit_arm_joint_reference_reserve(
                np.asarray([[0.0, np.nan]]),
                **common,
            )
        self.assertEqual(finite_error.exception.stage, "ik_motion_limits")
        self.assertEqual(
            finite_error.exception.code,
            FailureCode.NUMERICAL_INSTABILITY,
        )

        nonfinite_initial = dict(common)
        nonfinite_initial["initial_arm_q"] = np.asarray([0.0, np.inf])
        with self.assertRaises(PipelineError) as initial_error:
            module.audit_arm_joint_reference_reserve(
                np.zeros((1, 2)),
                **nonfinite_initial,
            )
        self.assertEqual(initial_error.exception.stage, "ik_motion_limits")
        self.assertEqual(
            initial_error.exception.code,
            FailureCode.NUMERICAL_INSTABILITY,
        )

    def test_builder_pd_configuration_writes_only_named_robot_slots(self) -> None:
        builder = SimpleNamespace(
            joint_q=[0.1, 0.2, 0.3, 0.4],
            joint_target_q=[-1.0, -1.0, 9.0, -1.0],
            joint_target_qd=[-2.0, -2.0, 8.0, -2.0],
            joint_target_ke=[-3.0, -3.0, 7.0, -3.0],
            joint_target_kd=[-4.0, -4.0, 6.0, -4.0],
        )
        NewtonPhysicsAssistedSimulator._set_builder_robot_pd(
            builder,
            (ref("arm", 0, 0),),
            (ref("left", 1, 1), ref("right", 3, 3)),
            arm_stiffness=650.0,
            arm_damping=100.0,
            finger_stiffness=300.0,
            finger_damping=40.0,
        )
        self.assertEqual(builder.joint_target_q, [0.1, 0.2, 9.0, 0.4])
        self.assertEqual(builder.joint_target_qd, [0.0, 0.0, 8.0, 0.0])
        self.assertEqual(builder.joint_target_ke, [650.0, 300.0, 7.0, 300.0])
        self.assertEqual(builder.joint_target_kd, [100.0, 40.0, 6.0, 40.0])

    def test_missing_or_wrong_newton_version_is_structured_unavailable(self) -> None:
        import src.physics_assisted as module

        missing = ImportError("not installed")
        with (
            mock.patch.object(module, "_NEWTON_IMPORT_ERROR", missing),
            mock.patch.object(module, "newton", None),
            mock.patch.object(module, "wp", None),
        ):
            with self.assertRaises(PipelineError) as caught:
                require_newton_v13()
        self.assertEqual(caught.exception.code, FailureCode.PHYSICS_UNAVAILABLE)
        self.assertEqual(caught.exception.stage, "physics_import")

        fake = type("FakeNewton", (), {"__version__": "1.5.0"})()
        with (
            mock.patch.object(module, "_NEWTON_IMPORT_ERROR", None),
            mock.patch.object(module, "newton", fake),
            mock.patch.object(module, "wp", object()),
        ):
            with self.assertRaises(PipelineError) as caught:
                require_newton_v13()
        self.assertEqual(caught.exception.details["detected_version"], "1.5.0")


class _FakePhysicsSimulator:
    def __init__(self, parameters: PhysicsParameters) -> None:
        self.parameters = parameters

    def run(self, commands: PhysicsCommandTrajectory) -> PhysicsRollout:
        frame_count = commands.frame_count
        door = np.zeros(frame_count, dtype=float)
        full_q = np.column_stack((commands.arm_joint_targets[:, 0], door))
        pose = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        handle = np.asarray([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
        collision = np.zeros(frame_count, dtype=bool)
        collision[3] = True
        finger_scale = (np.arange(frame_count, dtype=float) + 1.0) * 1.0e-4
        finger_multipliers = np.asarray(
            [1.0 if index % 2 == 0 else -2.0 for index in range(len(self.parameters.finger_joint_names))]
        )
        measured_finger_qd = finger_scale[:, None] * finger_multipliers[None, :]
        pairs = tuple(
            (("robot/panda_link5", "object/microwave_door"),)
            if index == 3
            else ()
            for index in range(frame_count)
        )
        return PhysicsRollout(
            phase_names=commands.phase_names.copy(),
            time_s=(
                (np.arange(frame_count, dtype=float) + 1.0)
                * self.parameters.dt
            ),
            command_joint_q=full_q.copy(),
            applied_arm_joint_target_q=(
                commands.arm_joint_targets.astype(np.float32)
            ),
            applied_arm_joint_target_qd=np.zeros(
                commands.arm_joint_targets.shape, dtype=np.float32
            ),
            measured_joint_q=full_q.copy(),
            measured_joint_qd=np.zeros_like(full_q),
            measured_arm_joint_q=commands.arm_joint_targets.copy(),
            measured_arm_joint_qd=np.zeros_like(commands.arm_joint_targets),
            measured_finger_joint_qd=measured_finger_qd,
            door_angle_rad=door,
            ee_pose_wxyz=np.repeat(pose[None, :], frame_count, axis=0),
            handle_link_pose_wxyz=np.repeat(
                handle[None, :], frame_count, axis=0
            ),
            body_pose_wxyz=np.repeat(
                np.stack((pose, handle))[None, :, :], frame_count, axis=0
            ),
            collision_flags=collision,
            grasp_constraint_active=np.asarray(
                [
                    str(phase) in {"actuate", "release"}
                    for phase in commands.phase_names
                ],
                dtype=bool,
            ),
            external_robot_joint_force_command=np.zeros_like(full_q),
            forbidden_contact_pairs=pairs,
            forbidden_contact_signed_clearance_m=tuple(
                (-0.001,) if index == 3 else ()
                for index in range(frame_count)
            ),
            body_labels=("robot/panda_hand", "object/handle"),
            joint_labels=("robot/door_hinge", "object/door_hinge"),
            finger_joint_names=self.parameters.finger_joint_names,
            metadata={
                "status": "completed",
                "backend": "fake_newton_1_3",
                "state_sample_timing": "post_step_end_of_frame",
                "simulation_dt_s": self.parameters.dt,
                "constraint_backend": "fake_fixed_grasp",
                "control_backend": "joint_pd",
                "robot_control_implementation": "newton_xpbd_joint_targets",
                "measured_finger_joint_velocity_source": (
                    "newton_eval_ik_post_step_name_resolved_joint_qd"
                ),
                "measured_finger_joint_names": list(
                    self.parameters.finger_joint_names
                ),
                "robot_target_write_backend": "indexed_scatter_controlled_robot_coordinates_and_dofs_only",
                "robot_initial_arm_joint_positions": list(
                    self.parameters.nominal_arm_joint_positions
                ),
                "robot_initial_gripper_width_m": (
                    self.parameters.open_gripper_width_m
                ),
                "robot_initial_joint_velocity_zero_verified": True,
                "robot_initial_joint_position_config_verified": True,
                "robot_initial_joint_acceleration_semantics": (
                    "zero_before_first_control_step"
                ),
                "arm_joint_velocity_limits": {
                    name: 1.5 for name in self.parameters.arm_joint_names
                },
                "finger_joint_velocity_limits": {
                    name: 0.2 for name in self.parameters.finger_joint_names
                },
                "robot_body_state_write_backend": "none",
                "robot_body_indices_written": [],
                "joint_force_write_backend": "none",
                "joint_pd_controller_used": True,
                "door_actuation": "passive_velocity_damping_only",
                "door_position_actuation": "none",
                "door_runtime_position_write_count": 0,
                "door_runtime_velocity_write_count": 0,
                "door_runtime_target_write_count": 0,
                "door_runtime_generalized_force_write_count": 0,
                "door_zero_write_evidence": "static_indexed_control_path_guarantee",
                "door_coord_excluded_from_driver": True,
                "door_dof_excluded_from_driver": True,
                "door_coord_excluded_from_target_writer": True,
                "door_dof_excluded_from_target_writer": True,
                "door_target_values_unchanged_verified": True,
                "robot_body_inverse_mass_positive_verified": True,
                "robot_body_flags_dynamic_verified": True,
                "robot_object_body_index_sets_disjoint": True,
                "door_reference_semantics": "diagnostic_only_never_applied",
                "collision_evidence_scope": "cross_asset_robot_object",
                "collision_margin_m": 0.003,
                "pose_layout": "xyz_wxyz",
                "grasp_parent_child_collision_filtered": False,
            },
        )


class PhysicsCliIntegrationTests(unittest.TestCase):
    def _config(self, directory: Path) -> Path:
        data = copy.deepcopy(
            json.loads((ROOT / "configs/microwave_franka.json").read_text())
        )
        data["project_root"] = str(ROOT)
        data["assets"]["object"]["urdf"] = str(
            ROOT / "assets/microwave/microwave.urdf"
        )
        data["assets"]["object"]["affordances"] = str(
            ROOT / "assets/microwave/affordances.json"
        )
        data["assets"]["robot"]["urdf"] = str(
            ROOT / "assets/microwave/microwave.urdf"
        )
        data["assets"]["robot"]["expected_urdf_sha256"] = (
            "d6ba39f326d52a02efe6c4292accc8503e32c3a19a5462a90e564cddf52177a1"
        )
        data["assets"]["robot"]["arm_joint_names"] = ["door_hinge"]
        data["assets"]["robot"]["default_joint_positions"] = [0.25]
        for phase in (
            "pregrasp",
            "approach",
            "close",
            "actuate",
            "release",
            "retreat",
        ):
            data["task"]["phases"][phase]["samples"] = 1
        data["output"]["root"] = str(directory / "configured-output")
        path = directory / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_handle_frame_composes_full_local_rigid_transform(self) -> None:
        root_half = np.sqrt(0.5)
        world_link = pose_matrix(
            [1.0, 2.0, 3.0], [root_half, 0.0, 0.0, root_half]
        )
        link_frame = pose_matrix(
            [0.2, 0.0, 0.0], [root_half, root_half, 0.0, 0.0]
        )
        position, quaternion = decompose_pose(world_link)
        actual = handle_frame_world_from_link_poses(
            np.concatenate((position, quaternion))[None, :], link_frame
        )[0]
        expected = compose_transforms(world_link, link_frame)
        np.testing.assert_allclose(actual, expected, atol=1.0e-12)
        np.testing.assert_allclose(actual[:3, 3], [1.0, 2.2, 3.0], atol=1.0e-12)

    def test_motion_metric_metadata_is_validated_and_name_ordered(self) -> None:
        import src.physics_assisted as module

        metadata = {
            "robot_initial_arm_joint_positions": [0.1, 0.2],
            "arm_joint_velocity_limits": {"joint_b": 2.0, "joint_a": 1.0},
        }
        initial, limits = module._physics_arm_motion_metric_inputs(
            metadata,
            ("joint_a", "joint_b"),
            (0.1, 0.2),
        )
        np.testing.assert_allclose(initial, [0.1, 0.2])
        np.testing.assert_allclose(limits, [1.0, 2.0])

        finger_limits = module._physics_finger_motion_metric_inputs(
            {
                "finger_joint_velocity_limits": {
                    "finger_right": 0.2,
                    "finger_left": 0.1,
                }
            },
            ("finger_left", "finger_right"),
        )
        self.assertEqual(list(finger_limits), ["finger_left", "finger_right"])
        self.assertEqual(
            finger_limits, {"finger_left": 0.1, "finger_right": 0.2}
        )

        with self.assertRaises(PipelineError) as caught:
            module._physics_arm_motion_metric_inputs(
                metadata,
                ("joint_a", "joint_b"),
                (0.1, 0.3),
            )
        self.assertEqual(caught.exception.stage, "physics_audit")

        invalid_limits = dict(metadata)
        invalid_limits["arm_joint_velocity_limits"] = {
            "joint_a": 1.0,
            "unexpected": 2.0,
        }
        with self.assertRaises(PipelineError) as caught:
            module._physics_arm_motion_metric_inputs(
                invalid_limits,
                ("joint_a", "joint_b"),
                (0.1, 0.2),
            )
        self.assertEqual(
            caught.exception.details["field"], "arm_joint_velocity_limits"
        )

        with self.assertRaises(PipelineError) as caught:
            module._physics_finger_motion_metric_inputs(
                {
                    "finger_joint_velocity_limits": {
                        "finger_left": 0.1,
                        "unexpected": 0.2,
                    }
                },
                ("finger_left", "finger_right"),
            )
        self.assertEqual(caught.exception.stage, "physics_audit")
        self.assertEqual(
            caught.exception.details["field"],
            "finger_joint_velocity_limits",
        )

    def test_measured_endpoint_qd_is_a_fail_closed_physics_gate(self) -> None:
        import src.physics_assisted as module

        computed = {"success": True, "gates": {}}
        result = module._with_measured_arm_motion_acceptance(
            computed,
            measured_arm_joint_qd=np.asarray([[0.25], [2.0]]),
            joint_velocity_limits_rad_s=np.asarray([1.0]),
            sample_dt_s=0.1,
            max_velocity_limit_ratio=1.0,
            max_acceleration_rad_s2=5.0,
            max_jerk_rad_s3=50.0,
        )
        self.assertFalse(result["success"])
        self.assertFalse(
            result["gates"]["max_measured_joint_velocity_limit_ratio"][
                "passed"
            ]
        )
        self.assertFalse(
            result["gates"]["final_measured_joint_velocity_limit_ratio"][
                "passed"
            ]
        )
        self.assertFalse(
            result["gates"]["max_measured_joint_acceleration_rad_s2"][
                "passed"
            ]
        )
        self.assertFalse(
            result["gates"]["max_measured_joint_jerk_rad_s3"]["passed"]
        )
        for gate_name in (
            "max_measured_joint_velocity_limit_ratio",
            "final_measured_joint_velocity_limit_ratio",
            "max_measured_joint_acceleration_rad_s2",
            "max_measured_joint_jerk_rad_s3",
        ):
            self.assertEqual(
                result["gates"][gate_name]["source"],
                "newton_eval_ik_post_step_joint_qd",
            )
        self.assertEqual(
            result["final_measured_arm_joint_velocity_rad_s"], [2.0]
        )

    def test_measured_finger_derivatives_are_named_si_gates_from_rest(self) -> None:
        import src.physics_assisted as module

        computed = {"success": True, "gates": {}}
        result = module._with_measured_finger_motion_acceptance(
            computed,
            measured_finger_joint_qd=np.asarray(
                [[0.01, -0.01], [0.02, -0.02]]
            ),
            finger_joint_names=("finger_right", "finger_left"),
            finger_velocity_limits_m_s={
                "finger_left": 0.1,
                "finger_right": 0.2,
            },
            sample_dt_s=0.1,
            max_velocity_limit_ratio=1.0,
            max_acceleration_m_s2=1.5,
            max_jerk_m_s3=30.0,
        )

        self.assertTrue(result["success"])
        self.assertEqual(
            result["measured_finger_joint_names"],
            ["finger_right", "finger_left"],
        )
        self.assertEqual(
            result["measured_finger_joint_velocity_limits_m_s"],
            {"finger_right": 0.2, "finger_left": 0.1},
        )
        self.assertEqual(
            result["measured_finger_motion_initial_velocity_m_s"],
            {"finger_right": 0.0, "finger_left": 0.0},
        )
        self.assertEqual(
            result["measured_finger_motion_initial_acceleration_m_s2"],
            {"finger_right": 0.0, "finger_left": 0.0},
        )
        self.assertEqual(result["measured_finger_acceleration_sample_count"], 2)
        self.assertEqual(result["measured_finger_jerk_sample_count"], 2)
        self.assertAlmostEqual(
            result["max_measured_finger_acceleration_m_s2"], 0.1
        )
        self.assertAlmostEqual(result["max_measured_finger_jerk_m_s3"], 1.0)
        for value in result[
            "per_finger_max_measured_acceleration_m_s2"
        ].values():
            self.assertAlmostEqual(value, 0.1)
        self.assertEqual(
            result["max_measured_finger_jerk_frame_index"], 0
        )
        self.assertEqual(
            result["max_measured_finger_jerk_finger_index"], 0
        )
        self.assertEqual(
            result["max_measured_finger_jerk_finger_name"], "finger_right"
        )
        self.assertEqual(
            result["final_measured_finger_velocity_m_s"],
            {"finger_right": 0.02, "finger_left": -0.02},
        )
        self.assertTrue(
            result["gates"]["max_measured_finger_acceleration_m_s2"][
                "passed"
            ]
        )
        self.assertTrue(
            result["gates"]["max_measured_finger_jerk_m_s3"]["passed"]
        )

    def test_measured_finger_jerk_and_invalid_evidence_fail_closed(self) -> None:
        import src.physics_assisted as module

        result = module._with_measured_finger_motion_acceptance(
            {"success": True, "gates": {}},
            measured_finger_joint_qd=np.asarray([[0.02, 0.0]]),
            finger_joint_names=("finger_a", "finger_b"),
            finger_velocity_limits_m_s={"finger_a": 0.2, "finger_b": 0.2},
            sample_dt_s=0.02,
            max_velocity_limit_ratio=1.0,
            max_acceleration_m_s2=1.5,
            max_jerk_m_s3=30.0,
        )
        self.assertTrue(
            result["gates"]["max_measured_finger_acceleration_m_s2"][
                "passed"
            ]
        )
        self.assertFalse(
            result["gates"]["max_measured_finger_jerk_m_s3"]["passed"]
        )
        self.assertFalse(result["success"])
        self.assertAlmostEqual(result["max_measured_finger_jerk_m_s3"], 50.0)

        for qd, names in (
            (np.asarray([[np.nan, 0.0]]), ("finger_a", "finger_b")),
            (np.zeros((1, 2)), ("finger_a", "finger_a")),
            (np.zeros((1, 1)), ("finger_a", "finger_b")),
        ):
            with self.subTest(qd_shape=qd.shape, names=names):
                with self.assertRaises(ValueError):
                    module._with_measured_finger_motion_acceptance(
                        {"success": True, "gates": {}},
                        measured_finger_joint_qd=qd,
                        finger_joint_names=names,
                        finger_velocity_limits_m_s={
                            name: 0.2 for name in set(names)
                        },
                        sample_dt_s=0.02,
                        max_velocity_limit_ratio=1.0,
                        max_acceleration_m_s2=1.5,
                        max_jerk_m_s3=30.0,
                    )

    def test_measured_finger_velocity_overshoot_is_a_fail_closed_gate(self) -> None:
        import src.physics_assisted as module

        result = module._with_measured_finger_motion_acceptance(
            {"success": True, "gates": {}},
            measured_finger_joint_qd=np.asarray([[0.1], [0.2], [0.3]]),
            finger_joint_names=("finger",),
            finger_velocity_limits_m_s={"finger": 0.2},
            sample_dt_s=0.1,
            max_velocity_limit_ratio=1.0,
            max_acceleration_m_s2=1.5,
            max_jerk_m_s3=30.0,
        )

        self.assertFalse(result["success"])
        self.assertTrue(
            result["gates"]["max_measured_finger_acceleration_m_s2"][
                "passed"
            ]
        )
        self.assertTrue(
            result["gates"]["max_measured_finger_jerk_m_s3"]["passed"]
        )
        self.assertFalse(
            result["gates"]["max_measured_finger_velocity_limit_ratio"][
                "passed"
            ]
        )
        self.assertFalse(
            result["gates"]["final_measured_finger_velocity_limit_ratio"][
                "passed"
            ]
        )
        self.assertAlmostEqual(
            result["max_measured_finger_velocity_limit_ratio"], 1.5
        )

    def test_physics_acceptance_fails_closed_when_reference_fails(self) -> None:
        import src.physics_assisted as module

        physics_pass = {
            "success": True,
            "gates": {
                "physics_metric": {
                    "value": 0.0,
                    "operator": "<=",
                    "threshold": 1.0,
                    "passed": True,
                }
            },
        }
        failed = module._with_kinematic_reference_acceptance(physics_pass, False)
        self.assertFalse(failed["success"])
        self.assertFalse(
            failed["gates"]["kinematic_reference_acceptance"]["passed"]
        )
        self.assertNotIn("kinematic_reference_acceptance", physics_pass["gates"])

        passed = module._with_kinematic_reference_acceptance(physics_pass, True)
        self.assertTrue(passed["success"])
        self.assertTrue(
            passed["gates"]["kinematic_reference_acceptance"]["passed"]
        )

    def test_reference_json_and_missing_door_audit_fail_closed(self) -> None:
        import src.physics_assisted as module

        with tempfile.TemporaryDirectory() as temporary:
            invalid = Path(temporary) / "invalid.json"
            invalid.write_text('{"value": NaN}', encoding="utf-8")
            with self.assertRaises(PipelineError) as caught:
                module._read_json_mapping(invalid, stage="physics_reference")
            self.assertEqual(caught.exception.code, FailureCode.OUTPUT_FAILURE)
            self.assertEqual(caught.exception.stage, "physics_reference")

        commands = PhysicsCommandTrajectory(
            np.asarray(["pregrasp", "approach", "close", "actuate", "retreat"]),
            np.zeros((5, 7)),
            np.asarray([0.08, 0.08, 0.03, 0.03, 0.08]),
            np.zeros(5),
        )
        params = PhysicsParameters.from_project_config(
            load_config(ROOT / "configs/microwave_franka.json")
        )
        rollout = _FakePhysicsSimulator(params).run(commands)
        metadata = dict(rollout.metadata)
        del metadata["door_runtime_target_write_count"]
        with self.assertRaises(PipelineError) as caught:
            module._validate_physics_rollout(
                replace(rollout, metadata=metadata), commands
            )
        self.assertEqual(caught.exception.stage, "physics_audit")
        self.assertEqual(
            caught.exception.details["field"],
            "door_runtime_target_write_count",
        )

        bad_initial_velocity = dict(rollout.metadata)
        bad_initial_velocity["robot_initial_joint_velocity_zero_verified"] = False
        with self.assertRaises(PipelineError) as caught:
            module._validate_physics_rollout(
                replace(rollout, metadata=bad_initial_velocity), commands
            )
        self.assertEqual(caught.exception.stage, "physics_audit")
        self.assertEqual(
            caught.exception.details["field"],
            "robot_initial_joint_velocity_zero_verified",
        )

        bad_time = rollout.time_s.copy()
        bad_time[0] = 0.0
        with self.assertRaises(PipelineError) as caught:
            module._validate_physics_rollout(
                replace(rollout, time_s=bad_time), commands
            )
        self.assertEqual(caught.exception.stage, "physics_audit")

    def test_measured_finger_rollout_evidence_is_required_and_name_aligned(self) -> None:
        import src.physics_assisted as module

        commands = PhysicsCommandTrajectory(
            np.asarray(["pregrasp", "approach", "close", "actuate", "retreat"]),
            np.zeros((5, 7)),
            np.asarray([0.08, 0.08, 0.03, 0.03, 0.08]),
            np.zeros(5),
        )
        params = PhysicsParameters.from_project_config(
            load_config(ROOT / "configs/microwave_franka.json")
        )
        rollout = _FakePhysicsSimulator(params).run(commands)

        legacy_payload = {
            name: getattr(rollout, name)
            for name in rollout.__dataclass_fields__
            if name != "measured_finger_joint_qd"
        }
        with self.assertRaises(PipelineError) as caught:
            module._validate_physics_rollout(
                SimpleNamespace(**legacy_payload), commands
            )
        self.assertEqual(caught.exception.code, FailureCode.PHYSICS_UNAVAILABLE)
        self.assertEqual(caught.exception.stage, "physics_result")

        nonfinite = rollout.measured_finger_joint_qd.copy()
        nonfinite[2, 1] = np.nan
        with self.assertRaises(PipelineError) as caught:
            module._validate_physics_rollout(
                replace(rollout, measured_finger_joint_qd=nonfinite),
                commands,
            )
        self.assertEqual(caught.exception.code, FailureCode.NUMERICAL_INSTABILITY)
        self.assertEqual(caught.exception.stage, "physics_result")

        with self.assertRaises(PipelineError) as caught:
            module._validate_physics_rollout(
                rollout,
                commands,
                expected_finger_joint_names=tuple(
                    reversed(params.finger_joint_names)
                ),
            )
        self.assertEqual(caught.exception.code, FailureCode.PHYSICS_UNAVAILABLE)
        self.assertEqual(caught.exception.stage, "physics_result")

        with self.assertRaises(PipelineError) as caught:
            module._validate_physics_rollout(
                replace(rollout, finger_joint_names=None), commands
            )
        self.assertEqual(caught.exception.code, FailureCode.PHYSICS_UNAVAILABLE)
        self.assertEqual(caught.exception.stage, "physics_result")
        self.assertIn("finger_joint_names", caught.exception.details)

        with self.assertRaises(PipelineError) as caught:
            module._validate_physics_rollout(
                replace(
                    rollout,
                    finger_joint_names=(
                        params.finger_joint_names[0],
                        None,
                    ),
                ),
                commands,
            )
        self.assertEqual(caught.exception.code, FailureCode.PHYSICS_UNAVAILABLE)
        self.assertEqual(caught.exception.stage, "physics_result")

    def test_cli_namespaces_reference_and_writes_measured_physics_artifacts(self) -> None:
        import src.physics_assisted as module

        from tests.test_run import FakeBackend, FakeCollisionEvaluator

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._config(root)
            output_dir = root / "physics-output"
            collision = FakeCollisionEvaluator()
            with (
                mock.patch("src.run.NewtonFrankaIKBackend", FakeBackend),
                mock.patch(
                    "src.run._default_collision_factory",
                    side_effect=lambda config, kinematics, backend: collision,
                ),
                mock.patch(
                    "src.physics_assisted.NewtonPhysicsAssistedSimulator",
                    _FakePhysicsSimulator,
                ),
                mock.patch.object(
                    module,
                    "compute_metrics",
                    wraps=module.compute_metrics,
                ) as metrics_spy,
            ):
                exit_code = main(
                    [
                        "--config",
                        str(config_path),
                        "--mode",
                        "physics_assisted",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 3)
            self.assertEqual(metrics_spy.call_count, 1)
            self.assertIs(
                metrics_spy.call_args.kwargs[
                    "include_position_difference_acceleration_jerk_gates"
                ],
                False,
            )
            for name in (
                "asset_inspection.json",
                "affordance_candidates.json",
                "collision_report.json",
                "rollout.jsonl",
                "trajectory.npz",
                "metrics.json",
                "resolved_config.json",
                "run.log",
            ):
                self.assertTrue((output_dir / name).is_file(), name)

            metrics = json.loads((output_dir / "metrics.json").read_text())
            frame_count = 6
            sample_dt_s = float(
                json.loads(config_path.read_text())["simulation"]["dt"]
            )
            self.assertEqual(metrics["mode"], "physics_assisted")
            self.assertEqual(metrics["run_status"], "acceptance_failed")
            reserve_audit = metrics["reference_arm_joint_reserve_audit"]
            self.assertTrue(reserve_audit["passed"])
            self.assertEqual(
                reserve_audit["arm_joint_tracking_reserve_rad"], 0.05
            )
            self.assertEqual(reserve_audit["control_limit_margin_rad"], 0.02)
            self.assertEqual(
                reserve_audit["min_hard_limit_clearance_rad"], 0.25
            )
            self.assertEqual(
                reserve_audit["min_hard_limit_clearance_sample_scope"],
                "initial_state",
            )
            self.assertEqual(
                reserve_audit["min_hard_limit_clearance_sample_index"], 0
            )
            self.assertIsNone(
                reserve_audit["min_hard_limit_clearance_frame_index"]
            )
            self.assertIsNone(reserve_audit["min_hard_limit_clearance_phase"])
            self.assertEqual(reserve_audit["reference_frame_count"], frame_count)
            self.assertEqual(
                reserve_audit["audited_sample_count"], frame_count + 1
            )
            self.assertEqual(
                reserve_audit["min_hard_limit_clearance_joint_name"],
                "door_hinge",
            )
            self.assertEqual(reserve_audit["control_bound_touch_count"], 0)
            self.assertTrue(
                metrics["gates"]["kinematic_reference_acceptance"]["passed"]
            )
            self.assertTrue(metrics["kinematic_reference"]["acceptance_passed"])
            self.assertEqual(metrics["collision_scope"], "cross_asset_robot_object")
            self.assertEqual(metrics["collision_frame_count"], 1)
            self.assertAlmostEqual(
                metrics["collision_frame_ratio"], 1.0 / frame_count
            )
            self.assertEqual(metrics["final_door_angle_deg"], 0.0)
            self.assertTrue(metrics["initial_joint_q_provided"])
            self.assertEqual(metrics["joint_velocity_limits_rad_s"], [1.5])
            self.assertEqual(metrics["joint_velocity_sample_count"], frame_count)
            self.assertEqual(
                metrics["joint_acceleration_sample_count"], frame_count
            )
            self.assertEqual(metrics["joint_jerk_sample_count"], frame_count)
            self.assertEqual(
                metrics["joint_motion_source"],
                "finite_difference_of_joint_q",
            )
            self.assertEqual(
                metrics["joint_motion_joint_q_source"],
                "newton_eval_ik_post_step_joint_q",
            )
            self.assertFalse(
                metrics[
                    "position_difference_acceleration_jerk_gates_enabled"
                ]
            )
            self.assertEqual(
                metrics["position_difference_acceleration_acceptance_role"],
                "diagnostic_only",
            )
            self.assertEqual(
                metrics["position_difference_jerk_acceptance_role"],
                "diagnostic_only",
            )
            self.assertNotIn(
                "max_joint_acceleration_rad_s2", metrics["gates"]
            )
            self.assertNotIn("max_joint_jerk_rad_s3", metrics["gates"])
            self.assertIn("max_joint_velocity_limit_ratio", metrics["gates"])
            self.assertIn("joint_limit_violation_count", metrics["gates"])
            self.assertIn(
                "joint_limit_violation_frame_ratio", metrics["gates"]
            )
            self.assertEqual(
                metrics["measured_joint_velocity_sample_count"], frame_count
            )
            self.assertEqual(metrics["final_measured_arm_joint_velocity_rad_s"], [0.0])
            self.assertTrue(
                metrics["gates"][
                    "max_measured_joint_velocity_limit_ratio"
                ]["passed"]
            )
            self.assertIn(
                "max_measured_joint_acceleration_rad_s2", metrics["gates"]
            )
            self.assertIn("max_measured_joint_jerk_rad_s3", metrics["gates"])
            expected_finger_names = [
                "panda_finger_joint1",
                "panda_finger_joint2",
            ]
            self.assertEqual(
                metrics["measured_finger_joint_names"], expected_finger_names
            )
            self.assertEqual(
                metrics["measured_finger_velocity_sample_count"], frame_count
            )
            self.assertEqual(
                metrics["measured_finger_acceleration_sample_count"],
                frame_count,
            )
            self.assertEqual(
                metrics["measured_finger_jerk_sample_count"], frame_count
            )
            expected_finger_acceleration = {
                "panda_finger_joint1": 0.0001 / sample_dt_s,
                "panda_finger_joint2": 0.0002 / sample_dt_s,
            }
            expected_final_finger_velocity = {
                "panda_finger_joint1": 0.0006,
                "panda_finger_joint2": -0.0012,
            }
            for name in expected_finger_names:
                self.assertAlmostEqual(
                    metrics["per_finger_max_measured_acceleration_m_s2"][
                        name
                    ],
                    expected_finger_acceleration[name],
                )
                self.assertAlmostEqual(
                    metrics["final_measured_finger_velocity_m_s"][name],
                    expected_final_finger_velocity[name],
                )
            self.assertEqual(
                metrics["measured_finger_joint_velocity_limits_m_s"],
                {name: 0.2 for name in expected_finger_names},
            )
            self.assertAlmostEqual(
                metrics["max_measured_finger_velocity_limit_ratio"], 0.006
            )
            self.assertAlmostEqual(
                metrics["final_measured_finger_velocity_limit_ratio"], 0.006
            )
            self.assertTrue(
                metrics["gates"][
                    "max_measured_finger_velocity_limit_ratio"
                ]["passed"]
            )
            self.assertTrue(
                metrics["gates"][
                    "final_measured_finger_velocity_limit_ratio"
                ]["passed"]
            )
            self.assertTrue(
                metrics["gates"][
                    "max_measured_finger_acceleration_m_s2"
                ]["passed"]
            )
            self.assertTrue(
                metrics["gates"]["max_measured_finger_jerk_m_s3"]["passed"]
            )
            self.assertEqual(
                metrics["physics_metadata"]["state_sample_timing"],
                "post_step_end_of_frame",
            )
            self.assertIs(
                metrics["physics_metadata"][
                    "grasp_parent_child_collision_filtered"
                ],
                False,
            )
            self.assertTrue(
                metrics["door_runtime_write_audit"]["guaranteed_zero_runtime_writes"]
            )
            self.assertEqual(
                metrics["door_runtime_write_audit"]["runtime_write_counts"],
                {"q": 0, "qd": 0, "target": 0, "generalized_force": 0},
            )
            log_rows = [
                json.loads(line)
                for line in (output_dir / "run.log").read_text().splitlines()
            ]
            reserve_events = [
                row
                for row in log_rows
                if row["event"]
                == "physics_reference_arm_joint_reserve_audited"
            ]
            self.assertEqual(len(reserve_events), 1)
            self.assertTrue(reserve_events[0]["details"]["audit"]["passed"])

            with np.load(output_dir / "trajectory.npz", allow_pickle=False) as data:
                np.testing.assert_allclose(
                    data["time_s"],
                    (np.arange(frame_count, dtype=float) + 1.0)
                    * sample_dt_s,
                    rtol=0.0,
                    atol=0.0,
                )
                np.testing.assert_allclose(data["door_angle_rad"], 0.0)
                np.testing.assert_array_equal(
                    data["arm_joint_names"], ["door_hinge"]
                )
                np.testing.assert_allclose(data["initial_arm_joint_q"], [0.25])
                np.testing.assert_allclose(
                    data["arm_joint_velocity_limits_rad_s"], [1.5]
                )
                np.testing.assert_allclose(
                    data["measured_arm_joint_qd"],
                    np.zeros((frame_count, 1)),
                )
                self.assertEqual(
                    data["applied_arm_joint_target_q"].dtype,
                    np.dtype(np.float32),
                )
                self.assertEqual(
                    data["applied_arm_joint_target_qd"].dtype,
                    np.dtype(np.float32),
                )
                np.testing.assert_array_equal(
                    data["applied_arm_joint_target_q"],
                    data["reference_command_arm_joint_q"].astype(np.float32),
                )
                expected_finger_qd = (
                    (np.arange(frame_count, dtype=float) + 1.0)[:, None]
                    * np.asarray([[0.0001, -0.0002]])
                )
                np.testing.assert_allclose(
                    data["measured_finger_joint_qd"], expected_finger_qd
                )

                np.testing.assert_array_equal(
                    data["finger_joint_names"], expected_finger_names
                )
                self.assertAlmostEqual(
                    float(data["initial_gripper_width_m"]), 0.08
                )
                self.assertGreater(float(data["reference_door_angle_rad"][-1]), 1.0)
                np.testing.assert_array_equal(
                    data["collision_flags"],
                    [False, False, False, True, False, False],
                )
                np.testing.assert_allclose(
                    data["handle_link_pose_wxyz"][:, :3],
                    np.tile([1.0, 2.0, 3.0], (frame_count, 1)),
                )
                np.testing.assert_allclose(
                    data["handle_world"][:, :3, 3],
                    np.tile([1.0, 2.0, 3.0], (frame_count, 1)),
                )

            first_rollout_row = json.loads(
                (output_dir / "rollout.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(
                first_rollout_row["finger_joint_velocities_m_s"],
                {
                    "panda_finger_joint1": 0.0001,
                    "panda_finger_joint2": -0.0002,
                },
            )
            self.assertEqual(
                first_rollout_row["command_arm_joint_position_semantics"],
                "planned_kinematic_reference_not_necessarily_applied",
            )
            self.assertIn(
                "door_hinge",
                first_rollout_row["applied_arm_joint_target_positions"],
            )
            self.assertEqual(
                first_rollout_row[
                    "applied_arm_joint_target_velocities_rad_s"
                ]["door_hinge"],
                0.0,
            )

            report = json.loads((output_dir / "collision_report.json").read_text())
            self.assertEqual(report["trajectory"]["scope"], "cross_asset_robot_object")
            self.assertEqual(report["trajectory"]["collision_frame_count"], 1)
            self.assertEqual(
                report["trajectory"]["frames"][3]["forbidden_contact_pairs"],
                [
                    {
                        "links": ["robot/panda_link5", "object/microwave_door"],
                        "signed_clearance_m": -0.001,
                    }
                ],
            )
            self.assertTrue((output_dir / "kinematic_reference" / "trajectory.npz").is_file())
            first_log = json.loads((output_dir / "run.log").read_text().splitlines()[0])
            self.assertEqual(first_log["details"]["mode"], "physics_assisted")

            def reject_constant(token: str) -> None:
                raise AssertionError(f"non-standard JSON constant: {token}")

            for path in output_dir.glob("*.json"):
                json.loads(path.read_text(), parse_constant=reject_constant)
            for path in output_dir.glob("*.jsonl"):
                for line in path.read_text().splitlines():
                    json.loads(line, parse_constant=reject_constant)

    def test_reference_reserve_failure_stops_before_expensive_physics(self) -> None:
        from tests.test_run import FakeBackend, FakeCollisionEvaluator

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._config(root)
            config_data = json.loads(config_path.read_text())
            config_data["simulation"]["robot_control"][
                "arm_joint_tracking_reserve_rad"
            ] = 0.30
            config_path.write_text(json.dumps(config_data), encoding="utf-8")
            output_dir = root / "reserve-failure"
            collision = FakeCollisionEvaluator()
            with (
                mock.patch("src.run.NewtonFrankaIKBackend", FakeBackend),
                mock.patch(
                    "src.run._default_collision_factory",
                    side_effect=lambda config, kinematics, backend: collision,
                ),
                mock.patch(
                    "src.physics_assisted.simulate_physics_assisted"
                ) as physics_spy,
            ):
                exit_code = main(
                    [
                        "--config",
                        str(config_path),
                        "--mode",
                        "physics_assisted",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 2)
            physics_spy.assert_not_called()
            metrics = json.loads((output_dir / "metrics.json").read_text())
            self.assertFalse(metrics["success"])
            self.assertEqual(metrics["run_status"], "failed")
            failure = metrics["failure"]
            self.assertEqual(failure["code"], FailureCode.IK_UNREACHABLE.value)
            self.assertEqual(failure["stage"], "ik_motion_limits")
            details = failure["details"]
            self.assertEqual(details["constraint"], "arm_joint_tracking_reserve")
            self.assertEqual(details["failed_checks"], ["hard_limit_clearance"])
            self.assertEqual(details["min_hard_limit_clearance_rad"], 0.25)
            self.assertEqual(details["control_bound_touch_count"], 0)
            self.assertFalse(details["passed"])
            log_rows = [
                json.loads(line)
                for line in (output_dir / "run.log").read_text().splitlines()
            ]
            self.assertEqual(
                log_rows[-2]["event"],
                "physics_reference_arm_joint_reserve_audited",
            )
            self.assertEqual(log_rows[-1]["event"], "run_failed")

    def test_initial_reserve_failures_stop_before_expensive_physics(self) -> None:
        from tests.test_run import FakeBackend, FakeCollisionEvaluator

        cases = (
            {
                "name": "hard_reserve_only",
                "initial_q": 0.03,
                "reserve": 0.05,
                "failed_checks": ["hard_limit_clearance"],
                "scope_field": "min_hard_limit_clearance_sample_scope",
                "frame_field": "min_hard_limit_clearance_frame_index",
                "phase_field": "min_hard_limit_clearance_phase",
            },
            {
                "name": "control_touch_only",
                "initial_q": 0.02,
                "reserve": 0.01,
                "failed_checks": ["control_bound_touch"],
                "scope_field": "first_control_bound_touch_sample_scope",
                "frame_field": "first_control_bound_touch_frame_index",
                "phase_field": "first_control_bound_touch_phase",
            },
        )
        for case in cases:
            with (
                self.subTest(case=case["name"]),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                config_path = self._config(root)
                config_data = json.loads(config_path.read_text())
                config_data["assets"]["robot"]["default_joint_positions"] = [
                    case["initial_q"]
                ]
                config_data["simulation"]["dt"] = 1.0
                config_data["simulation"]["robot_control"][
                    "arm_joint_tracking_reserve_rad"
                ] = case["reserve"]
                config_path.write_text(json.dumps(config_data), encoding="utf-8")
                output_dir = root / case["name"]
                collision = FakeCollisionEvaluator()
                with (
                    mock.patch("src.run.NewtonFrankaIKBackend", FakeBackend),
                    mock.patch(
                        "src.run._default_collision_factory",
                        side_effect=lambda config, kinematics, backend: collision,
                    ),
                    mock.patch(
                        "src.physics_assisted.simulate_physics_assisted"
                    ) as physics_spy,
                ):
                    exit_code = main(
                        [
                            "--config",
                            str(config_path),
                            "--mode",
                            "physics_assisted",
                            "--output-dir",
                            str(output_dir),
                        ]
                    )

                self.assertEqual(exit_code, 2)
                physics_spy.assert_not_called()
                metrics = json.loads((output_dir / "metrics.json").read_text())
                details = metrics["failure"]["details"]
                self.assertEqual(details["failed_checks"], case["failed_checks"])
                self.assertEqual(details[case["scope_field"]], "initial_state")
                self.assertIsNone(details[case["frame_field"]])
                self.assertIsNone(details[case["phase_field"]])
                self.assertEqual(details["reference_frame_count"], 6)
                self.assertEqual(details["audited_sample_count"], 7)
                self.assertEqual(
                    details["trajectory_hard_limit_reserve_violation_count"],
                    0,
                )
                self.assertEqual(
                    details["trajectory_control_bound_touch_count"], 0
                )


if __name__ == "__main__":
    unittest.main()
