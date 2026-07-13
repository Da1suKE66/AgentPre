from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from src.config import load_config
from src.errors import FailureCode, PipelineError
from src.physics_assisted import (
    NewtonPhysicsAssistedSimulator,
    PhysicsCommandTrajectory,
    PhysicsParameters,
    PhysicsRollout,
    ScalarJointRef,
    build_robot_position_targets,
    fixed_grasp_activation_window,
    evaluate_fixed_grasp_activation_gate,
    handle_frame_world_from_link_poses,
    kinematic_body_twists,
    planned_fixed_grasp_anchors,
    plan_massless_fixed_joint_collapse,
    require_newton_v13,
    resolve_named_scalar_joints,
)
from src.run import main
from src.transforms import compose_transforms, decompose_pose, pose_matrix


ROOT = Path(__file__).resolve().parents[1]


def ref(name: str, coord: int, dof: int) -> ScalarJointRef:
    return ScalarJointRef(
        configured_name=name,
        label=f"robot/{name}",
        joint_index=coord,
        coord_index=coord,
        dof_index=dof,
    )


class PhysicsTrajectoryTests(unittest.TestCase):
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

    def test_fixed_grasp_activates_after_close_and_releases_at_retreat(self) -> None:
        phases = np.asarray(
            ["pregrasp", "approach", "close", "close", "actuate", "actuate", "retreat"]
        )
        window = fixed_grasp_activation_window(phases, "close")
        self.assertEqual(window.activation_frame, 4)
        self.assertEqual(window.release_frame, 6)
        self.assertFalse(window.is_active(3))
        self.assertTrue(window.is_active(4))
        self.assertTrue(window.is_active(5))
        self.assertFalse(window.is_active(6))

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
            hand_world, handle_world, anchors, frame_index=4
        )
        self.assertTrue(passing.passed)

        remote_hand = hand_world.copy()
        remote_hand[0, 3] += 0.02
        rejected = evaluate_fixed_grasp_activation_gate(
            remote_hand, handle_world, anchors, frame_index=4
        )
        self.assertFalse(rejected.passed)
        self.assertGreater(rejected.position_error_m, 0.015)

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
    def test_physics_scene_starts_at_first_robot_command(self) -> None:
        params = PhysicsParameters.from_project_config(
            load_config("configs/microwave_franka.json")
        )
        simulator = NewtonPhysicsAssistedSimulator(params)
        builder = SimpleNamespace(joint_q=[-9.0, -9.0, -9.0, 0.0])
        simulator._set_builder_initial_coordinates(
            builder,
            (ref("arm", 0, 0),),
            (ref("left", 1, 1), ref("right", 2, 2)),
            ref("door", 3, 3),
            [0.7],
            0.05,
        )
        np.testing.assert_allclose(
            builder.joint_q,
            [0.7, 0.025, 0.025, 0.0],
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
        self.assertEqual(params.robot_control_backend, "kinematic_body_driver")

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
        pairs = tuple(
            (("robot/panda_link5", "object/microwave_door"),)
            if index == 3
            else ()
            for index in range(frame_count)
        )
        return PhysicsRollout(
            phase_names=commands.phase_names.copy(),
            time_s=np.arange(frame_count, dtype=float) * self.parameters.dt,
            command_joint_q=full_q.copy(),
            measured_joint_q=full_q.copy(),
            measured_joint_qd=np.zeros_like(full_q),
            robot_driver_arm_joint_q=commands.arm_joint_targets.copy(),
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
                [False, False, False, True, False]
            ),
            external_robot_joint_force_command=np.zeros_like(full_q),
            forbidden_contact_pairs=pairs,
            forbidden_contact_signed_clearance_m=tuple(
                (-0.001,) if index == 3 else ()
                for index in range(frame_count)
            ),
            body_labels=("robot/panda_hand", "object/handle"),
            joint_labels=("robot/door_hinge", "object/door_hinge"),
            metadata={
                "status": "completed",
                "backend": "fake_newton_1_3",
                "control_backend": "kinematic_body_driver",
                "door_actuation": "none",
                "door_runtime_position_write_count": 0,
                "door_runtime_velocity_write_count": 0,
                "door_runtime_target_write_count": 0,
                "door_runtime_generalized_force_write_count": 0,
                "door_zero_write_evidence": "static_indexed_control_path_guarantee",
                "door_coord_excluded_from_driver": True,
                "door_dof_excluded_from_driver": True,
                "robot_object_body_index_sets_disjoint": True,
                "door_reference_semantics": "diagnostic_only_never_applied",
                "collision_evidence_scope": "cross_asset_robot_object",
                "collision_margin_m": 0.003,
                "pose_layout": "xyz_wxyz",
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
        data["assets"]["robot"]["arm_joint_names"] = ["door_hinge"]
        data["assets"]["robot"]["default_joint_positions"] = [0.25]
        for phase in ("pregrasp", "approach", "close", "actuate", "retreat"):
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

    def test_cli_namespaces_reference_and_writes_measured_physics_artifacts(self) -> None:
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
            self.assertEqual(metrics["mode"], "physics_assisted")
            self.assertEqual(metrics["run_status"], "acceptance_failed")
            self.assertEqual(metrics["collision_scope"], "cross_asset_robot_object")
            self.assertEqual(metrics["collision_frame_count"], 1)
            self.assertAlmostEqual(metrics["collision_frame_ratio"], 0.2)
            self.assertEqual(metrics["final_door_angle_deg"], 0.0)
            self.assertTrue(
                metrics["door_runtime_write_audit"]["guaranteed_zero_runtime_writes"]
            )
            self.assertEqual(
                metrics["door_runtime_write_audit"]["runtime_write_counts"],
                {"q": 0, "qd": 0, "target": 0, "generalized_force": 0},
            )

            with np.load(output_dir / "trajectory.npz", allow_pickle=False) as data:
                np.testing.assert_allclose(data["door_angle_rad"], 0.0)
                self.assertGreater(float(data["reference_door_angle_rad"][-1]), 1.0)
                np.testing.assert_array_equal(
                    data["collision_flags"], [False, False, False, True, False]
                )
                np.testing.assert_allclose(
                    data["handle_link_pose_wxyz"][:, :3],
                    np.tile([1.0, 2.0, 3.0], (5, 1)),
                )
                np.testing.assert_allclose(
                    data["handle_world"][:, :3, 3],
                    np.tile([1.0, 2.0, 3.0], (5, 1)),
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


if __name__ == "__main__":
    unittest.main()
