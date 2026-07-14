from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.config import load_config, validate_config
from src.errors import FailureCode, PipelineError
from src.output import write_json, write_jsonl, write_trajectory


ROOT = Path(__file__).resolve().parents[1]


class ConfigOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = json.loads((ROOT / "configs/microwave_franka.json").read_text())

    def test_checked_in_config_is_valid_and_paths_are_explicit(self) -> None:
        config = load_config(ROOT / "configs/microwave_franka.json")
        self.assertEqual(config.get("runtime.device"), "cpu")
        self.assertEqual(
            config.get("assets.object.expected_urdf_sha256"),
            "d6ba39f326d52a02efe6c4292accc8503e32c3a19a5462a90e564cddf52177a1",
        )
        self.assertEqual(
            config.get("assets.robot.expected_urdf_sha256"),
            "ad9f5298a4d1a375cf16824b0de4f0d1c7cc446597964b80aa639ca830e998a1",
        )
        with tempfile.TemporaryDirectory() as directory:
            old = os.environ.get("AGENTPRE_CACHE_ROOT")
            os.environ["AGENTPRE_CACHE_ROOT"] = directory
            try:
                self.assertEqual(config.asset_path("robot").parent.parent.parent, Path(directory) / "assets")
            finally:
                if old is None:
                    os.environ.pop("AGENTPRE_CACHE_ROOT", None)
                else:
                    os.environ["AGENTPRE_CACHE_ROOT"] = old

    def test_asset_hashes_are_required_lowercase_sha256(self) -> None:
        for kind in ("object", "robot"):
            with self.subTest(kind=kind, case="missing"):
                invalid = copy.deepcopy(self.data)
                del invalid["assets"][kind]["expected_urdf_sha256"]
                with self.assertRaises(PipelineError) as caught:
                    validate_config(invalid)
                self.assertEqual(caught.exception.code, FailureCode.CONFIG_INVALID)
                self.assertEqual(caught.exception.stage, "config")

            for value in (
                "a" * 63,
                "A" * 64,
                "g" * 64,
                "a" * 65,
            ):
                with self.subTest(kind=kind, value=value):
                    invalid = copy.deepcopy(self.data)
                    invalid["assets"][kind]["expected_urdf_sha256"] = value
                    with self.assertRaises(PipelineError) as caught:
                        validate_config(invalid)
                    self.assertEqual(
                        caught.exception.details["field"],
                        f"assets.{kind}.expected_urdf_sha256",
                    )

    def test_cache_placeholder_defaults_to_required_remote_cache(self) -> None:
        config = load_config(ROOT / "configs/microwave_franka.json")
        old = os.environ.pop("AGENTPRE_CACHE_ROOT", None)
        try:
            self.assertEqual(
                config.asset_path("robot"),
                Path("/cache/liluchen/agentpre/assets/franka_description/robots/panda_arm_hand.urdf"),
            )
        finally:
            if old is not None:
                os.environ["AGENTPRE_CACHE_ROOT"] = old

    def test_bad_quaternion_and_gpu_mode_are_rejected(self) -> None:
        invalid = copy.deepcopy(self.data)
        invalid["assets"]["object"]["world_pose"]["orientation_wxyz"] = [0, 0, 0, 0]
        with self.assertRaises(PipelineError):
            validate_config(invalid)

        invalid = copy.deepcopy(self.data)
        invalid["runtime"]["threads"] = 2
        with self.assertRaises(PipelineError):
            validate_config(invalid)

        invalid = copy.deepcopy(self.data)
        invalid["runtime"]["device"] = "cuda:0"
        with self.assertRaises(PipelineError):
            validate_config(invalid)

        invalid = copy.deepcopy(self.data)
        invalid["collision"]["scope"] = "all_collisions"
        with self.assertRaises(PipelineError):
            validate_config(invalid)

        invalid = copy.deepcopy(self.data)
        invalid["simulation"]["robot_control"]["backend"] = "kinematic_body_driver"
        with self.assertRaises(PipelineError):
            validate_config(invalid)

    def test_joint_pd_controls_are_explicit_and_validated(self) -> None:
        self.assertEqual(
            self.data["simulation"]["robot_control"]["backend"],
            "joint_pd",
        )
        control = self.data["simulation"]["robot_control"]
        self.assertEqual(control["implementation"], "newton_xpbd_joint_targets")
        self.assertEqual(control["target_velocity_mode"], "finite_difference")
        for name in (
            "arm_stiffness",
            "arm_damping",
            "finger_stiffness",
            "finger_damping",
        ):
            invalid = copy.deepcopy(self.data)
            invalid["simulation"]["robot_control"][name] = 0.0
            with self.assertRaises(PipelineError):
                validate_config(invalid)

        self.assertGreater(self.data["ik"]["control_limit_margin_rad"], 0.0)
        invalid = copy.deepcopy(self.data)
        invalid["ik"]["control_limit_margin_rad"] = 0.0
        with self.assertRaises(PipelineError):
            validate_config(invalid)

        door_control = self.data["simulation"]["door_control"]
        self.assertEqual(door_control["backend"], "passive_velocity_damping")
        self.assertEqual(door_control["target_stiffness"], 0.0)
        self.assertGreater(door_control["target_damping"], 0.0)
        self.assertEqual(door_control["target_velocity_rad_s"], 0.0)
        invalid = copy.deepcopy(self.data)
        invalid["simulation"]["door_control"]["target_stiffness"] = 0.1
        with self.assertRaises(PipelineError):
            validate_config(invalid)
        invalid = copy.deepcopy(self.data)
        invalid["simulation"]["door_control"]["target_damping"] = 0.0
        with self.assertRaises(PipelineError):
            validate_config(invalid)
        invalid = copy.deepcopy(self.data)
        invalid["simulation"]["door_control"]["target_velocity_rad_s"] = 0.1
        with self.assertRaises(PipelineError):
            validate_config(invalid)

    def test_articraft_record_may_attach_handle_frame_to_door_link(self) -> None:
        record = copy.deepcopy(self.data)
        door_link = record["assets"]["object"]["door_link"]
        record["assets"]["object"]["handle_link"] = door_link
        record["collision"]["allowed_contact_links"][-1] = door_link

        validate_config(record)
        self.assertEqual(
            record["assets"]["object"]["handle_link"],
            record["assets"]["object"]["door_link"],
        )

    def test_checked_in_articraft_record_config_and_affordance_are_consistent(self) -> None:
        config = load_config(ROOT / "configs/articraft_microwave_franka.json")
        affordance_path = config.resolve_path(
            config.get("assets.object.affordances")
        )
        affordance = json.loads(affordance_path.read_text(encoding="utf-8"))
        source = json.loads(
            (affordance_path.parent / "source.json").read_text(encoding="utf-8")
        )
        frame = affordance["frames"][config.get("assets.object.handle_frame")]

        self.assertIn(
            "rec_microwave_oven_5e86f3429e954dcd9ab6c9d3a94db707",
            config.get("assets.object.source"),
        )
        self.assertEqual(config.get("assets.object.door_joint"), "door_hinge")
        self.assertEqual(config.get("assets.object.door_link"), "door")
        self.assertEqual(config.get("assets.object.handle_link"), "door")
        self.assertEqual(frame["link"], "door")
        self.assertEqual(frame["position"], [0.44, -0.055, 0.0])
        self.assertEqual(frame["approach_axis"], [0.0, 1.0, 0.0])
        self.assertIn(source["data_commit"], source["record_model_url"])
        self.assertIn(source["data_commit"], affordance["asset_source"])
        self.assertEqual(
            config.get("collision.allowed_contact_links"),
            ["panda_leftfinger", "panda_rightfinger", "door"],
        )

        object_world = np.asarray(
            config.get("assets.object.world_pose")["position"], dtype=float
        )
        hinge_origin = np.asarray(
            source["door_joint"]["origin_xyz_m"], dtype=float
        )
        handle_local = np.asarray(
            source["handle_geometry"]["center_xyz_m"], dtype=float
        )
        closed_handle_world = object_world + hinge_origin + handle_local
        np.testing.assert_allclose(
            closed_handle_world,
            source["scene_alignment"]["closed_handle_world_position_m"],
            atol=1.0e-12,
        )

        # The robot starts at negative world Y.  The authored +Y approach axis
        # must therefore move from an outward pregrasp toward the door, not
        # through the appliance from behind.
        approach_world = np.asarray(frame["approach_axis"], dtype=float)
        grasp_world = closed_handle_world + np.asarray(
            config.get("task.grasp_offset")["position"], dtype=float
        )
        pregrasp_world = grasp_world - approach_world * float(
            config.get("task.pregrasp_distance_m")
        )
        travel = grasp_world - pregrasp_world
        np.testing.assert_allclose(
            travel / np.linalg.norm(travel),
            approach_world,
            atol=1.0e-12,
        )
        robot_y = float(config.get("assets.robot.world_pose")["position"][1])
        self.assertLess(robot_y, pregrasp_world[1])
        self.assertLess(pregrasp_world[1], grasp_world[1])
        with tempfile.TemporaryDirectory() as directory:
            old = os.environ.get("AGENTPRE_CACHE_ROOT")
            os.environ["AGENTPRE_CACHE_ROOT"] = directory
            try:
                self.assertEqual(
                    config.asset_path("object"),
                    Path(directory)
                    / "assets"
                    / "articraft"
                    / "rec_microwave_oven_5e86f3429e954dcd9ab6c9d3a94db707"
                    / "model.urdf",
                )
            finally:
                if old is None:
                    os.environ.pop("AGENTPRE_CACHE_ROOT", None)
                else:
                    os.environ["AGENTPRE_CACHE_ROOT"] = old

    def test_reverse_door_motion_is_configurable_but_zero_motion_is_not(self) -> None:
        reverse = copy.deepcopy(self.data)
        reverse["task"]["closed_angle_deg"] = 0.0
        reverse["task"]["goal_angle_deg"] = -65.0
        validate_config(reverse)

        reverse["task"]["goal_angle_deg"] = 0.0
        with self.assertRaises(PipelineError):
            validate_config(reverse)

    def test_outputs_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "metrics.json", {"scalar": np.float32(1.5), "array": np.arange(3)})
            write_jsonl(root / "rollout.jsonl", [{"frame": 0}, {"frame": 1}])
            write_trajectory(root / "trajectory.npz", {"q": np.eye(2)})
            self.assertEqual(json.loads((root / "metrics.json").read_text())["array"], [0, 1, 2])
            self.assertEqual(len((root / "rollout.jsonl").read_text().splitlines()), 2)
            np.testing.assert_allclose(np.load(root / "trajectory.npz")["q"], np.eye(2))

    def test_json_outputs_reject_non_standard_nan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(PipelineError) as raised:
                write_json(root / "invalid.json", {"value": float("nan")})
            self.assertEqual(raised.exception.code, FailureCode.OUTPUT_FAILURE)
            with self.assertRaises(PipelineError) as raised:
                write_jsonl(root / "invalid.jsonl", [{"value": float("inf")}])
            self.assertEqual(raised.exception.code, FailureCode.OUTPUT_FAILURE)


if __name__ == "__main__":
    unittest.main()
