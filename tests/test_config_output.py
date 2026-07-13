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
        invalid["simulation"]["robot_control"]["backend"] = "torque_pd"
        with self.assertRaises(PipelineError):
            validate_config(invalid)

    def test_effort_limited_pd_config_is_not_part_of_checked_in_schema(self) -> None:
        self.assertEqual(
            self.data["simulation"]["robot_control"]["backend"],
            "kinematic_body_driver",
        )
        self.assertNotIn("robot_pd", self.data["simulation"])

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
