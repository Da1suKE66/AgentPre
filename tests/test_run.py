from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from src.errors import FailureCode
from src.newton_backend import (
    IKTrajectoryResult,
    IKWaypointResult,
    PoseError,
    WaypointValidation,
)
from src.run import _finite_or_none, _numeric_json, main
from src.transforms import pose_matrix


ROOT = Path(__file__).resolve().parents[1]


class FakeBackend:
    """Exact Cartesian backend used only to isolate CLI orchestration."""

    instances: list["FakeBackend"] = []

    def __init__(self, parameters) -> None:
        self.parameters = parameters
        self.solve_lengths: list[int] = []
        type(self).instances.append(self)

    def initial_tcp_transform(self) -> np.ndarray:
        return pose_matrix([0.0, -0.4, 0.6], [1.0, 0.0, 0.0, 0.0])

    def solve_waypoints(self, target_positions, target_orientations_wxyz):
        self.solve_lengths.append(len(target_positions))
        results = []
        for index, (position, orientation) in enumerate(
            zip(target_positions, target_orientations_wxyz, strict=True)
        ):
            validation = WaypointValidation(
                success=True,
                pose_error=PoseError(position_m=0.0, orientation_rad=0.0),
                joint_limit_violations=(),
                has_nonfinite=False,
                failed_checks=(),
                failure_codes=(),
            )
            results.append(
                IKWaypointResult(
                    waypoint_index=index,
                    joint_positions=(0.25,),
                    arm_joint_positions=(0.25,),
                    target_position=tuple(float(value) for value in position),
                    target_orientation_wxyz=tuple(
                        float(value) for value in orientation
                    ),
                    actual_position=tuple(float(value) for value in position),
                    actual_orientation_wxyz=tuple(
                        float(value) for value in orientation
                    ),
                    objective_cost=0.0,
                    validation=validation,
                )
            )
        return IKTrajectoryResult(tuple(results))


class FakeCollisionEvaluator:
    def __init__(self, *, reject_candidate: bool = False) -> None:
        self.reject_candidate = reject_candidate
        self.candidate_ids: list[str] = []
        self.trajectory_calls = 0

    def candidate_is_collision_free(self, candidate, gripper_world, arm_joint_q):
        self.candidate_ids.append(candidate.candidate_id)
        self.asserted_target_shape = np.asarray(gripper_world).shape
        assert np.asarray(arm_joint_q).shape == (1,)
        if self.reject_candidate:
            return False, "synthetic candidate collision"
        return True

    def trajectory_collision_flags(self, plan, arm_joint_q):
        self.trajectory_calls += 1
        assert np.asarray(arm_joint_q).shape == (len(plan.phase_names), 1)
        return np.zeros(len(plan.phase_names), dtype=bool)


class RunTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeBackend.instances.clear()

    def _config(self, directory: Path, *, samples: int = 2) -> Path:
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
        # The fake backend does not parse a robot; this valid one-joint URDF
        # lets the runner prove name-based limit extraction independently.
        data["assets"]["robot"]["urdf"] = str(
            ROOT / "assets/microwave/microwave.urdf"
        )
        data["assets"]["robot"]["arm_joint_names"] = ["door_hinge"]
        data["assets"]["robot"]["default_joint_positions"] = [0.25]
        for phase in ("pregrasp", "approach", "close", "actuate", "retreat"):
            data["task"]["phases"][phase]["samples"] = samples
        data["output"]["root"] = str(directory / "configured-output")
        path = directory / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_nonfinite_rollout_diagnostics_are_json_null(self) -> None:
        payload = {
            "objective_cost": _finite_or_none(float("nan")),
            "realized_pose": _numeric_json(
                np.asarray([[float("inf"), 1.0], [2.0, float("nan")]])
            ),
        }
        self.assertEqual(
            payload,
            {
                "objective_cost": None,
                "realized_pose": [[None, 1.0], [2.0, None]],
            },
        )
        json.dumps(payload, allow_nan=False)

    def test_cli_writes_complete_config_driven_five_phase_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._config(root, samples=2)
            output_dir = root / "explicit-output"
            collision = FakeCollisionEvaluator()
            with (
                mock.patch("src.run.NewtonFrankaIKBackend", FakeBackend),
                mock.patch(
                    "src.run._default_collision_factory",
                    side_effect=lambda config, kinematics, backend: collision,
                ),
            ):
                exit_code = main(
                    [
                        "--config",
                        str(config_path),
                        "--mode",
                        "kinematic",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            for name in (
                "rollout.jsonl",
                "trajectory.npz",
                "metrics.json",
                "run.log",
                "asset_inspection.json",
                "affordance_candidates.json",
                "collision_report.json",
                "resolved_config.json",
            ):
                self.assertTrue((output_dir / name).is_file(), name)

            rows = [
                json.loads(line)
                for line in (output_dir / "rollout.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 10)
            self.assertEqual(
                [rows[index]["phase"] for index in range(0, 10, 2)],
                ["pregrasp", "approach", "close", "actuate", "retreat"],
            )
            self.assertEqual(rows[0]["arm_joint_positions"].keys(), {"door_hinge"})
            self.assertTrue(all(row["ik"]["success"] for row in rows))
            self.assertTrue(all(not row["collision"] for row in rows))

            with np.load(output_dir / "trajectory.npz") as trajectory:
                self.assertEqual(trajectory["arm_joint_q"].shape, (10, 1))
                self.assertEqual(trajectory["target_gripper_world"].shape, (10, 4, 4))
                self.assertEqual(trajectory["achieved_gripper_world"].shape, (10, 4, 4))
                np.testing.assert_array_equal(
                    trajectory["phase_names"],
                    [
                        "pregrasp",
                        "pregrasp",
                        "approach",
                        "approach",
                        "close",
                        "close",
                        "actuate",
                        "actuate",
                        "retreat",
                        "retreat",
                    ],
                )

            metrics = json.loads((output_dir / "metrics.json").read_text())
            self.assertTrue(metrics["success"])
            self.assertEqual(metrics["run_status"], "success")
            self.assertEqual(metrics["frame_count"], 10)
            self.assertEqual(metrics["ik_waypoint_success_rate"], 1.0)
            self.assertEqual(metrics["collision_frame_ratio"], 0.0)
            self.assertEqual(metrics["collision_scope"], "cross_asset_robot_object")
            self.assertEqual(FakeBackend.instances[0].solve_lengths, [1, 10])
            self.assertEqual(collision.candidate_ids, ["frame:handle_grasp"])
            self.assertEqual(collision.trajectory_calls, 1)

            resolved = json.loads((output_dir / "resolved_config.json").read_text())
            self.assertEqual(
                resolved["assets"]["object"]["urdf"],
                str(ROOT / "assets/microwave/microwave.urdf"),
            )
            self.assertEqual(
                resolved["resolved_runtime"]["output_dir"], str(output_dir.resolve())
            )
            self.assertEqual(
                resolved["resolved_runtime"]["environment"]["CUDA_VISIBLE_DEVICES"],
                "",
            )

    def test_cli_writes_structured_collision_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._config(root, samples=1)
            output_dir = root / "failed-output"
            collision = FakeCollisionEvaluator(reject_candidate=True)
            with (
                mock.patch("src.run.NewtonFrankaIKBackend", FakeBackend),
                mock.patch(
                    "src.run._default_collision_factory",
                    side_effect=lambda config, kinematics, backend: collision,
                ),
            ):
                exit_code = main(
                    [
                        "--config",
                        str(config_path),
                        "--mode",
                        "kinematic",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 2)
            metrics = json.loads((output_dir / "metrics.json").read_text())
            self.assertFalse(metrics["success"])
            self.assertEqual(metrics["run_status"], "failed")
            self.assertEqual(metrics["failure"]["code"], FailureCode.COLLISION.value)
            self.assertEqual(metrics["failure"]["stage"], "candidate_selection")
            lines = (output_dir / "run.log").read_text().splitlines()
            self.assertGreater(len(lines), 1)
            log = json.loads(lines[-1])
            self.assertEqual(log["event"], "run_failed")

    def test_default_output_root_allocates_non_overwriting_run_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._config(root, samples=1)

            def build_collision(config, kinematics, backend):
                return FakeCollisionEvaluator()

            with (
                mock.patch("src.run.NewtonFrankaIKBackend", FakeBackend),
                mock.patch(
                    "src.run._default_collision_factory",
                    side_effect=build_collision,
                ),
            ):
                self.assertEqual(
                    main(["--config", str(config_path), "--mode", "kinematic"]),
                    0,
                )
                self.assertEqual(
                    main(["--config", str(config_path), "--mode", "kinematic"]),
                    0,
                )

            output_root = root / "configured-output"
            run_dirs = sorted(path.name for path in output_root.iterdir())
            self.assertEqual(
                run_dirs,
                [
                    "kinematic_seed_20260714_0001",
                    "kinematic_seed_20260714_0002",
                ],
            )
            self.assertTrue(
                (output_root / run_dirs[0] / "metrics.json").is_file()
            )
            self.assertTrue(
                (output_root / run_dirs[1] / "metrics.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
