from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np

from src.door_kinematics import DoorKinematics
from src.trajectory import generate_task_plan
from src.transforms import compose_transforms, invert_transform, pose_matrix
from src.urdf_model import load_urdf


ROOT = Path(__file__).resolve().parents[1]


class DoorTrajectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        model = load_urdf(ROOT / "assets/microwave/microwave.urdf")
        self.kinematics = DoorKinematics(
            model=model,
            root_world_transform=pose_matrix([0.4, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0]),
            door_joint_name="door_hinge",
            door_link_name="microwave_door",
            handle_link_name="handle",
        )

    def test_handle_fk_follows_arbitrary_joint_axis(self) -> None:
        closed = self.kinematics.handle_frame_transform(0.0, np.eye(4))
        opened = self.kinematics.handle_frame_transform(math.radians(65.0), np.eye(4))
        np.testing.assert_allclose(closed[:3, 3], [0.58, -0.28, 0.5], atol=1.0e-9)
        self.assertLess(opened[1, 3], closed[1, 3])
        lower, upper = self.kinematics.limits
        self.assertEqual(lower, 0.0)
        self.assertGreater(upper, math.radians(65.0))

    def test_explicit_handle_frame_may_be_authored_on_the_door_link(self) -> None:
        same_link_kinematics = DoorKinematics(
            model=self.kinematics.model,
            root_world_transform=self.kinematics.root_world_transform,
            door_joint_name="door_hinge",
            door_link_name="microwave_door",
            handle_link_name="microwave_door",
        )
        # This is the checked-in fixture's door->handle_mount transform.  A
        # single-link Articraft export can encode it directly in affordances.json.
        door_to_handle_frame = pose_matrix(
            [0.43, -0.0525, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        )

        for angle in (0.0, math.radians(65.0)):
            explicit_frame_world = same_link_kinematics.handle_frame_transform(
                angle,
                door_to_handle_frame,
            )
            separate_link_world = self.kinematics.handle_frame_transform(
                angle,
                np.eye(4),
            )
            np.testing.assert_allclose(
                explicit_frame_world,
                separate_link_world,
                atol=1.0e-12,
            )

    def test_five_phase_plan_and_fixed_grasp_transform(self) -> None:
        phase_samples = {
            "pregrasp": 2,
            "approach": 3,
            "close": 2,
            "actuate": 5,
            "retreat": 2,
        }
        handle_to_gripper = pose_matrix([0.0, -0.01, 0.0], [0.0, 1.0, 0.0, 0.0])
        plan = generate_task_plan(
            kinematics=self.kinematics,
            link_to_handle_frame=np.eye(4),
            handle_approach_axis=np.asarray([0.0, 1.0, 0.0]),
            handle_to_gripper=handle_to_gripper,
            initial_gripper_world=pose_matrix([0.0, -0.4, 0.6], [1.0, 0.0, 0.0, 0.0]),
            closed_angle_rad=0.0,
            goal_angle_rad=math.radians(65.0),
            phase_samples=phase_samples,
            pregrasp_distance_m=0.12,
            retreat_distance_m=0.1,
            open_gripper_width_m=0.08,
            closed_gripper_width_m=0.028,
            dt=1.0 / 60.0,
        )
        self.assertEqual(len(plan.time_s), sum(phase_samples.values()))
        self.assertEqual(set(plan.phase_names.tolist()), set(phase_samples))
        self.assertAlmostEqual(plan.door_angle_rad[-1], math.radians(65.0))
        self.assertTrue(np.isfinite(plan.target_gripper_world).all())

        actuate = np.flatnonzero(plan.phase_names == "actuate")
        for index in actuate:
            relative = compose_transforms(
                invert_transform(plan.handle_world[index]),
                plan.target_gripper_world[index],
            )
            np.testing.assert_allclose(relative, handle_to_gripper, atol=1.0e-8)


if __name__ == "__main__":
    unittest.main()
