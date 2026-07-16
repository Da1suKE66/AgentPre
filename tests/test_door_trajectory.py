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

    def test_six_phase_plan_and_fixed_grasp_transform(self) -> None:
        phase_samples = {
            "pregrasp": 2,
            "approach": 3,
            "close": 2,
            "actuate": 5,
            "release": 4,
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
        np.testing.assert_allclose(
            plan.time_s,
            (np.arange(sum(phase_samples.values()), dtype=float) + 1.0)
            * (1.0 / 60.0),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(set(plan.phase_names.tolist()), set(phase_samples))
        self.assertAlmostEqual(plan.door_angle_rad[-1], math.radians(65.0))
        self.assertTrue(np.isfinite(plan.target_gripper_world).all())

        grasped = np.flatnonzero(
            np.isin(plan.phase_names, ["actuate", "release"])
        )
        for index in grasped:
            relative = compose_transforms(
                invert_transform(plan.handle_world[index]),
                plan.target_gripper_world[index],
            )
            np.testing.assert_allclose(relative, handle_to_gripper, atol=1.0e-8)

        actuate = np.flatnonzero(plan.phase_names == "actuate")
        release = np.flatnonzero(plan.phase_names == "release")
        retreat = np.flatnonzero(plan.phase_names == "retreat")
        np.testing.assert_allclose(
            plan.door_angle_rad[release],
            math.radians(65.0),
            atol=0.0,
        )
        np.testing.assert_allclose(
            plan.handle_world[release],
            np.repeat(
                plan.handle_world[actuate[-1]][None, :, :], len(release), axis=0
            ),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            plan.target_gripper_world[release],
            np.repeat(
                plan.target_gripper_world[actuate[-1]][None, :, :],
                len(release),
                axis=0,
            ),
            atol=1.0e-12,
        )
        linear = np.arange(1, len(release) + 1, dtype=float) / len(release)
        progress = 10.0 * linear**3 - 15.0 * linear**4 + 6.0 * linear**5
        np.testing.assert_allclose(
            plan.gripper_width_m[release],
            (1.0 - progress) * 0.028 + progress * 0.08,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(plan.gripper_width_m[retreat], 0.08, atol=0.0)
        self.assertGreater(
            np.linalg.norm(
                plan.target_gripper_world[retreat[-1], :3, 3]
                - plan.target_gripper_world[release[-1], :3, 3]
            ),
            0.09,
        )

    def test_phase_progress_is_monotonic_and_c2_smooth_at_endpoints(self) -> None:
        sample_count = 40
        goal_angle = math.radians(65.0)
        plan = generate_task_plan(
            kinematics=self.kinematics,
            link_to_handle_frame=np.eye(4),
            handle_approach_axis=np.asarray([0.0, 1.0, 0.0]),
            handle_to_gripper=pose_matrix(
                [0.0, -0.01, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ),
            initial_gripper_world=pose_matrix(
                [0.0, -0.4, 0.6],
                [1.0, 0.0, 0.0, 0.0],
            ),
            closed_angle_rad=0.0,
            goal_angle_rad=goal_angle,
            phase_samples={phase: sample_count for phase in (
                "pregrasp",
                "approach",
                "close",
                "actuate",
                "release",
                "retreat",
            )},
            pregrasp_distance_m=0.12,
            retreat_distance_m=0.1,
            open_gripper_width_m=0.08,
            closed_gripper_width_m=0.028,
            dt=1.0 / 60.0,
        )

        actuate = np.flatnonzero(plan.phase_names == "actuate")
        progress = plan.door_angle_rad[actuate] / goal_angle
        linear = np.arange(1, sample_count + 1, dtype=float) / sample_count
        expected = 10.0 * linear**3 - 15.0 * linear**4 + 6.0 * linear**5

        np.testing.assert_allclose(progress, expected, atol=1.0e-12)
        self.assertTrue(np.all(np.diff(progress) >= 0.0))
        self.assertAlmostEqual(float(progress[-1]), 1.0)

        # The previous phase supplies the omitted t=0 sample.  Including it
        # here makes the finite differences exercise the actual phase join.
        joined_progress = np.concatenate(([0.0], progress))
        velocity = np.diff(joined_progress)
        acceleration = np.diff(joined_progress, n=2)

        peak_velocity = float(np.max(np.abs(velocity)))
        peak_acceleration = float(np.max(np.abs(acceleration)))
        self.assertLess(abs(float(velocity[0])), 0.01 * peak_velocity)
        self.assertLess(abs(float(velocity[-1])), 0.01 * peak_velocity)
        self.assertLess(abs(float(acceleration[0])), 0.25 * peak_acceleration)
        self.assertLess(abs(float(acceleration[-1])), 0.25 * peak_acceleration)


if __name__ == "__main__":
    unittest.main()
