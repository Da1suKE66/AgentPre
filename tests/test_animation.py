from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

from src.animation import (
    AnimationExportError,
    load_animation_data,
    write_animation_html,
)


def _payload(document: str) -> dict[str, object]:
    match = re.search(
        r'<script id="agentpre-data" type="application/json">(.*?)</script>',
        document,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError("embedded trajectory payload is missing")
    return json.loads(match.group(1))


class AnimationTests(unittest.TestCase):
    def test_physics_body_poses_are_preferred_and_embedded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = root / "trajectory.npz"
            body_pose = np.zeros((3, 3, 7), dtype=float)
            body_pose[..., 3] = 1.0
            body_pose[:, :, 0] = np.asarray(
                [[0.0, 0.4, 0.8], [0.0, 0.5, 0.9], [0.0, 0.6, 1.0]]
            )
            np.savez_compressed(
                trajectory,
                body_pose_wxyz=body_pose,
                body_labels=np.asarray(
                    ["robot/panda_link0", "robot/panda_hand", "object/door_handle"]
                ),
                ee_pose_wxyz=np.asarray(
                    [[0.4, 0.0, 0.5, 1, 0, 0, 0], [0.5, 0.0, 0.5, 1, 0, 0, 0], [0.6, 0.0, 0.5, 1, 0, 0, 0]]
                ),
                handle_link_pose_wxyz=np.asarray(
                    [[0.8, 0.0, 0.5, 1, 0, 0, 0], [0.9, 0.0, 0.5, 1, 0, 0, 0], [1.0, 0.0, 0.5, 1, 0, 0, 0]]
                ),
                door_angle_rad=np.deg2rad([0.0, 15.0, 30.0]),
                time_s=np.asarray([4.0, 4.1, 4.2]),
                phase_names=np.asarray(["approach", "actuate", "actuate"]),
            )

            output = write_animation_html(
                trajectory, root / "animation.html", title="Microwave <run>"
            )
            document = output.read_text(encoding="utf-8")
            data = _payload(document)

            self.assertEqual(data["source"], "physics_body_pose")
            self.assertEqual(data["frameCount"], 3)
            self.assertEqual(data["timeS"], [0.0, 0.1, 0.2])
            self.assertEqual(data["bodyGroups"]["robot"], [0, 1])
            self.assertEqual(data["bodyGroups"]["highlighted"], [1, 2])
            self.assertEqual(data["doorAngleDeg"], [0.0, 15.0, 30.0])
            self.assertIn("Microwave &lt;run&gt;", document)
            self.assertIn('id="toggle"', document)
            self.assertIn('id="frame" type="range"', document)
            self.assertNotIn("fetch(", document)
            self.assertNotIn("<script src=", document)

    def test_kinematic_transform_fallback_has_paths_and_curve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = root / "trajectory.npz"
            tcp = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
            handle = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
            tcp[:, 0, 3] = [0.0, 0.1]
            handle[:, 0, 3] = [0.3, 0.2]
            np.savez_compressed(
                trajectory,
                achieved_gripper_world=tcp,
                handle_world=handle,
                door_angle_rad=np.asarray([0.0, 0.5]),
            )

            data = load_animation_data(trajectory, fallback_fps=20.0)

            self.assertEqual(data["source"], "kinematic")
            self.assertIsNone(data["bodyXY"])
            self.assertEqual(data["tcpSource"], "achieved_gripper_world")
            self.assertEqual(data["handleSource"], "handle_world")
            self.assertEqual(data["timeS"], [0.0, 0.05])
            self.assertEqual(data["tcpHandleCm"], [30.0, 10.0])

    def test_rejects_incomplete_or_nonfinite_physics_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incomplete = root / "incomplete.npz"
            np.savez_compressed(incomplete, body_pose_wxyz=np.zeros((1, 1, 7)))
            with self.assertRaisesRegex(AnimationExportError, "together"):
                load_animation_data(incomplete)

            nonfinite = root / "nonfinite.npz"
            pose = np.zeros((1, 1, 7))
            pose[0, 0, 0] = np.nan
            np.savez_compressed(
                nonfinite, body_pose_wxyz=pose, body_labels=np.asarray(["robot/base"])
            )
            with self.assertRaisesRegex(AnimationExportError, "NaN or Infinity"):
                load_animation_data(nonfinite)

    def test_labels_cannot_escape_the_embedded_json_script(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = root / "trajectory.npz"
            poses = np.zeros((1, 1, 7))
            poses[..., 3] = 1.0
            np.savez_compressed(
                trajectory,
                body_pose_wxyz=poses,
                body_labels=np.asarray(["</script><p>unsafe</p>"]),
            )
            output = write_animation_html(trajectory, root / "animation.html")
            document = output.read_text(encoding="utf-8")
            self.assertNotIn("</script><p>unsafe", document)
            self.assertEqual(
                _payload(document)["bodyLabels"], ["</script><p>unsafe</p>"]
            )


if __name__ == "__main__":
    unittest.main()
