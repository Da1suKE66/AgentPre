from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import textwrap
import unittest

import numpy as np

from src.affordances import load_affordances
from src.asset_semantics import (
    SemanticInferenceError,
    infer_microwave_semantics,
    infer_task_semantics,
)
from src.transforms import quaternion_to_matrix


def articraft_style_urdf() -> str:
    """Small structural replica of the current materialized Articraft record."""

    return textwrap.dedent(
        """\
        <robot name="commercial_microwave">
          <link name="cabinet"><collision><geometry><box size="0.56 0.42 0.34"/></geometry></collision></link>
          <link name="door">
            <collision name="door_panel">
              <origin xyz="0.2625 0 0"/><geometry><box size="0.475 0.03 0.216"/></geometry>
            </collision>
            <collision name="viewing_glass">
              <origin xyz="0.22 -0.017 0"/><geometry><box size="0.375 0.004 0.14"/></geometry>
            </collision>
            <collision name="hinge_barrel"><geometry><cylinder radius="0.012" length="0.205"/></geometry></collision>
            <collision name="pull_grip">
              <origin xyz="0.44 -0.055 0"/><geometry><cylinder radius="0.017" length="0.172"/></geometry>
            </collision>
            <collision name="handle_mount_0">
              <origin xyz="0.44 -0.036 0.064"/><geometry><box size="0.044 0.043 0.02"/></geometry>
            </collision>
            <collision name="handle_mount_1">
              <origin xyz="0.44 -0.036 -0.064"/><geometry><box size="0.044 0.043 0.02"/></geometry>
            </collision>
          </link>
          <link name="turntable"><collision><geometry><cylinder radius="0.155" length="0.01"/></geometry></collision></link>
          <link name="selector_knob_0"><collision><geometry><cylinder radius="0.026" length="0.03"/></geometry></collision></link>
          <joint name="door_hinge" type="revolute">
            <parent link="cabinet"/><child link="door"/><axis xyz="0 0 -1"/>
            <limit lower="0" upper="1.75" effort="18" velocity="1.2"/>
          </joint>
          <joint name="turntable_spin" type="continuous">
            <parent link="cabinet"/><child link="turntable"/><axis xyz="0 0 1"/>
          </joint>
          <joint name="knob_0_spin" type="continuous">
            <parent link="cabinet"/><child link="selector_knob_0"/><axis xyz="0 0 1"/>
          </joint>
        </robot>
        """
    )


def obfuscated_urdf() -> str:
    """No door/handle words: topology and geometry must carry the decision."""

    return textwrap.dedent(
        """\
        <robot name="opaque_appliance">
          <link name="body"><collision><geometry><box size="0.6 0.45 0.4"/></geometry></collision></link>
          <link name="moving_a">
            <collision name="surface_a"><origin xyz="0.25 0 0"/><geometry><box size="0.5 0.035 0.3"/></geometry></collision>
          </link>
          <link name="rail_a"><collision><geometry><box size="0.025 0.035 0.22"/></geometry></collision></link>
          <joint name="j0" type="revolute">
            <parent link="body"/><child link="moving_a"/><axis xyz="0 0 1"/>
            <limit lower="0" upper="1.5" effort="20" velocity="1"/>
          </joint>
          <joint name="fixed_a" type="fixed">
            <parent link="moving_a"/><child link="rail_a"/><origin xyz="0.43 -0.055 0"/>
          </joint>
          <link name="moving_b"><collision><geometry><cylinder radius="0.025" length="0.03"/></geometry></collision></link>
          <joint name="j1" type="revolute">
            <parent link="body"/><child link="moving_b"/><axis xyz="0 0 1"/>
            <limit lower="-0.3" upper="0.3" effort="1" velocity="2"/>
          </joint>
        </robot>
        """
    )


class AssetSemanticInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_urdf(self, contents: str, name: str = "model.urdf") -> Path:
        path = self.root / name
        path.write_text(contents, encoding="utf-8")
        return path

    def test_current_fixture_infers_separate_handle_link(self) -> None:
        result = infer_microwave_semantics(
            Path(__file__).parents[1] / "assets/microwave/microwave.urdf"
        )
        self.assertEqual(result.door_joint_name, "door_hinge")
        self.assertEqual(result.door_link_name, "microwave_door")
        self.assertEqual(result.handle.link_name, "handle")
        self.assertEqual(result.handle.position, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(result.goal_angle_deg, 65.0)
        self.assertGreater(result.door_confidence, 0.9)
        self.assertGreater(result.handle_confidence, 0.9)

    def test_articraft_same_link_pull_grip_matches_calibrated_frame(self) -> None:
        result = infer_microwave_semantics(self.write_urdf(articraft_style_urdf()))
        self.assertEqual(result.door_joint_name, "door_hinge")
        self.assertEqual(result.door_link_name, "door")
        self.assertEqual(result.handle.link_name, "door")
        self.assertEqual(result.handle.geometry_name, "pull_grip")
        np.testing.assert_allclose(result.handle.position, [0.44, -0.055, 0.0], atol=1e-12)
        np.testing.assert_allclose(result.handle.quaternion_wxyz, [1, 0, 0, 0], atol=1e-12)
        self.assertEqual(result.handle.gripper_closing_axis, (1.0, 0.0, 0.0))
        self.assertEqual(result.handle.approach_axis, (0.0, 1.0, 0.0))
        self.assertAlmostEqual(result.handle.recommended_gripper_width_m, 0.044)

        rotation = quaternion_to_matrix(result.handle.quaternion_wxyz)
        approach = rotation @ np.asarray(result.handle.approach_axis)
        pregrasp = np.asarray(result.handle.position) - 0.12 * approach
        # Front/exterior is negative Y in this record; pregrasp must not be in the oven.
        self.assertLess(pregrasp[1], result.handle.position[1])

    def test_geometry_and_topology_work_without_semantic_names(self) -> None:
        result = infer_microwave_semantics(self.write_urdf(obfuscated_urdf()))
        self.assertEqual(result.door_joint_name, "j0")
        self.assertEqual(result.door_link_name, "moving_a")
        self.assertEqual(result.handle.link_name, "rail_a")
        self.assertEqual(result.handle.geometry_name, "collision_0")
        self.assertGreater(result.door_candidates[0].score, result.door_candidates[1].score)
        self.assertGreater(result.handle_candidates[0].score, result.handle_candidates[1].score)

    def test_equal_door_candidates_fail_closed(self) -> None:
        urdf = textwrap.dedent(
            """\
            <robot name="ambiguous">
              <link name="base"/>
              <link name="door_a"><collision><origin xyz="0.2 0 0"/><geometry><box size="0.4 0.03 0.3"/></geometry></collision></link>
              <link name="door_b"><collision><origin xyz="0.2 0 0"/><geometry><box size="0.4 0.03 0.3"/></geometry></collision></link>
              <joint name="door_hinge_a" type="revolute"><parent link="base"/><child link="door_a"/><axis xyz="0 0 1"/><limit lower="0" upper="1.4"/></joint>
              <joint name="door_hinge_b" type="revolute"><parent link="base"/><child link="door_b"/><axis xyz="0 0 1"/><limit lower="0" upper="1.4"/></joint>
            </robot>
            """
        )
        with self.assertRaises(SemanticInferenceError) as raised:
            infer_microwave_semantics(self.write_urdf(urdf))
        self.assertEqual(raised.exception.code, "DOOR_INFERENCE_AMBIGUOUS")
        self.assertEqual(len(raised.exception.context["candidates"]), 2)

    def test_equal_handle_candidates_fail_closed(self) -> None:
        source = articraft_style_urdf().replace(
            """<collision name="pull_grip">
      <origin xyz="0.44 -0.055 0"/><geometry><cylinder radius="0.017" length="0.172"/></geometry>
    </collision>""",
            """<collision name="pull_grip_a">
      <origin xyz="0.44 -0.055 0.05"/><geometry><cylinder radius="0.017" length="0.172"/></geometry>
    </collision>
    <collision name="pull_grip_b">
      <origin xyz="0.44 -0.055 -0.05"/><geometry><cylinder radius="0.017" length="0.172"/></geometry>
    </collision>""",
        )
        with self.assertRaises(SemanticInferenceError) as raised:
            infer_microwave_semantics(self.write_urdf(source))
        self.assertEqual(raised.exception.code, "HANDLE_INFERENCE_AMBIGUOUS")

    def test_provider_returns_loadable_precise_affordance_sidecar(self) -> None:
        raw = infer_task_semantics(self.write_urdf(articraft_style_urdf()))
        self.assertEqual(raw["decisions"]["door_joint"]["value"], "door_hinge")
        self.assertEqual(raw["decisions"]["handle_link"]["value"], "door")
        self.assertIn("semantic_evidence", raw)
        sidecar = self.root / "affordances.json"
        sidecar.write_text(json.dumps(raw["affordance_payload"]), encoding="utf-8")
        parsed = load_affordances(sidecar)
        frame = parsed.require_frame("inferred_pull_grip")
        np.testing.assert_allclose(frame.position, [0.44, -0.055, 0.0])
        np.testing.assert_allclose(
            frame.quaternion_wxyz, [1, 0, 0, 0], atol=1e-12
        )
        self.assertEqual(frame.gripper_closing_axis, (1.0, 0.0, 0.0))
        self.assertEqual(frame.approach_axis, (0.0, 1.0, 0.0))
        self.assertAlmostEqual(frame.recommended_gripper_width_m, 0.044)


if __name__ == "__main__":
    unittest.main()
