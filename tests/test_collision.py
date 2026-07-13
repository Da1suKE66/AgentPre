from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import textwrap
import unittest

import numpy as np

from src.collision import (
    BACKEND_NAME,
    CollisionError,
    NamedAABBCollisionChecker,
    load_collision_shapes,
    named_link_fk,
)


def transform(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    value = np.eye(4, dtype=float)
    value[:3, 3] = [x, y, z]
    return value


def rotation_z(angle_rad: float) -> np.ndarray:
    value = np.eye(4, dtype=float)
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    value[:3, :3] = [
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return value


def single_box_urdf(robot_name: str, link_name: str) -> str:
    return textwrap.dedent(
        f"""\
        <robot name="{robot_name}">
          <link name="{link_name}">
            <collision name="long_box">
              <geometry><box size="4.0 0.2 0.2"/></geometry>
            </collision>
          </link>
        </robot>
        """
    )


def robot_urdf() -> str:
    return textwrap.dedent(
        """\
        <?xml version="1.0"?>
        <robot name="named_robot">
          <link name="arm">
            <collision name="arm_collision">
              <geometry><box size="0.2 0.2 0.2"/></geometry>
            </collision>
          </link>
          <link name="finger">
            <collision>
              <geometry><sphere radius="0.05"/></geometry>
            </collision>
          </link>
          <joint name="finger_slide" type="prismatic">
            <parent link="arm"/><child link="finger"/>
            <origin xyz="0.4 0 0"/>
            <axis xyz="1 0 0"/>
            <limit lower="0" upper="0.1" effort="1" velocity="1"/>
          </joint>
        </robot>
        """
    )


def object_urdf() -> str:
    return textwrap.dedent(
        """\
        <?xml version="1.0"?>
        <robot name="named_object">
          <link name="cabinet">
            <collision>
              <geometry><box size="0.2 0.2 0.2"/></geometry>
            </collision>
          </link>
          <link name="handle">
            <collision>
              <geometry><cylinder radius="0.04" length="0.2"/></geometry>
            </collision>
          </link>
          <joint name="handle_mount" type="fixed">
            <parent link="cabinet"/><child link="handle"/>
            <origin xyz="0.4 0 0"/>
          </joint>
        </robot>
        """
    )


class CollisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.robot_path = self.write("robot.urdf", robot_urdf())
        self.object_path = self.write("object.urdf", object_urdf())

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write(self, relative: str, contents: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        return path

    def checker(
        self, *, margin_m: float = 0.0, allowed: tuple[str, ...] = ()
    ) -> NamedAABBCollisionChecker:
        return NamedAABBCollisionChecker(
            self.robot_path,
            self.object_path,
            broad_phase="sap",
            margin_m=margin_m,
            allowed_contact_links=allowed,
        )

    @staticmethod
    def robot_transforms(
        *, arm_x: float = 0.0, finger_x: float = 1.0
    ) -> dict[str, np.ndarray]:
        return {"arm": transform(arm_x), "finger": transform(finger_x)}

    @staticmethod
    def object_transforms(
        *, cabinet_x: float = 3.0, handle_x: float = 1.0
    ) -> dict[str, np.ndarray]:
        return {"cabinet": transform(cabinet_x), "handle": transform(handle_x)}

    def test_named_cross_asset_overlap_is_reported_with_pair_and_reason(self) -> None:
        result = self.checker().check_frame(
            self.robot_transforms(), self.object_transforms(), frame_index=7
        )
        self.assertEqual(result.backend, BACKEND_NAME)
        self.assertEqual(result.frame_index, 7)
        self.assertTrue(result.collision)
        self.assertEqual(result.checked_shape_pairs, 1)
        self.assertEqual(result.potential_shape_pairs, 4)
        self.assertEqual(len(result.pairs), 1)
        pair = result.pairs[0]
        self.assertEqual((pair.robot_link, pair.object_link), ("finger", "handle"))
        self.assertEqual(pair.reason, "obb_overlap")
        self.assertLessEqual(pair.signed_clearance_m, 0.0)
        self.assertFalse(pair.allowed)
        self.assertTrue(pair.broad_phase_candidate)
        self.assertEqual(pair.narrow_phase, "obb_sat_15_axes")
        self.assertEqual(result.reasons, ("obb_overlap",))
        self.assertEqual(len(result.broad_phase_candidates), 1)
        json.dumps(result.to_dict())

    def test_grasp_contact_is_allowed_only_when_both_named_links_are_allowed(self) -> None:
        checker = self.checker(allowed=("finger", "handle"))
        intended = checker.check_candidate(
            self.robot_transforms(), self.object_transforms()
        )
        self.assertTrue(intended.collision_free)
        self.assertEqual(len(intended.pairs), 0)
        self.assertEqual(len(intended.allowed_pairs), 1)
        self.assertTrue(intended.allowed_pairs[0].allowed)

        # ``handle`` being allowed does not hide a collision with a non-allowed arm link.
        accidental = checker.check_candidate(
            self.robot_transforms(arm_x=1.0, finger_x=5.0),
            self.object_transforms(),
        )
        self.assertTrue(accidental.collision)
        self.assertEqual(
            {(pair.robot_link, pair.object_link) for pair in accidental.pairs},
            {("arm", "handle")},
        )

    def test_configured_margin_is_applied_once_to_pair_separation(self) -> None:
        # Finger sphere ends at x=1.05, handle AABB begins at x=1.09: 4 cm gap.
        far = self.object_transforms(handle_x=1.13)
        without_margin = self.checker(margin_m=0.039).check_frame(
            self.robot_transforms(), far
        )
        self.assertFalse(without_margin.collision)

        within_margin = self.checker(margin_m=0.041).check_frame(
            self.robot_transforms(), far
        )
        self.assertTrue(within_margin.collision)
        pair = within_margin.pairs[0]
        self.assertEqual(pair.reason, "within_margin")
        self.assertAlmostEqual(pair.signed_clearance_m, 0.04, places=12)

    def test_trajectory_returns_real_ordered_flags_and_frame_evidence(self) -> None:
        checker = self.checker()
        robot_frames = [self.robot_transforms()] * 3
        object_frames = [
            self.object_transforms(handle_x=2.0),
            self.object_transforms(handle_x=1.0),
            self.object_transforms(handle_x=-2.0),
        ]
        report = checker.check_trajectory(robot_frames, object_frames)
        self.assertEqual(report.flags, (False, True, False))
        self.assertEqual(report.collision_frame_count, 1)
        self.assertAlmostEqual(report.collision_frame_ratio, 1.0 / 3.0)
        self.assertEqual([frame.frame_index for frame in report.frames], [0, 1, 2])
        serialized = report.to_dict()
        self.assertEqual(serialized["scope"], "cross_asset_robot_object")
        self.assertTrue(serialized["conservative"])
        self.assertEqual(serialized["broad_phase"], "sap")
        self.assertEqual(serialized["broad_phase_backend"], "sap_world_aabb")
        self.assertEqual(serialized["narrow_phase"], "obb_sat_15_axes")
        self.assertEqual(serialized["flags"], [False, True, False])
        json.dumps(serialized)

    def test_rotated_long_boxes_can_overlap_in_aabb_without_obb_contact(self) -> None:
        robot_path = self.write(
            "long_robot.urdf", single_box_urdf("long_robot", "robot_bar")
        )
        object_path = self.write(
            "long_object.urdf", single_box_urdf("long_object", "object_bar")
        )
        checker = NamedAABBCollisionChecker(
            robot_path,
            object_path,
            broad_phase="sap",
            margin_m=0.0,
            allowed_contact_links=(),
        )
        robot_pose = rotation_z(math.pi / 4.0)
        # The bars are parallel with a 10 cm OBB gap along their thin axis.
        object_pose = robot_pose @ transform(y=0.3)

        robot_min, robot_max = checker.robot_shapes[0].world_bounds(robot_pose)
        object_min, object_max = checker.object_shapes[0].world_bounds(object_pose)
        self.assertTrue(
            np.all(np.minimum(robot_max, object_max) >= np.maximum(robot_min, object_min))
        )

        result = checker.check_frame(
            {"robot_bar": robot_pose}, {"object_bar": object_pose}
        )
        self.assertFalse(result.collision)
        self.assertEqual(result.checked_shape_pairs, 1)
        self.assertEqual(result.obb_tested_shape_pairs, 1)
        self.assertEqual(len(result.broad_phase_candidates), 1)
        candidate = result.broad_phase_candidates[0]
        self.assertTrue(candidate.broad_phase_candidate)
        self.assertLessEqual(candidate.broad_phase_signed_clearance_m, 0.0)
        self.assertEqual(candidate.reason, "obb_separated")
        self.assertAlmostEqual(candidate.signed_clearance_m, 0.1, places=12)
        self.assertGreaterEqual(candidate.tested_separating_axes, 6)
        self.assertEqual(result.pairs, ())

        serialized = result.to_dict()
        audit = serialized["broad_phase_candidates"][0]
        self.assertTrue(audit["broad_phase_candidate"])
        self.assertEqual(audit["narrow_phase"], "obb_sat_15_axes")
        self.assertEqual(audit["narrow_phase_outcome"], "obb_separated")

    def test_common_rigid_rotation_does_not_change_obb_result(self) -> None:
        robot_path = self.write(
            "invariant_robot.urdf", single_box_urdf("robot", "robot_bar")
        )
        object_path = self.write(
            "invariant_object.urdf", single_box_urdf("object", "object_bar")
        )
        checker = NamedAABBCollisionChecker(
            robot_path,
            object_path,
            broad_phase="sap",
            margin_m=0.0,
            allowed_contact_links=(),
        )
        robot_pose = rotation_z(math.pi / 4.0)
        object_pose = robot_pose @ transform(y=0.3)
        original = checker.check_frame(
            {"robot_bar": robot_pose}, {"object_bar": object_pose}
        )

        common_rotation = rotation_z(math.radians(31.0))
        rotated = checker.check_frame(
            {"robot_bar": common_rotation @ robot_pose},
            {"object_bar": common_rotation @ object_pose},
        )

        self.assertEqual(original.collision, rotated.collision)
        self.assertFalse(original.collision)
        self.assertEqual(len(original.broad_phase_candidates), 1)
        self.assertEqual(len(rotated.broad_phase_candidates), 1)
        self.assertEqual(
            original.broad_phase_candidates[0].reason,
            rotated.broad_phase_candidates[0].reason,
        )
        self.assertAlmostEqual(
            original.broad_phase_candidates[0].signed_clearance_m,
            rotated.broad_phase_candidates[0].signed_clearance_m,
            places=12,
        )

    def test_collision_origin_rotation_and_primitive_bounds_are_conservative(self) -> None:
        path = self.write(
            "rotated.urdf",
            textwrap.dedent(
                """\
                <robot name="rotated">
                  <link name="rotated_box">
                    <collision>
                      <origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/>
                      <geometry><box size="2 1 0.5"/></geometry>
                    </collision>
                  </link>
                </robot>
                """
            ),
        )
        shape = load_collision_shapes(path, asset="fixture")[0]
        minimum, maximum = shape.world_bounds(np.eye(4))
        np.testing.assert_allclose(minimum, [0.5, 1.0, 2.75], atol=1.0e-12)
        np.testing.assert_allclose(maximum, [1.5, 3.0, 3.25], atol=1.0e-12)

    def test_obj_mesh_scale_origin_and_package_resolution_are_used(self) -> None:
        package = self.root / "mesh_package"
        self.write(
            "mesh_package/meshes/shape.obj",
            "v -1 -2 -3\nv 1 2 3\nv -1 2 3\n",
        )
        urdf = self.write(
            "mesh_package/robots/mesh.urdf",
            textwrap.dedent(
                """\
                <robot name="mesh_fixture">
                  <link name="mesh_link">
                    <collision>
                      <origin xyz="1 0 0"/>
                      <geometry>
                        <mesh filename="package://mesh_package/meshes/shape.obj" scale="0.1 0.2 0.3"/>
                      </geometry>
                    </collision>
                  </link>
                </robot>
                """
            ),
        )
        shape = load_collision_shapes(urdf, asset="mesh")[0]
        minimum, maximum = shape.world_bounds(np.eye(4))
        np.testing.assert_allclose(minimum, [0.9, -0.4, -0.9])
        np.testing.assert_allclose(maximum, [1.1, 0.4, 0.9])
        self.assertIn(str(package / "meshes/shape.obj"), shape.source)

    def test_name_based_fk_uses_joint_names_and_world_pose(self) -> None:
        path = self.write(
            "fk.urdf",
            textwrap.dedent(
                """\
                <robot name="fk_fixture">
                  <link name="root"><collision><geometry><sphere radius="0.1"/></geometry></collision></link>
                  <link name="tip"><collision><geometry><sphere radius="0.1"/></geometry></collision></link>
                  <joint name="named_hinge" type="revolute">
                    <parent link="root"/><child link="tip"/>
                    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
                    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
                  </joint>
                </robot>
                """
            ),
        )
        poses = named_link_fk(
            path,
            transform(10.0, 0.0, 0.0),
            {"named_hinge": math.pi / 2.0},
        )
        self.assertEqual(set(poses), {"root", "tip"})
        np.testing.assert_allclose(poses["tip"][:3, 3], [11.0, 0.0, 0.0])
        np.testing.assert_allclose(
            poses["tip"][:3, :3],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            atol=1.0e-12,
        )

    def test_missing_or_nonfinite_link_transform_fails_closed(self) -> None:
        checker = self.checker()
        with self.assertRaises(CollisionError) as raised:
            checker.check_frame({"arm": np.eye(4)}, self.object_transforms())
        self.assertEqual(raised.exception.code, "LINK_TRANSFORM_MISSING")
        self.assertEqual(raised.exception.context["missing_links"], ["finger"])

        invalid = self.robot_transforms()
        invalid["finger"] = invalid["finger"].copy()
        invalid["finger"][0, 3] = math.nan
        with self.assertRaises(CollisionError) as raised:
            checker.check_frame(invalid, self.object_transforms())
        self.assertEqual(raised.exception.code, "TRANSFORM_INVALID")

    def test_missing_geometry_unknown_allowed_link_and_empty_trajectory_fail(self) -> None:
        visual_only = self.write(
            "visual_only.urdf",
            '<robot name="visual"><link name="body"><visual><geometry><box size="1 1 1"/></geometry></visual></link></robot>',
        )
        with self.assertRaises(CollisionError) as raised:
            load_collision_shapes(visual_only, asset="visual")
        self.assertEqual(raised.exception.code, "COLLISION_GEOMETRY_MISSING")

        with self.assertRaises(CollisionError) as raised:
            self.checker(allowed=("typo_link",))
        self.assertEqual(raised.exception.code, "ALLOWED_LINK_NOT_FOUND")

        with self.assertRaises(CollisionError) as raised:
            NamedAABBCollisionChecker(
                self.robot_path,
                self.object_path,
                broad_phase="ignored_algorithm",
                margin_m=0.0,
                allowed_contact_links=(),
            )
        self.assertEqual(raised.exception.code, "BROAD_PHASE_UNSUPPORTED")

        with self.assertRaises(CollisionError) as raised:
            self.checker().check_trajectory([], [])
        self.assertEqual(raised.exception.code, "TRAJECTORY_EMPTY")

        with self.assertRaises(CollisionError) as raised:
            self.checker().check_trajectory(
                [self.robot_transforms()],
                [self.object_transforms(), self.object_transforms()],
            )
        self.assertEqual(raised.exception.code, "TRAJECTORY_LENGTH_MISMATCH")


if __name__ == "__main__":
    unittest.main()
