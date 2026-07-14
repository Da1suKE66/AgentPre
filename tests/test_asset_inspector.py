from __future__ import annotations

import json
from pathlib import Path
import tempfile
import textwrap
import unittest

import numpy as np

from src.asset_inspector import inspect_asset
from src.urdf_model import URDFModelError, load_urdf, resolve_mesh_path


def valid_urdf(mesh_filename: str = "meshes/body.obj") -> str:
    return textwrap.dedent(
        f"""\
        <?xml version="1.0"?>
        <robot name="test_microwave">
          <link name="base">
            <inertial>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <mass value="1.0"/>
              <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
            </inertial>
            <visual><geometry><mesh filename="{mesh_filename}" scale="1 1 1"/></geometry></visual>
          </link>
          <link name="door">
            <inertial>
              <mass value="0.5"/>
              <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
            </inertial>
          </link>
          <link name="handle">
            <inertial>
              <mass value="0.1"/>
              <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
            </inertial>
          </link>
          <joint name="door_hinge" type="revolute">
            <parent link="base"/>
            <child link="door"/>
            <origin xyz="0.2 -0.3 0.1" rpy="0 0 0"/>
            <axis xyz="0 0 1"/>
            <limit lower="0" upper="1.57" effort="10" velocity="1"/>
          </joint>
          <joint name="handle_mount" type="fixed">
            <parent link="door"/>
            <child link="handle"/>
            <origin xyz="0 0.4 0" rpy="0 0 0"/>
          </joint>
        </robot>
        """
    )


class AssetInspectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "meshes").mkdir()
        (self.root / "meshes" / "body.obj").write_text("o body\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_urdf(self, contents: str) -> Path:
        path = self.root / "microwave.urdf"
        path.write_text(contents, encoding="utf-8")
        return path

    def inspect(self, path: Path, **kwargs):
        return inspect_asset(
            path,
            door_joint_name=kwargs.pop("door_joint_name", "door_hinge"),
            door_link_name=kwargs.pop("door_link_name", "door"),
            handle_link_name=kwargs.pop("handle_link_name", "handle"),
            **kwargs,
        )

    def test_parser_exposes_links_joints_and_meshes_by_name(self) -> None:
        path = self.write_urdf(valid_urdf())
        model = load_urdf(path)
        self.assertEqual(model.link_names, ("base", "door", "handle"))
        self.assertEqual(model.joint_names, ("door_hinge", "handle_mount"))
        joint = model.require_joint("door_hinge")
        self.assertEqual(joint.joint_type, "revolute")
        self.assertEqual(joint.axis, (0.0, 0.0, 1.0))
        self.assertEqual(joint.origin.xyz, (0.2, -0.3, 0.1))
        self.assertEqual(joint.limit.lower, 0.0)
        self.assertEqual(joint.limit.upper, 1.57)
        mesh = model.require_link("base").meshes[0]
        self.assertEqual(
            resolve_mesh_path(mesh, model.path),
            (self.root / "meshes" / "body.obj").resolve(),
        )
        with self.assertRaises(URDFModelError) as raised:
            model.require_joint("not_an_index_or_name")
        self.assertEqual(raised.exception.code, "JOINT_NOT_FOUND")

    def test_valid_asset_has_serializable_report(self) -> None:
        report = self.inspect(self.write_urdf(valid_urdf()))
        self.assertTrue(report.ok, [issue.to_dict() for issue in report.errors])
        self.assertEqual(report.door_joint["name"], "door_hinge")
        self.assertEqual(report.door_joint["axis"], [0.0, 0.0, 1.0])
        self.assertEqual(report.door_joint["limit"]["upper"], 1.57)
        self.assertEqual(len(report.meshes), 1)
        self.assertTrue(report.meshes[0].exists)
        json.dumps(report.to_dict())

    def test_door_link_may_also_be_the_handle_frame_parent(self) -> None:
        report = self.inspect(
            self.write_urdf(valid_urdf()),
            handle_link_name="door",
        )

        self.assertTrue(report.ok, [issue.to_dict() for issue in report.errors])
        self.assertNotIn(
            "HANDLE_NOT_ATTACHED_TO_DOOR",
            {issue.code for issue in report.errors},
        )

    def test_configured_names_are_checked_without_indices(self) -> None:
        path = self.write_urdf(valid_urdf())
        report = self.inspect(
            path,
            door_joint_name="missing_joint",
            door_link_name="missing_door",
            handle_link_name="missing_handle",
        )
        codes = {issue.code for issue in report.errors}
        self.assertEqual(
            codes,
            {"DOOR_JOINT_NOT_FOUND", "DOOR_LINK_NOT_FOUND", "HANDLE_LINK_NOT_FOUND"},
        )

    def test_invalid_door_type_and_axis_are_structured_errors(self) -> None:
        path = self.write_urdf(valid_urdf().replace('type="revolute"', 'type="fixed"', 1))
        report = self.inspect(path)
        self.assertIn("DOOR_JOINT_TYPE_INVALID", {issue.code for issue in report.errors})

        path = self.write_urdf(valid_urdf().replace('type="revolute"', 'type="continuous"', 1))
        report = self.inspect(path)
        self.assertIn("DOOR_JOINT_TYPE_INVALID", {issue.code for issue in report.errors})

        path = self.write_urdf(valid_urdf().replace('axis xyz="0 0 1"', 'axis xyz="0 0 0"'))
        report = self.inspect(path)
        self.assertIn("JOINT_AXIS_ZERO", {issue.code for issue in report.errors})

    def test_missing_mesh_file_is_reported(self) -> None:
        path = self.write_urdf(valid_urdf("meshes/not_here.obj"))
        report = self.inspect(path)
        self.assertFalse(report.ok)
        issue = next(issue for issue in report.errors if issue.code == "MESH_FILE_NOT_FOUND")
        self.assertEqual(issue.context["link"], "base")
        self.assertTrue(str(issue.context["resolved_path"]).endswith("meshes/not_here.obj"))

    def test_package_mesh_can_be_resolved_explicitly(self) -> None:
        package_root = self.root / "microwave_description"
        (package_root / "meshes").mkdir(parents=True)
        (package_root / "meshes" / "body.obj").write_text("o body\n", encoding="utf-8")
        path = self.write_urdf(
            valid_urdf("package://microwave_description/meshes/body.obj")
        )
        unresolved = self.inspect(path)
        self.assertIn("MESH_URI_UNRESOLVED", {issue.code for issue in unresolved.errors})
        resolved = self.inspect(path, package_paths={"microwave_description": package_root})
        self.assertTrue(resolved.ok, [issue.to_dict() for issue in resolved.errors])

    def test_negative_mass_and_nonphysical_inertia_are_rejected(self) -> None:
        contents = valid_urdf().replace('mass value="1.0"', 'mass value="-1.0"', 1)
        contents = contents.replace('ixx="0.1"', 'ixx="-0.1"', 1)
        report = self.inspect(self.write_urdf(contents))
        codes = {issue.code for issue in report.errors}
        self.assertIn("MASS_NEGATIVE", codes)
        self.assertIn("INERTIA_DIAGONAL_NEGATIVE", codes)
        self.assertIn("INERTIA_NOT_POSITIVE_SEMIDEFINITE", codes)

    def test_missing_inertial_is_rejected_because_it_cannot_be_verified(self) -> None:
        contents = valid_urdf().replace(
            """<link name="handle">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>""",
            """<link name="handle"/>""",
        )
        report = self.inspect(self.write_urdf(contents))
        self.assertFalse(report.ok)
        issue = next(
            issue for issue in report.errors if issue.code == "LINK_INERTIAL_MISSING"
        )
        self.assertEqual(issue.context["link"], "handle")

    def test_malformed_numeric_fields_become_structured_parse_errors(self) -> None:
        path = self.write_urdf(valid_urdf().replace('axis xyz="0 0 1"', 'axis xyz="0 0"'))
        report = self.inspect(path)
        self.assertFalse(report.ok)
        self.assertEqual(report.errors[0].code, "INVALID_VECTOR")
        self.assertEqual(report.errors[0].context["joint"], "door_hinge")

        path = self.write_urdf(valid_urdf().replace('mass value="1.0"', 'mass value="nan"', 1))
        report = self.inspect(path)
        self.assertEqual(report.errors[0].code, "NONFINITE_NUMBER")

    def test_handle_must_be_attached_below_door(self) -> None:
        contents = valid_urdf().replace(
            '<parent link="door"/>\n    <child link="handle"/>',
            '<parent link="base"/>\n    <child link="handle"/>',
        )
        report = self.inspect(self.write_urdf(contents))
        self.assertIn("HANDLE_NOT_ATTACHED_TO_DOOR", {issue.code for issue in report.errors})


if __name__ == "__main__":
    unittest.main()
