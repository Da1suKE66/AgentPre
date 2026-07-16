from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from src.proxy_inertials import ProxyInertialError, prepare_proxy_inertials


class ProxyInertialTests(unittest.TestCase):
    def test_derives_collision_aabb_proxy_and_preserves_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.urdf"
            source.write_text(
                """<robot name="asset">
  <link name="door">
    <visual><origin xyz="10 0 0"/><geometry><box size="8 8 8"/></geometry></visual>
    <collision><origin xyz="1 2 3"/><geometry><box size="0.4 0.2 0.1"/></geometry></collision>
  </link>
</robot>""",
                encoding="utf-8",
            )
            source_bytes = source.read_bytes()
            result = prepare_proxy_inertials(source, root / "job" / "prepared.urdf")

            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertEqual(result.generated_links, ("door",))
            self.assertEqual(result.preserved_links, ())
            self.assertTrue(result.output_urdf.is_file())
            link = ET.parse(result.output_urdf).getroot().find("link")
            self.assertEqual(link.find("inertial/origin").get("xyz"), "1 2 3")
            mass = float(link.find("inertial/mass").get("value"))
            self.assertAlmostEqual(mass, 2.4)
            inertia = link.find("inertial/inertia")
            self.assertAlmostEqual(float(inertia.get("ixx")), mass * (0.2**2 + 0.1**2) / 12)
            self.assertEqual(result.links["door"]["geometry_sources"], ["collision[0]:box"])
            self.assertEqual(result.to_dict()["approximation"], True)

    def test_preserves_existing_inertial_and_rewrites_relative_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mesh = root / "mesh.obj"
            mesh.write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 2 4\n",
                encoding="utf-8",
            )
            source = root / "source.urdf"
            source.write_text(
                """<robot name="asset">
  <link name="base">
    <inertial><origin xyz="0 0 0"/><mass value="7"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
    <collision><geometry><mesh filename="mesh.obj"/></geometry></collision>
  </link>
</robot>""",
                encoding="utf-8",
            )
            result = prepare_proxy_inertials(source, root / "out" / "model.urdf")
            self.assertEqual(result.generated_links, ())
            self.assertEqual(result.preserved_links, ("base",))
            document = ET.parse(result.output_urdf).getroot()
            self.assertEqual(document.find("link/inertial/mass").get("value"), "7")
            self.assertEqual(
                document.find("link/collision/geometry/mesh").get("filename"),
                str(mesh.resolve()),
            )
            self.assertEqual(result.rewritten_mesh_references, 1)

    def test_applies_positive_floor_to_degenerate_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "thin.urdf"
            source.write_text(
                '<robot name="thin"><link name="part"><collision><geometry>'
                '<box size="0.001 0.001 0.001"/>'
                '</geometry></collision></link></robot>',
                encoding="utf-8",
            )
            result = prepare_proxy_inertials(source, root / "prepared.urdf")
            mass = float(
                ET.parse(result.output_urdf).getroot().find("link/inertial/mass").get("value")
            )
            self.assertEqual(mass, 0.02)
            inertia = ET.parse(result.output_urdf).getroot().find("link/inertial/inertia")
            self.assertTrue(all(float(inertia.get(name)) > 0.0 for name in ("ixx", "iyy", "izz")))

    def test_rejects_link_without_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "empty.urdf"
            source.write_text(
                '<robot name="empty"><link name="part"/></robot>', encoding="utf-8"
            )
            with self.assertRaises(ProxyInertialError) as raised:
                prepare_proxy_inertials(source, root / "prepared.urdf")
            self.assertEqual(raised.exception.code, "GEOMETRY_UNAVAILABLE")
            self.assertFalse((root / "prepared.urdf").exists())

    def test_rejects_source_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "model.urdf"
            source.write_text(
                '<robot name="a"><link name="x"><collision><geometry>'
                '<sphere radius="1"/></geometry></collision></link></robot>',
                encoding="utf-8",
            )
            with self.assertRaises(ProxyInertialError) as raised:
                prepare_proxy_inertials(source, source)
            self.assertEqual(raised.exception.code, "SOURCE_OVERWRITE_FORBIDDEN")


if __name__ == "__main__":
    unittest.main()
