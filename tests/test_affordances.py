from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import textwrap
import unittest

import numpy as np

from src.affordances import (
    AffordanceError,
    CandidateGenerationConfig,
    CheckResult,
    GraspCandidate,
    extract_handle_geometry,
    load_affordances,
    resolve_handle_candidates,
    select_grasp_candidate,
)
from src.transforms import quaternion_to_matrix


def affordances_json(*, frames: dict | None = None, order: str = "wxyz") -> dict:
    return {
        "schema_version": 1,
        "quaternion_order": order,
        "frames": frames
        if frames is not None
        else {
            "handle_grasp": {
                "link": "handle",
                "position": [0.1, -0.2, 0.3],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "gripper_closing_axis": [1.0, 0.0, 0.0],
                "approach_axis": [0.0, -1.0, 0.0],
                "recommended_gripper_width_m": 0.04,
            }
        },
    }


def box_urdf(
    *,
    geometry: str = '<box size="0.02 0.04 0.20"/>',
    origin: str = '<origin xyz="1 2 3" rpy="0 0 0"/>',
) -> str:
    return textwrap.dedent(
        f"""\
        <?xml version="1.0"?>
        <robot name="fixture">
          <link name="base"/>
          <link name="handle">
            <collision>
              {origin}
              <geometry>{geometry}</geometry>
            </collision>
          </link>
          <joint name="handle_mount" type="fixed">
            <parent link="base"/>
            <child link="handle"/>
          </joint>
        </robot>
        """
    )


class AffordanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_json(self, data: dict) -> Path:
        path = self.root / "affordances.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def write_urdf(self, contents: str) -> Path:
        path = self.root / "fixture.urdf"
        path.write_text(contents, encoding="utf-8")
        return path

    def generation_config(
        self,
        *,
        width_margin_m: float = 0.005,
        max_gripper_width_m: float = 0.03,
        max_candidates: int = 3,
        primitive_radial_samples: int = 12,
    ) -> CandidateGenerationConfig:
        return CandidateGenerationConfig(
            width_margin_m=width_margin_m,
            max_gripper_width_m=max_gripper_width_m,
            max_candidates=max_candidates,
            primitive_radial_samples=primitive_radial_samples,
        )

    def test_loads_valid_wxyz_handle_frame(self) -> None:
        data = load_affordances(self.write_json(affordances_json()))
        frame = data.require_frame("handle_grasp")
        self.assertEqual(frame.link_name, "handle")
        self.assertEqual(frame.position, (0.1, -0.2, 0.3))
        self.assertEqual(frame.quaternion_wxyz, (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(frame.gripper_closing_axis, (1.0, 0.0, 0.0))
        self.assertEqual(frame.approach_axis, (0.0, -1.0, 0.0))
        self.assertEqual(frame.recommended_gripper_width_m, 0.04)
        np.testing.assert_allclose(frame.transform[:3, 3], frame.position)
        json.dumps(frame.to_dict())

    def test_invalid_order_quaternion_axes_and_width_are_rejected(self) -> None:
        with self.assertRaises(AffordanceError) as raised:
            load_affordances(self.write_json(affordances_json(order="xyzw")))
        self.assertEqual(raised.exception.code, "QUATERNION_ORDER_INVALID")

        mutations = (
            ("quaternion_wxyz", [2.0, 0.0, 0.0, 0.0], "QUATERNION_NOT_NORMALIZED"),
            ("gripper_closing_axis", [0.0, 0.0, 0.0], "AXIS_NOT_NORMALIZED"),
            ("approach_axis", [1.0, 0.0, 0.0], "FRAME_AXES_NOT_ORTHOGONAL"),
            ("recommended_gripper_width_m", 0.0, "GRIPPER_WIDTH_INVALID"),
        )
        for key, value, code in mutations:
            with self.subTest(key=key):
                raw = affordances_json()
                raw["frames"]["handle_grasp"][key] = value
                with self.assertRaises(AffordanceError) as raised:
                    load_affordances(self.write_json(raw))
                self.assertEqual(raised.exception.code, code)

    def test_authored_frame_is_preferred_over_geometry(self) -> None:
        path = self.write_json(affordances_json())
        urdf_path = self.write_urdf(box_urdf())
        resolution = resolve_handle_candidates(
            path,
            "handle_grasp",
            urdf_path,
            "handle",
            config=self.generation_config(max_gripper_width_m=0.08),
        )
        self.assertFalse(resolution.used_geometry_fallback)
        self.assertIsNone(resolution.geometry)
        self.assertEqual(len(resolution.candidates), 1)
        self.assertEqual(resolution.candidates[0].candidate_id, "frame:handle_grasp")
        self.assertEqual(resolution.candidates[0].gripper_width_m, 0.04)

    def test_missing_frame_falls_back_to_box_aabb_and_pca(self) -> None:
        affordance_path = self.write_json(affordances_json(frames={}))
        urdf_path = self.write_urdf(box_urdf())
        resolution = resolve_handle_candidates(
            affordance_path,
            "missing_frame",
            urdf_path,
            "handle",
            config=self.generation_config(),
        )
        self.assertTrue(resolution.used_geometry_fallback)
        self.assertEqual(resolution.reasons[0].code, "FRAME_NOT_FOUND")
        self.assertEqual(len(resolution.candidates), 3)
        assert resolution.geometry is not None
        np.testing.assert_allclose(resolution.geometry.aabb_min, [0.99, 1.98, 2.9])
        np.testing.assert_allclose(resolution.geometry.aabb_max, [1.01, 2.02, 3.1])
        np.testing.assert_allclose(resolution.geometry.aabb_center, [1.0, 2.0, 3.0])
        for candidate in resolution.candidates:
            np.testing.assert_allclose(candidate.position, [1.0, 2.0, 3.0])
            self.assertAlmostEqual(candidate.gripper_width_m, 0.025)
            self.assertLessEqual(candidate.gripper_width_m, 0.03)
            self.assertAlmostEqual(
                float(
                    np.dot(candidate.gripper_closing_axis, candidate.approach_axis)
                ),
                0.0,
                places=12,
            )
            rotation = quaternion_to_matrix(candidate.quaternion_wxyz)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)

    def test_box_cylinder_and_sphere_geometry_apply_local_origins(self) -> None:
        cases = (
            ('<box size="0.2 0.1 0.05"/>', [-0.1, -0.05, -0.025], [0.1, 0.05, 0.025]),
            ('<cylinder radius="0.1" length="0.4"/>', [-0.1, -0.1, -0.2], [0.1, 0.1, 0.2]),
            ('<sphere radius="0.2"/>', [-0.2, -0.2, -0.2], [0.2, 0.2, 0.2]),
        )
        for geometry_xml, expected_min, expected_max in cases:
            with self.subTest(geometry=geometry_xml):
                path = self.write_urdf(
                    box_urdf(
                        geometry=geometry_xml,
                        origin='<origin xyz="0.5 -0.25 1.0" rpy="0 0 0"/>',
                    )
                )
                geometry = extract_handle_geometry(
                    path, "handle", primitive_radial_samples=16
                )
                np.testing.assert_allclose(
                    geometry.aabb_min,
                    np.asarray(expected_min) + [0.5, -0.25, 1.0],
                    atol=1.0e-12,
                )
                np.testing.assert_allclose(
                    geometry.aabb_max,
                    np.asarray(expected_max) + [0.5, -0.25, 1.0],
                    atol=1.0e-12,
                )
                np.testing.assert_allclose(
                    geometry.principal_axes.T @ geometry.principal_axes,
                    np.eye(3),
                    atol=1.0e-12,
                )

    def test_obj_vertices_scale_and_origin_are_supported_without_trimesh(self) -> None:
        mesh_dir = self.root / "meshes"
        mesh_dir.mkdir()
        (mesh_dir / "handle.obj").write_text(
            textwrap.dedent(
                """\
                o handle
                v -1 -2 -3
                v 1 -2 -3
                v -1 2 -3
                v 1 2 3
                """
            ),
            encoding="utf-8",
        )
        urdf = box_urdf(
            geometry='<mesh filename="meshes/handle.obj" scale="0.1 0.2 0.3"/>',
            origin='<origin xyz="1 2 3" rpy="0 0 0"/>',
        )
        geometry = extract_handle_geometry(
            self.write_urdf(urdf), "handle", primitive_radial_samples=8
        )
        np.testing.assert_allclose(geometry.aabb_min, [0.9, 1.6, 2.1])
        np.testing.assert_allclose(geometry.aabb_max, [1.1, 2.4, 3.9])
        np.testing.assert_allclose(geometry.aabb_center, [1.0, 2.0, 3.0])
        self.assertEqual(geometry.sources, ("collision[0]:mesh",))

    def test_width_margin_limit_and_candidate_limit_are_caller_owned(self) -> None:
        path = self.write_urdf(box_urdf(origin='<origin xyz="0 0 0" rpy="0 0 0"/>'))
        affordance_path = self.write_json(affordances_json(frames={}))
        with self.assertRaises(AffordanceError) as raised:
            resolve_handle_candidates(
                affordance_path,
                "missing",
                path,
                "handle",
                config=self.generation_config(
                    width_margin_m=0.02,
                    max_gripper_width_m=0.039,
                    max_candidates=8,
                ),
            )
        self.assertEqual(raised.exception.code, "GRIPPER_WIDTH_EXCEEDED")

        resolution = resolve_handle_candidates(
            affordance_path,
            "missing",
            path,
            "handle",
            config=self.generation_config(
                width_margin_m=0.02,
                max_gripper_width_m=0.041,
                max_candidates=1,
            ),
        )
        self.assertEqual(len(resolution.candidates), 1)
        self.assertAlmostEqual(resolution.candidates[0].gripper_width_m, 0.04)

    def test_selection_is_deterministic_and_retains_rejection_reasons(self) -> None:
        candidates = tuple(
            GraspCandidate(
                candidate_id=f"candidate-{rank}",
                rank=rank,
                link_name="handle",
                position=(0.0, 0.0, 0.0),
                quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                gripper_closing_axis=(1.0, 0.0, 0.0),
                approach_axis=(0.0, 1.0, 0.0),
                gripper_width_m=0.04,
                source="test",
            )
            for rank in (2, 0, 1)
        )
        calls: list[tuple[str, str]] = []

        def reachable(candidate: GraspCandidate):
            calls.append(("reach", candidate.candidate_id))
            if candidate.rank == 0:
                return False, "outside configured reach envelope"
            return True

        def collision_free(candidate: GraspCandidate):
            calls.append(("collision", candidate.candidate_id))
            if candidate.rank == 1:
                return CheckResult(False, "finger intersects door", {"pair": "finger-door"})
            return True

        selection = select_grasp_candidate(
            candidates,
            reachability_check=reachable,
            collision_free_check=collision_free,
        )
        self.assertTrue(selection.ok)
        assert selection.selected is not None
        self.assertEqual(selection.selected.candidate_id, "candidate-2")
        self.assertEqual(
            calls,
            [
                ("reach", "candidate-0"),
                ("reach", "candidate-1"),
                ("collision", "candidate-1"),
                ("reach", "candidate-2"),
                ("collision", "candidate-2"),
            ],
        )
        self.assertEqual(selection.evaluations[0].reasons[0].code, "IK_UNREACHABLE")
        self.assertEqual(
            selection.evaluations[0].reasons[0].message,
            "outside configured reach envelope",
        )
        self.assertEqual(selection.evaluations[1].reasons[0].code, "COLLISION")
        self.assertEqual(
            selection.evaluations[1].reasons[0].context["pair"], "finger-door"
        )
        json.dumps(selection.to_dict())

    def test_failed_callbacks_are_preserved_without_hiding_all_candidate_failures(self) -> None:
        candidate = GraspCandidate(
            candidate_id="only",
            rank=0,
            link_name="handle",
            position=(0.0, 0.0, 0.0),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            gripper_closing_axis=(1.0, 0.0, 0.0),
            approach_axis=(0.0, 1.0, 0.0),
            gripper_width_m=0.04,
            source="test",
        )

        def failed_reachability(_: GraspCandidate):
            raise RuntimeError("solver did not converge")

        selection = select_grasp_candidate(
            [candidate],
            reachability_check=failed_reachability,
            collision_free_check=lambda _: True,
        )
        self.assertFalse(selection.ok)
        self.assertIsNone(selection.selected)
        self.assertEqual(
            selection.failure_reasons[0].code, "REACHABILITY_CHECK_ERROR"
        )
        self.assertIn("RuntimeError", selection.failure_reasons[0].message)

    def test_missing_link_and_missing_geometry_have_structured_errors(self) -> None:
        path = self.write_urdf(box_urdf())
        with self.assertRaises(AffordanceError) as raised:
            extract_handle_geometry(path, "not_handle", primitive_radial_samples=8)
        self.assertEqual(raised.exception.code, "HANDLE_LINK_NOT_FOUND")

        missing = self.write_urdf(
            '<?xml version="1.0"?><robot name="empty"><link name="handle"/></robot>'
        )
        with self.assertRaises(AffordanceError) as raised:
            extract_handle_geometry(missing, "handle", primitive_radial_samples=8)
        self.assertEqual(raised.exception.code, "HANDLE_GEOMETRY_MISSING")


if __name__ == "__main__":
    unittest.main()
