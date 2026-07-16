from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from src.agent_cli import (
    CONFIG_NAME,
    GENERATED_AFFORDANCES_NAME,
    MANIFEST_NAME,
    execute_workspace,
    prepare_workspace,
    resolve_source,
    run_agent,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_URDF = ROOT / "assets" / "microwave" / "microwave.urdf"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AgentCliTests(unittest.TestCase):
    def _isolated_fixture(self, root: Path) -> Path:
        asset = root / "asset"
        (asset / "meshes").mkdir(parents=True)
        shutil.copyfile(FIXTURE_URDF, asset / "model.urdf")
        shutil.copyfile(
            ROOT / "assets" / "microwave" / "meshes" / "unit_cube.obj",
            asset / "meshes" / "unit_cube.obj",
        )
        return asset / "model.urdf"

    @staticmethod
    def _provider(*, same_link: bool = False):
        def infer(path: Path):
            del path
            handle_link = "microwave_door" if same_link else "handle"
            return {
                "decisions": {
                    "door_joint": {
                        "value": "door_hinge",
                        "source": "test_provider",
                        "confidence": 0.97,
                        "rationale": "test semantic result",
                    },
                    "door_link": {
                        "value": "microwave_door",
                        "source": "test_provider",
                        "confidence": 0.98,
                    },
                    "handle_link": {
                        "value": handle_link,
                        "source": "test_provider",
                        "confidence": 0.96,
                    },
                },
                "affordance_payload": {
                    "schema_version": 1,
                    "asset_name": "test_microwave",
                    "quaternion_order": "wxyz",
                    "frames": {
                        "inferred_pull_grip": {
                            "link": handle_link,
                            "position": [0.43, -0.0525, 0.0] if same_link else [0.0, 0.0, 0.0],
                            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                            "gripper_closing_axis": [1.0, 0.0, 0.0],
                            "approach_axis": [0.0, 1.0, 0.0],
                            "recommended_gripper_width_m": 0.044,
                        }
                    },
                },
                "semantic_evidence": [{"kind": "unit_test"}],
                "warnings": [],
            }

        return infer

    def test_resolve_articraft_metadata_with_relative_materialized_urdf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            urdf = self._isolated_fixture(root)
            metadata = root / "source.json"
            metadata.write_text(
                json.dumps(
                    {
                        "record_id": "rec_test",
                        "materialized_urdf": str(urdf.relative_to(root)),
                    }
                ),
                encoding="utf-8",
            )
            resolved = resolve_source(articraft_record=str(metadata))
            self.assertEqual(resolved["kind"], "articraft_record")
            self.assertEqual(resolved["record_id"], "rec_test")
            self.assertEqual(resolved["urdf"], urdf.resolve())
            self.assertEqual(resolved["urdf_sha256"], _sha256(urdf))

    def test_prepare_writes_frozen_config_policy_and_decision_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            urdf = self._isolated_fixture(root)
            workspace = root / "job"
            outcome = prepare_workspace(
                workspace=workspace,
                urdf=urdf,
                semantic_provider=self._provider(),
            )
            self.assertTrue(outcome.success, outcome.failure)
            config = json.loads((workspace / CONFIG_NAME).read_text(encoding="utf-8"))
            manifest = json.loads((workspace / MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "prepared")
            self.assertTrue(manifest["readiness"]["ready_for_execution"])
            self.assertFalse(manifest["semantic_inference"]["review_required"])
            self.assertEqual(manifest["semantic_inference"]["low_confidence_decisions"], [])
            self.assertEqual(manifest["input"]["urdf_sha256"], _sha256(urdf))
            self.assertEqual(
                manifest["policy"]["sha256"],
                _sha256(ROOT / "configs" / "upper_agent_policy.json"),
            )
            self.assertEqual(manifest["policy"]["search"]["max_kinematic_attempts"], 7)
            for name in ("door_joint", "door_link", "handle_link", "handle_frame"):
                decision = manifest["semantic_inference"]["decisions"][name]
                self.assertIn("source", decision)
                self.assertIn("confidence", decision)
                self.assertIn("override", decision)
            self.assertEqual(config["assets"]["object"]["urdf"], str(urdf.resolve()))
            self.assertEqual(config["assets"]["object"]["door_link"], "microwave_door")
            self.assertEqual(config["assets"]["object"]["handle_frame"], "inferred_pull_grip")
            self.assertEqual(
                config["assets"]["object"]["world_pose"]["position"],
                [0.39999999999999997, 0.0, 0.5],
            )
            self.assertNotIn("door", config["collision"]["allowed_contact_links"])
            self.assertIn("microwave_door", config["collision"]["allowed_contact_links"])

    def test_prepare_derives_workspace_local_proxy_inertials_without_mutating_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            urdf = self._isolated_fixture(root)
            tree = ET.parse(urdf)
            for link in tree.getroot().findall("link"):
                inertial = link.find("inertial")
                if inertial is not None:
                    link.remove(inertial)
            tree.write(urdf, encoding="unicode")
            source_bytes = urdf.read_bytes()
            source_hash = _sha256(urdf)
            workspace = root / "proxy-job"

            outcome = prepare_workspace(
                workspace=workspace,
                urdf=urdf,
                semantic_provider=self._provider(),
            )

            self.assertTrue(outcome.success, outcome.failure)
            self.assertEqual(urdf.read_bytes(), source_bytes)
            config = json.loads((workspace / CONFIG_NAME).read_text(encoding="utf-8"))
            runtime_urdf = Path(config["assets"]["object"]["urdf"])
            self.assertNotEqual(runtime_urdf, urdf.resolve())
            self.assertTrue(runtime_urdf.is_file())
            runtime_root = ET.parse(runtime_urdf).getroot()
            self.assertTrue(runtime_root.findall("link"))
            self.assertTrue(
                all(link.find("inertial") is not None for link in runtime_root.findall("link"))
            )
            manifest = json.loads((workspace / MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(manifest["input"]["source_urdf"], str(urdf.resolve()))
            self.assertEqual(manifest["input"]["source_urdf_sha256"], source_hash)
            self.assertNotEqual(manifest["input"]["asset_preparation"]["method"], "none")
            self.assertEqual(
                manifest["artifacts"]["runtime_urdf"]["path"], str(runtime_urdf)
            )

    def test_same_link_handle_uses_provider_frame_instead_of_door_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            urdf = self._isolated_fixture(root)
            workspace = root / "same-link"
            outcome = prepare_workspace(
                workspace=workspace,
                urdf=urdf,
                semantic_provider=self._provider(same_link=True),
            )
            self.assertTrue(outcome.success, outcome.failure)
            sidecar = json.loads(
                (workspace / GENERATED_AFFORDANCES_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(list(sidecar["frames"]), ["inferred_pull_grip"])
            self.assertEqual(sidecar["frames"]["inferred_pull_grip"]["link"], "microwave_door")
            config = json.loads((workspace / CONFIG_NAME).read_text(encoding="utf-8"))
            self.assertEqual(config["assets"]["object"]["handle_link"], "microwave_door")
            self.assertEqual(config["assets"]["object"]["handle_frame"], "inferred_pull_grip")

    def test_same_link_without_frame_is_a_structured_failure(self) -> None:
        def incomplete_provider(path: Path):
            del path
            return {
                "door_joint": "door_hinge",
                "door_link": "microwave_door",
                "handle_link": "microwave_door",
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outcome = prepare_workspace(
                workspace=root / "job",
                urdf=self._isolated_fixture(root),
                semantic_provider=incomplete_provider,
            )
            self.assertFalse(outcome.success)
            self.assertEqual(outcome.failure["code"], "handle_frame_not_inferred")
            manifest = json.loads((root / "job" / MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(manifest["failure"]["stage"], "semantic_inference")

    def test_explicit_overrides_and_robot_urdf_are_hashed_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            urdf = self._isolated_fixture(root)
            workspace = root / "job"
            outcome = prepare_workspace(
                workspace=workspace,
                urdf=urdf,
                semantic_provider=self._provider(),
                door_joint="door_hinge",
                robot_urdf=urdf,
                goal_angle_deg=60.0,
            )
            self.assertTrue(outcome.success, outcome.failure)
            config = json.loads((workspace / CONFIG_NAME).read_text(encoding="utf-8"))
            manifest = json.loads((workspace / MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(config["assets"]["robot"]["urdf"], str(urdf.resolve()))
            self.assertEqual(config["assets"]["robot"]["expected_urdf_sha256"], _sha256(urdf))
            self.assertEqual(config["task"]["goal_angle_deg"], 60.0)
            self.assertTrue(
                manifest["semantic_inference"]["decisions"]["door_joint"]["override"]["applied"]
            )
            self.assertTrue(
                manifest["semantic_inference"]["decisions"]["goal_angle_deg"]["override"]["applied"]
            )

    def test_execute_updates_same_manifest_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "job"
            prepared = prepare_workspace(
                workspace=workspace,
                urdf=self._isolated_fixture(root),
                semantic_provider=self._provider(),
            )
            self.assertTrue(prepared.success, prepared.failure)

            def runner(argv):
                output = Path(argv[argv.index("--output-dir") + 1])
                output.mkdir(parents=True)
                (output / "metrics.json").write_text(
                    json.dumps({"success": True, "run_status": "passed"}),
                    encoding="utf-8",
                )
                tcp = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
                handle = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
                handle[:, 0, 3] = [0.2, 0.1]
                np.savez_compressed(
                    output / "trajectory.npz",
                    achieved_gripper_world=tcp,
                    handle_world=handle,
                    door_angle_rad=np.asarray([0.0, 0.5]),
                )
                return 0

            outcome = execute_workspace(workspace=workspace, mode="kinematic", runner=runner)
            self.assertTrue(outcome.success, outcome.failure)
            manifest = json.loads((workspace / MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "succeeded")
            self.assertEqual(len(manifest["executions"]), 1)
            self.assertEqual(manifest["executions"][0]["mode"], "kinematic")
            self.assertTrue(manifest["executions"][0]["success"])
            self.assertEqual(manifest["executions"][0]["animation"]["status"], "succeeded")
            self.assertTrue((workspace / "runs" / "kinematic" / "animation.html").is_file())

    def test_execute_preserves_structured_low_level_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "job"
            prepared = prepare_workspace(
                workspace=workspace,
                urdf=self._isolated_fixture(root),
                semantic_provider=self._provider(),
            )
            self.assertTrue(prepared.success, prepared.failure)

            failure = {
                "code": "ik_unreachable",
                "stage": "ik",
                "message": "no candidate reached",
                "details": {"candidate_count": 3},
            }

            def runner(argv):
                output = Path(argv[argv.index("--output-dir") + 1])
                output.mkdir(parents=True)
                (output / "metrics.json").write_text(
                    json.dumps({"success": False, "failure": failure}),
                    encoding="utf-8",
                )
                return 3

            outcome = execute_workspace(workspace=workspace, mode="kinematic", runner=runner)
            self.assertFalse(outcome.success)
            self.assertEqual(outcome.exit_code, 3)
            self.assertEqual(outcome.failure, failure)
            manifest = json.loads((workspace / MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(manifest["failure"]["code"], "ik_unreachable")

    def test_execute_rejects_changed_config_before_calling_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "job"
            prepared = prepare_workspace(
                workspace=workspace,
                urdf=self._isolated_fixture(root),
                semantic_provider=self._provider(),
            )
            self.assertTrue(prepared.success, prepared.failure)
            (workspace / CONFIG_NAME).write_text("{}\n", encoding="utf-8")
            called = False

            def runner(argv):
                nonlocal called
                called = True
                return 0

            outcome = execute_workspace(workspace=workspace, mode="kinematic", runner=runner)
            self.assertFalse(outcome.success)
            self.assertFalse(called)
            self.assertEqual(outcome.failure["code"], "prepared_config_changed")

    def test_run_searches_offsets_until_first_kinematic_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "job"
            calls = []

            def runner(argv):
                mode = argv[argv.index("--mode") + 1]
                config_path = Path(argv[argv.index("--config") + 1])
                output = Path(argv[argv.index("--output-dir") + 1])
                output.mkdir(parents=True)
                calls.append((mode, config_path))
                success = len(calls) == 3
                payload = {"success": success}
                if not success:
                    payload["failure"] = {
                        "code": "ik_unreachable",
                        "stage": "ik",
                        "message": "synthetic search miss",
                        "details": {"attempt": len(calls)},
                    }
                (output / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
                return 0 if success else 2

            outcome = run_agent(
                workspace=workspace,
                urdf=self._isolated_fixture(root),
                semantic_provider=self._provider(),
                runner=runner,
            )
            self.assertTrue(outcome.success, outcome.failure)
            self.assertEqual(len(calls), 3)
            manifest = json.loads((workspace / MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(manifest["search"]["selected_attempt"], "kinematic_002")
            self.assertEqual(len(manifest["search"]["attempts"]), 3)
            self.assertEqual(
                manifest["search"]["attempts"][2]["object_offset_m"],
                [0.05, 0.0, 0.0],
            )
            selected_path = Path(manifest["artifacts"]["selected_config"]["path"])
            selected = json.loads(selected_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(
                selected["assets"]["object"]["world_pose"]["position"][0],
                0.45,
            )

    def test_run_invokes_physics_only_after_kinematic_success_with_same_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "job"
            calls = []

            def runner(argv):
                mode = argv[argv.index("--mode") + 1]
                config_path = Path(argv[argv.index("--config") + 1])
                output = Path(argv[argv.index("--output-dir") + 1])
                output.mkdir(parents=True)
                calls.append((mode, config_path))
                (output / "metrics.json").write_text(
                    json.dumps({"success": True, "mode": mode}), encoding="utf-8"
                )
                return 0

            outcome = run_agent(
                workspace=workspace,
                urdf=self._isolated_fixture(root),
                semantic_provider=self._provider(),
                with_physics=True,
                runner=runner,
            )
            self.assertTrue(outcome.success, outcome.failure)
            self.assertEqual([mode for mode, _ in calls], ["kinematic", "physics_assisted"])
            self.assertEqual(calls[0][1], calls[1][1])
            manifest = json.loads((workspace / MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertTrue(manifest["search"]["physics"]["success"])
            self.assertEqual(
                manifest["search"]["physics"]["config"]["sha256"],
                manifest["artifacts"]["selected_config"]["sha256"],
            )

    def test_run_stops_at_policy_max_attempts_and_reports_structured_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "job"
            policy = json.loads(
                (ROOT / "configs" / "upper_agent_policy.json").read_text(encoding="utf-8")
            )
            policy["base_config"] = str(
                (ROOT / "configs" / "articraft_microwave_franka.json").resolve()
            )
            policy["search"]["max_kinematic_attempts"] = 2
            policy_path = root / "policy.json"
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            calls = 0

            def runner(argv):
                nonlocal calls
                calls += 1
                output = Path(argv[argv.index("--output-dir") + 1])
                output.mkdir(parents=True)
                (output / "metrics.json").write_text(
                    json.dumps(
                        {
                            "success": False,
                            "failure": {
                                "code": "ik_unreachable",
                                "stage": "ik",
                                "message": "miss",
                                "details": {},
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                return 2

            outcome = run_agent(
                workspace=workspace,
                urdf=self._isolated_fixture(root),
                policy=policy_path,
                semantic_provider=self._provider(),
                with_physics=True,
                runner=runner,
            )
            self.assertFalse(outcome.success)
            self.assertEqual(calls, 2)
            self.assertEqual(outcome.failure["code"], "kinematic_search_exhausted")
            manifest = json.loads((workspace / MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["search"]["attempts"]), 2)
            self.assertIsNone(manifest["search"]["physics"])

    def test_run_does_not_retry_an_infrastructure_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "job"
            calls = 0

            def runner(argv):
                nonlocal calls
                calls += 1
                output = Path(argv[argv.index("--output-dir") + 1])
                output.mkdir(parents=True)
                (output / "metrics.json").write_text(
                    json.dumps(
                        {
                            "success": False,
                            "failure": {
                                "code": "unexpected_error",
                                "stage": "runtime",
                                "message": "compiler unavailable",
                                "details": {},
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                return 2

            outcome = run_agent(
                workspace=workspace,
                urdf=self._isolated_fixture(root),
                semantic_provider=self._provider(),
                runner=runner,
            )
            self.assertFalse(outcome.success)
            self.assertEqual(calls, 1)
            self.assertEqual(outcome.failure["code"], "unexpected_error")
            manifest = json.loads((workspace / MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertTrue(manifest["search"]["aborted_nonretryable"])


if __name__ == "__main__":
    unittest.main()
