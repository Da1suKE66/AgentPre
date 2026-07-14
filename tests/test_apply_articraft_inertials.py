from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import scripts.apply_articraft_inertials as inertial_module
from scripts.apply_articraft_inertials import (
    InertialSpecificationError,
    SIDECAR_NAME,
    apply_inertials,
    validate_completed_inertials,
)


ROOT = Path(__file__).resolve().parents[1]
RECORD_ID = "rec_microwave_oven_5e86f3429e954dcd9ab6c9d3a94db707"
DATA_COMMIT = "0cdcaa49f5571e9b4df04476c7f09587ee3ab7bd"


def _inertial(mass: float = 2.0) -> dict[str, object]:
    return {
        "mass_kg": mass,
        "origin_xyz_m": [0.1, -0.2, 0.3],
        "origin_rpy_rad": [0.0, 0.0, 0.0],
        "inertia_kg_m2": {
            "ixx": 0.1,
            "ixy": 0.0,
            "ixz": 0.0,
            "iyy": 0.2,
            "iyz": 0.0,
            "izz": 0.3,
        },
    }


def _specification(*link_names: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "state": "ready",
        "source_urdf_sha256": "0" * 64,
        "record": {
            "id": RECORD_ID,
            "revision": "rev_000001",
            "data_commit": DATA_COMMIT,
            "model_url": "https://example.invalid/pinned/model.py",
        },
        "units": {
            "mass": "kg",
            "length": "m",
            "angle": "rad",
            "inertia": "kg*m^2",
        },
        "links": {name: _inertial() for name in link_names},
        "notes": ["dependency-light deterministic test specification"],
    }


def _write_case(root: Path, spec: dict[str, object], *link_names: str) -> tuple[Path, Path]:
    urdf = root / "model.urdf"
    links = "".join(f'<link name="{name}"><visual/></link>' for name in link_names)
    urdf.write_text(f'<robot name="fixture">{links}</robot>\n', encoding="utf-8")
    spec["source_urdf_sha256"] = hashlib.sha256(urdf.read_bytes()).hexdigest()
    spec_path = root / "inertials.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return urdf, spec_path


class ApplyArticraftInertialsTests(unittest.TestCase):
    def test_checked_in_spec_is_ready_for_all_actual_links(self) -> None:
        path = ROOT / "assets" / "articraft" / RECORD_ID / "inertials.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["state"], "ready")
        self.assertEqual(
            payload["source_urdf_sha256"],
            "03f6aa1366ddbf0b740f1e051bfb0b8f673c9cf7bc293ad37f6ac60289beff36",
        )
        self.assertEqual(
            set(payload["links"]),
            {
                "cabinet",
                "door",
                "turntable",
                "selector_knob_0",
                "selector_knob_1",
            },
        )
        for link in payload["links"].values():
            self.assertGreater(link["mass_kg"], 0.0)
            self.assertEqual(len(link["origin_xyz_m"]), 3)
            self.assertEqual(link["origin_rpy_rad"], [0.0, 0.0, 0.0])
            self.assertGreater(link["inertia_kg_m2"]["ixx"], 0.0)
            self.assertGreater(link["inertia_kg_m2"]["iyy"], 0.0)
            self.assertGreater(link["inertia_kg_m2"]["izz"], 0.0)
            for component in ("ixy", "ixz", "iyz"):
                self.assertEqual(link["inertia_kg_m2"][component], 0.0)
        expected = {
            "cabinet": (26.33232, [0.0, -0.0205, 0.17]),
            "door": (2.8864512, [0.244, -0.0285, 0.0]),
            "turntable": (0.4964544, [0.0, 0.0, -0.0015]),
            "selector_knob_0": (0.0738896670888, [0.0, 0.0, 0.01948]),
            "selector_knob_1": (0.0738896670888, [0.0, 0.0, 0.01948]),
        }
        for name, (mass, origin) in expected.items():
            self.assertEqual(payload["links"][name]["mass_kg"], mass)
            self.assertEqual(payload["links"][name]["origin_xyz_m"], origin)

    def test_injects_every_value_writes_sidecar_and_is_byte_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urdf, spec = _write_case(
                root, _specification("cabinet", "door"), "cabinet", "door"
            )
            pre_hash = hashlib.sha256(urdf.read_bytes()).hexdigest()

            first = apply_inertials(urdf, spec)
            sidecar = root / SIDECAR_NAME
            self.assertTrue(first["modified"])
            self.assertTrue(sidecar.is_file())
            document = ET.parse(urdf)
            for name in ("cabinet", "door"):
                link = document.find(f"./link[@name='{name}']")
                self.assertIsNotNone(link)
                inertial = link.find("inertial")
                self.assertEqual(inertial.find("mass").get("value"), "2")
                self.assertEqual(
                    inertial.find("origin").get("xyz"),
                    "0.10000000000000001 -0.20000000000000001 0.29999999999999999",
                )
                self.assertEqual(inertial.find("inertia").get("iyy"), "0.20000000000000001")
            completion = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(completion["urdf"]["pre_sha256"], pre_hash)
            self.assertEqual(
                completion["urdf"]["post_sha256"],
                hashlib.sha256(urdf.read_bytes()).hexdigest(),
            )
            self.assertEqual(completion["injected_links"], ["cabinet", "door"])
            self.assertEqual(
                completion["specification"]["sha256"],
                hashlib.sha256(spec.read_bytes()).hexdigest(),
            )
            urdf_bytes = urdf.read_bytes()
            sidecar_bytes = sidecar.read_bytes()
            urdf_mtime = urdf.stat().st_mtime_ns
            sidecar_mtime = sidecar.stat().st_mtime_ns

            second = apply_inertials(urdf, spec)
            self.assertFalse(second["modified"])
            self.assertEqual(urdf.read_bytes(), urdf_bytes)
            self.assertEqual(sidecar.read_bytes(), sidecar_bytes)
            self.assertEqual(urdf.stat().st_mtime_ns, urdf_mtime)
            self.assertEqual(sidecar.stat().st_mtime_ns, sidecar_mtime)

    def test_interruption_after_sidecar_write_is_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urdf, spec = _write_case(root, _specification("cabinet"), "cabinet")
            pristine = urdf.read_bytes()
            real_write = inertial_module._write_atomic
            calls = 0

            def interrupt_second_write(*args: object, **kwargs: object) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated interruption before URDF replacement")
                real_write(*args, **kwargs)

            with patch.object(
                inertial_module, "_write_atomic", side_effect=interrupt_second_write
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    apply_inertials(urdf, spec)

            sidecar = root / SIDECAR_NAME
            self.assertEqual(urdf.read_bytes(), pristine)
            self.assertTrue(sidecar.is_file())
            recovered = apply_inertials(urdf, spec)
            self.assertTrue(recovered["modified"])
            verified = apply_inertials(urdf, spec)
            self.assertFalse(verified["modified"])

    def test_concurrent_calls_are_serialized_by_transaction_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urdf, spec = _write_case(root, _specification("cabinet"), "cabinet")
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(
                    executor.map(lambda _: apply_inertials(urdf, spec), range(2))
                )
            self.assertEqual(
                sorted(result["modified"] for result in results), [False, True]
            )

    def test_read_only_validator_never_completes_a_pristine_urdf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urdf, spec = _write_case(root, _specification("cabinet"), "cabinet")
            pristine = urdf.read_bytes()
            with self.assertRaisesRegex(
                InertialSpecificationError, "requires every URDF link"
            ):
                validate_completed_inertials(urdf, spec)
            self.assertEqual(urdf.read_bytes(), pristine)
            self.assertFalse((root / SIDECAR_NAME).exists())

    def test_pristine_urdf_hash_must_match_structured_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urdf, spec = _write_case(root, _specification("cabinet"), "cabinet")
            payload = json.loads(spec.read_text(encoding="utf-8"))
            payload["source_urdf_sha256"] = "f" * 64
            spec.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            pristine = urdf.read_bytes()
            with self.assertRaisesRegex(
                InertialSpecificationError, "pristine compiled URDF SHA-256"
            ):
                apply_inertials(urdf, spec)
            self.assertEqual(urdf.read_bytes(), pristine)
            self.assertFalse((root / SIDECAR_NAME).exists())

    def test_todo_spec_fails_before_writing_anything(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = _specification("cabinet")
            payload["state"] = "todo"
            payload["links"]["cabinet"] = {
                "mass_kg": None,
                "origin_xyz_m": [None, None, None],
                "origin_rpy_rad": [None, None, None],
                "inertia_kg_m2": {key: None for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")},
            }
            urdf, spec = _write_case(root, payload, "cabinet")
            before = urdf.read_bytes()
            with self.assertRaisesRegex(InertialSpecificationError, "still TODO"):
                apply_inertials(urdf, spec)
            self.assertEqual(urdf.read_bytes(), before)
            self.assertFalse((root / SIDECAR_NAME).exists())

    def test_unknown_extra_and_partial_link_sets_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urdf, spec = _write_case(
                root, _specification("cabinet", "ghost"), "cabinet", "door"
            )
            before = urdf.read_bytes()
            with self.assertRaisesRegex(InertialSpecificationError, "link set"):
                apply_inertials(urdf, spec)
            self.assertEqual(urdf.read_bytes(), before)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = _specification("cabinet", "door")
            existing = payload["links"]["cabinet"]
            urdf, spec = _write_case(root, payload, "cabinet", "door")
            document = ET.parse(urdf)
            cabinet = document.find("./link[@name='cabinet']")
            inertial = ET.Element("inertial")
            ET.SubElement(inertial, "origin", xyz="0.1 -0.2 0.3", rpy="0 0 0")
            ET.SubElement(inertial, "mass", value=str(existing["mass_kg"]))
            ET.SubElement(
                inertial,
                "inertia",
                {key: str(value) for key, value in existing["inertia_kg_m2"].items()},
            )
            cabinet.insert(0, inertial)
            document.write(urdf, encoding="utf-8", xml_declaration=True)
            before = urdf.read_bytes()
            with self.assertRaisesRegex(InertialSpecificationError, "partially postprocessed"):
                apply_inertials(urdf, spec)
            self.assertEqual(urdf.read_bytes(), before)

    def test_mismatched_existing_inertial_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urdf, spec = _write_case(root, _specification("cabinet"), "cabinet")
            apply_inertials(urdf, spec)
            document = ET.parse(urdf)
            document.find("./link/inertial/mass").set("value", "9")
            document.write(urdf, encoding="utf-8", xml_declaration=True)
            before = urdf.read_bytes()
            with self.assertRaisesRegex(InertialSpecificationError, "does not match"):
                apply_inertials(urdf, spec)
            self.assertEqual(urdf.read_bytes(), before)

    def test_nonfinite_nonpositive_mass_and_nonpositive_definite_inertia_fail(self) -> None:
        cases: list[tuple[str, dict[str, object], str]] = []
        nonfinite = _specification("cabinet")
        nonfinite["links"]["cabinet"]["mass_kg"] = float("nan")
        cases.append(("nonfinite", nonfinite, "non-finite"))
        zero_mass = _specification("cabinet")
        zero_mass["links"]["cabinet"]["mass_kg"] = 0.0
        cases.append(("zero_mass", zero_mass, "must be positive"))
        indefinite = _specification("cabinet")
        indefinite["links"]["cabinet"]["inertia_kg_m2"]["izz"] = -0.3
        cases.append(("indefinite", indefinite, "positive-definite"))
        unrealizable = _specification("cabinet")
        unrealizable["links"]["cabinet"]["inertia_kg_m2"] = {
            "ixx": 1.0,
            "ixy": 0.0,
            "ixz": 0.0,
            "iyy": 1.0,
            "iyz": 0.0,
            "izz": 3.0,
        }
        cases.append(("unrealizable", unrealizable, "physically realizable"))

        for name, payload, message in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                urdf, spec = _write_case(root, payload, "cabinet")
                before = urdf.read_bytes()
                with self.assertRaisesRegex(InertialSpecificationError, message):
                    apply_inertials(urdf, spec)
                self.assertEqual(urdf.read_bytes(), before)
                self.assertFalse((root / SIDECAR_NAME).exists())

    def test_build_wrapper_orders_compile_injection_and_materialization(self) -> None:
        source = (ROOT / "scripts/build_articraft_asset.sh").read_text(encoding="utf-8")
        compile_index = source.index("articraft compile")
        injector_index = source.index('scripts/apply_articraft_inertials.py')
        inspector_index = source.index('src.asset_inspector')
        materialize_index = source.index('scripts/materialize_articraft_asset.py')
        self.assertLess(compile_index, injector_index)
        self.assertLess(injector_index, inspector_index)
        self.assertLess(inspector_index, materialize_index)
        for required in (
            'INERTIAL_SPEC="${ROOT}/assets/articraft/${RECORD_ID}/inertials.json"',
            '--urdf "${MATERIALIZATION_ROOT}/model.urdf"',
            '--spec "${INERTIAL_SPEC}"',
            '--sidecar "${INERTIAL_SIDECAR}"',
            '--inertial-spec "${INERTIAL_SPEC}"',
            '--inertial-sidecar "${INERTIAL_SIDECAR}"',
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main()
