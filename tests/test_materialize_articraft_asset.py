from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.apply_articraft_inertials import apply_inertials
from scripts.materialize_articraft_asset import (
    DATA_COMMIT,
    MANIFEST_NAME,
    MODEL_URL,
    RECORD_ID,
    RECORD_REVISION,
    materialize,
)


ROOT = Path(__file__).resolve().parents[1]


class MaterializeArticraftAssetTests(unittest.TestCase):
    def test_build_wrapper_is_pinned_offline_cpu_only_and_strict(self) -> None:
        source = (ROOT / "scripts/build_articraft_asset.sh").read_text(
            encoding="utf-8"
        )
        for required in (
            "59eb5e0ed72a734111012b43f881423b15d4931d",
            "0cdcaa49f5571e9b4df04476c7f09587ee3ab7bd",
            "--frozen",
            "--offline",
            "--no-sync",
            "--validate",
            "--strict-geom-qc",
            'export CUDA_VISIBLE_DEVICES=""',
            "materialize_articraft_asset.py",
            "src.asset_inspector",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)

    def test_setup_wrapper_pins_uv_python_commits_cache_and_cpu(self) -> None:
        source = (ROOT / "scripts/setup_articraft_env.sh").read_text(
            encoding="utf-8"
        )
        for required in (
            "/cache/liluchen/articraft-env",
            "/cache/liluchen/articraft-uv-bootstrap",
            "/cache/liluchen/articraft-uv-cache",
            "59eb5e0ed72a734111012b43f881423b15d4931d",
            "0cdcaa49f5571e9b4df04476c7f09587ee3ab7bd",
            'UV_VERSION="0.9.17"',
            "UV_PYTHON_DOWNLOADS=never",
            'UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"',
            'UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-1}"',
            'UV_CONCURRENT_BUILDS="${UV_CONCURRENT_BUILDS:-1}"',
            'UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-1}"',
            "--frozen",
            "--no-dev",
            'export CUDA_VISIBLE_DEVICES=""',
            "import cadquery, manifold3d, trimesh",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)

    def _compiled_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        source = root / "compiled"
        mesh = source / "assets" / "meshes" / "door.obj"
        mesh.parent.mkdir(parents=True)
        mesh.write_text("v 0 0 0\n", encoding="utf-8")
        (source / "model.urdf").write_text(
            '<robot name="fixture"><link name="cabinet"/></robot>\n',
            encoding="utf-8",
        )
        urdf = source / "model.urdf"
        (source / "compile_report.json").write_text(
            '{"compile_elapsed_seconds": 1.0}\n', encoding="utf-8"
        )
        spec = root / "inertials.json"
        spec.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "state": "ready",
                    "source_urdf_sha256": hashlib.sha256(
                        urdf.read_bytes()
                    ).hexdigest(),
                    "record": {
                        "id": RECORD_ID,
                        "revision": RECORD_REVISION,
                        "data_commit": DATA_COMMIT,
                        "model_url": MODEL_URL,
                    },
                    "units": {
                        "mass": "kg",
                        "length": "m",
                        "angle": "rad",
                        "inertia": "kg*m^2",
                    },
                    "links": {
                        "cabinet": {
                            "mass_kg": 1.0,
                            "origin_xyz_m": [0.0, 0.0, 0.0],
                            "origin_rpy_rad": [0.0, 0.0, 0.0],
                            "inertia_kg_m2": {
                                "ixx": 1.0,
                                "ixy": 0.0,
                                "ixz": 0.0,
                                "iyy": 1.0,
                                "iyz": 0.0,
                                "izz": 1.0,
                            },
                        }
                    },
                    "notes": ["strict materializer fixture"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        sidecar = source / "agentpre_inertial_completion.json"
        result = apply_inertials(urdf, spec, sidecar)
        self.assertTrue(result["modified"])
        return source, spec, sidecar

    def _materialize(
        self,
        source: Path,
        cache: Path,
        spec: Path,
        sidecar: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        return materialize(
            source,
            cache,
            inertial_spec=spec,
            inertial_sidecar=sidecar,
            **kwargs,
        )

    def test_materializes_to_cache_with_hash_manifest_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, spec, sidecar = self._compiled_fixture(root)
            cache = root / "cache"

            first = self._materialize(source, cache, spec, sidecar)
            destination = cache / "assets" / "articraft" / RECORD_ID
            self.assertTrue(first["copied"])
            self.assertEqual(first["file_count"], 2)
            self.assertTrue((destination / "model.urdf").is_file())
            self.assertTrue((destination / "assets/meshes/door.obj").is_file())
            on_disk = json.loads(
                (cache / "assets" / "articraft" / MANIFEST_NAME).read_text()
            )
            self.assertNotIn("copied", on_disk)
            self.assertEqual(first, {**on_disk, "copied": True})
            self.assertEqual(on_disk["files"], first["files"])
            self.assertEqual(on_disk["data_license"], "CC-BY-4.0")
            self.assertIsNotNone(on_disk["source_compile_report"])
            provenance = on_disk["inertial_postprocessing"]
            self.assertEqual(provenance["specification"]["path"], str(spec.resolve()))
            self.assertEqual(
                provenance["specification"]["sha256"],
                hashlib.sha256(spec.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                provenance["completion_sidecar"]["content"]["injected_links"],
                ["cabinet"],
            )
            self.assertEqual(
                provenance["completion_sidecar"]["sha256"],
                hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            )
            self.assertFalse((destination / sidecar.name).exists())
            self.assertFalse((destination / "compile_report.json").exists())
            manifest_path = cache / "assets" / "articraft" / MANIFEST_NAME
            manifest_bytes = manifest_path.read_bytes()
            manifest_mtime_ns = manifest_path.stat().st_mtime_ns

            # Compilation time is diagnostic and may change when the exact
            # same URDF/mesh build is regenerated.
            (source / "compile_report.json").write_text(
                '{"compile_elapsed_seconds": 2.0}\n', encoding="utf-8"
            )
            second = self._materialize(source, cache, spec, sidecar)
            self.assertFalse(second["copied"])
            self.assertEqual(second["files"], first["files"])
            self.assertEqual(manifest_path.read_bytes(), manifest_bytes)
            self.assertEqual(manifest_path.stat().st_mtime_ns, manifest_mtime_ns)

    def test_specification_drift_is_rejected_against_the_sidecar_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, spec, sidecar = self._compiled_fixture(root)
            cache = root / "cache"
            self._materialize(source, cache, spec, sidecar)
            manifest_path = cache / "assets" / "articraft" / MANIFEST_NAME
            manifest_bytes = manifest_path.read_bytes()

            spec.write_text('{"fixture_specification": "changed"}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "failed strict validation"):
                self._materialize(source, cache, spec, sidecar)
            self.assertEqual(manifest_path.read_bytes(), manifest_bytes)

    def test_forged_sidecar_cannot_hide_mismatched_urdf_mass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, spec, sidecar = self._compiled_fixture(root)
            urdf = source / "model.urdf"
            altered = urdf.read_text(encoding="utf-8").replace(
                '<mass value="1" />', '<mass value="9" />', 1
            )
            self.assertIn('<mass value="9" />', altered)
            urdf.write_text(altered, encoding="utf-8")
            forged = json.loads(sidecar.read_text(encoding="utf-8"))
            forged["urdf"]["post_sha256"] = hashlib.sha256(
                urdf.read_bytes()
            ).hexdigest()
            sidecar.write_text(
                json.dumps(forged, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            cache = root / "cache"

            with self.assertRaisesRegex(RuntimeError, "do not strictly match"):
                self._materialize(source, cache, spec, sidecar)
            self.assertFalse(cache.exists())

    def test_refuses_to_replace_a_different_cached_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, spec, sidecar = self._compiled_fixture(root)
            cache = root / "cache"
            self._materialize(source, cache, spec, sidecar)
            destination_urdf = (
                cache / "assets" / "articraft" / RECORD_ID / "model.urdf"
            )
            destination_urdf.write_text("changed\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                self._materialize(source, cache, spec, sidecar)

    def test_refuses_to_relabel_existing_cache_with_different_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, spec, sidecar = self._compiled_fixture(root)
            cache = root / "cache"
            self._materialize(source, cache, spec, sidecar)

            destination = cache / "assets" / "articraft" / RECORD_ID
            shutil.rmtree(destination)

            with self.assertRaisesRegex(RuntimeError, "record does not match"):
                self._materialize(
                    source, cache, spec, sidecar, data_commit="0" * 40
                )
            self.assertFalse(destination.exists())

    def test_commit_ids_and_symlinked_content_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, spec, sidecar = self._compiled_fixture(root)
            cache = root / "cache"
            with self.assertRaises(ValueError):
                self._materialize(source, cache, spec, sidecar, data_commit="main")

            external = root / "external.obj"
            external.write_text("v 1 2 3\n", encoding="utf-8")
            link = source / "assets" / "meshes" / "external.obj"
            try:
                os.symlink(external, link)
            except (NotImplementedError, OSError):
                self.skipTest("symlinks are unavailable")
            with self.assertRaisesRegex(RuntimeError, "must not contain symlinks"):
                self._materialize(source, cache, spec, sidecar)

    def test_symlinked_source_root_and_cache_component_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, spec, sidecar = self._compiled_fixture(root)
            source_link = root / "compiled-link"
            cache_target = root / "cache-target"
            cache_target.mkdir()
            cache_link = root / "cache-link"
            try:
                os.symlink(source, source_link)
                os.symlink(cache_target, cache_link)
            except (NotImplementedError, OSError):
                self.skipTest("symlinks are unavailable")

            with self.assertRaisesRegex(RuntimeError, "source root.*symlink"):
                self._materialize(source_link, root / "cache", spec, sidecar)
            with self.assertRaisesRegex(RuntimeError, "cache root.*symlink"):
                self._materialize(source, cache_link, spec, sidecar)

    def test_symlinked_destination_is_never_accepted_as_the_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, spec, sidecar = self._compiled_fixture(root)
            cache = root / "cache"
            asset_parent = cache / "assets" / "articraft"
            asset_parent.mkdir(parents=True)
            outside = root / "outside"
            outside.mkdir()
            try:
                os.symlink(outside, asset_parent / RECORD_ID)
            except (NotImplementedError, OSError):
                self.skipTest("symlinks are unavailable")

            with self.assertRaisesRegex(RuntimeError, "must not be a symlink"):
                self._materialize(source, cache, spec, sidecar)

    def test_missing_compiled_urdf_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "compiled"
            source.mkdir()
            with self.assertRaises(FileNotFoundError):
                materialize(
                    source,
                    root / "cache",
                    inertial_spec=root / "missing-spec.json",
                    inertial_sidecar=root / "missing-sidecar.json",
                )

    def test_non_regular_urdf_is_rejected_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, spec, sidecar = self._compiled_fixture(root)
            urdf = source / "model.urdf"
            urdf.unlink()
            try:
                os.mkfifo(urdf)
            except (AttributeError, NotImplementedError, OSError):
                self.skipTest("FIFO creation is unavailable")

            with self.assertRaises(FileNotFoundError):
                self._materialize(source, root / "cache", spec, sidecar)


if __name__ == "__main__":
    unittest.main()
