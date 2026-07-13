#!/usr/bin/env python3
"""Materialize the configured Franka URDF/mesh tree in the AgentPre cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def referenced_meshes(urdf: Path, package_root: Path) -> tuple[Path, ...]:
    """Resolve every URDF mesh reference and fail on unsupported packages."""

    try:
        root = ET.parse(urdf).getroot()
    except (ET.ParseError, OSError) as exc:
        raise RuntimeError(f"cannot parse Franka URDF {urdf}: {exc}") from exc
    resolved: set[Path] = set()
    package_prefix = "package://franka_description/"
    for element in root.findall(".//mesh"):
        filename = (element.get("filename") or "").strip()
        if not filename:
            raise RuntimeError("Franka URDF contains a mesh without a filename")
        if filename.startswith(package_prefix):
            path = package_root / filename[len(package_prefix) :]
        elif filename.startswith("package://"):
            raise RuntimeError(f"unsupported package mesh reference: {filename}")
        else:
            path = urdf.parent / filename
        resolved.add(path.resolve())
    if not resolved:
        raise RuntimeError(f"Franka URDF contains no mesh references: {urdf}")
    missing = [str(path) for path in sorted(resolved) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Franka asset tree is incomplete; missing referenced meshes: "
            + ", ".join(missing)
        )
    return tuple(sorted(resolved))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(os.environ.get("AGENTPRE_CACHE_ROOT", "/cache/liluchen/agentpre")),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("AGENTPRE_ROOT", "/workspace/liluchen/AgentPre"))
        / "configs"
        / "microwave_franka.json",
    )
    args = parser.parse_args()

    cache_root = args.cache_root.expanduser().resolve()
    asset_root = cache_root / "assets"
    asset_root.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.expanduser().read_text(encoding="utf-8"))
    robot = config["assets"]["robot"]
    configured_urdf = str(robot["urdf"]).replace("${AGENTPRE_CACHE_ROOT}", str(cache_root))
    urdf = Path(os.path.expandvars(configured_urdf)).expanduser().resolve()
    stable_path = urdf.parents[1]
    source = Path(str(robot["bootstrap_source"])).expanduser().resolve()
    if not urdf.is_file():
        if stable_path.exists():
            raise RuntimeError(f"configured asset root exists but URDF is missing: {urdf}")
        if not (source / "robots" / "panda_arm_hand.urdf").is_file():
            raise FileNotFoundError(
                "configured Franka bootstrap source is unavailable; provide a complete "
                f"franka_description tree at {source}"
            )
        shutil.copytree(source, stable_path)
    mesh_paths = referenced_meshes(urdf, stable_path)

    manifest = {
        "asset": str(robot["name"]),
        "asset_license": str(robot["asset_license"]),
        "bootstrap_source": str(source),
        "stable_path": str(stable_path),
        "urdf": str(urdf),
        "urdf_sha256": sha256(urdf),
        "referenced_mesh_count": len(mesh_paths),
        "referenced_meshes": [
            {
                "path": str(path),
                "sha256": sha256(path),
            }
            for path in mesh_paths
        ],
    }
    manifest_path = asset_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
