#!/usr/bin/env python3
"""Materialize the configured Franka URDF/mesh tree in the AgentPre cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    collision_probe = stable_path / "meshes" / "collision" / "link7.stl"
    if not collision_probe.is_file():
        raise FileNotFoundError(f"Franka collision mesh is missing: {collision_probe}")

    manifest = {
        "asset": str(robot["name"]),
        "asset_license": str(robot["asset_license"]),
        "bootstrap_source": str(source),
        "stable_path": str(stable_path),
        "urdf": str(urdf),
        "urdf_sha256": sha256(urdf),
        "collision_probe": str(collision_probe),
    }
    manifest_path = asset_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
