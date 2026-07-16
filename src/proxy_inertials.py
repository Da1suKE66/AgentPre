"""Deterministic workspace-local inertial proxies for geometry-complete URDFs.

These values are simulation aids, not measured physical properties.  Missing
link inertials are derived from the collision-envelope AABB (visual geometry is
used only when collision geometry is absent) while the source URDF is kept
byte-for-byte unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import numpy as np

from .affordances import AffordanceError, extract_handle_geometry
from .urdf_model import MeshReference, resolve_mesh_path


class ProxyInertialError(ValueError):
    """Structured failure raised when a safe proxy cannot be produced."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = dict(context)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "context": self.context}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: float) -> str:
    return format(float(value), ".17g")


@dataclass(frozen=True, slots=True)
class ProxyInertialResult:
    source_urdf: Path
    output_urdf: Path
    source_sha256: str
    output_sha256: str
    density_kg_m3: float
    minimum_mass_kg: float
    maximum_mass_kg: float
    generated_links: tuple[str, ...]
    preserved_links: tuple[str, ...]
    rewritten_mesh_references: int
    links: Mapping[str, Mapping[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": "deterministic_collision_aabb_proxy",
            "approximation": True,
            "physical_claim": "simulation_proxy_not_manufacturer_measurement",
            "formula": {
                "center_of_mass": "union geometry AABB center in link frame",
                "mass": "clip(density * AABB volume, minimum_mass, maximum_mass)",
                "inertia": "uniform solid-box diagonal tensor about AABB center",
                "geometry_preference": "collision, then visual only when collision is absent",
            },
            "density_kg_m3": self.density_kg_m3,
            "minimum_mass_kg": self.minimum_mass_kg,
            "maximum_mass_kg": self.maximum_mass_kg,
            "source_urdf": str(self.source_urdf),
            "source_urdf_sha256": self.source_sha256,
            "output_urdf": str(self.output_urdf),
            "output_urdf_sha256": self.output_sha256,
            "generated_links": list(self.generated_links),
            "preserved_links": list(self.preserved_links),
            "rewritten_mesh_references": self.rewritten_mesh_references,
            "links": {name: dict(value) for name, value in self.links.items()},
        }


def _validate_positive(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProxyInertialError("PARAMETER_INVALID", f"{field} must be numeric", field=field)
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ProxyInertialError(
            "PARAMETER_INVALID", f"{field} must be finite and positive", field=field, value=value
        )
    return result


def _rewrite_mesh_references(root: ET.Element, source_urdf: Path) -> int:
    rewritten = 0
    for link in root.findall("link"):
        link_name = (link.get("name") or "").strip()
        for usage in ("visual", "collision"):
            for entry in link.findall(usage):
                mesh = entry.find("geometry/mesh")
                if mesh is None:
                    continue
                filename = (mesh.get("filename") or "").strip()
                parsed = urlparse(filename)
                if parsed.scheme == "package":
                    continue
                reference = MeshReference(link_name, usage, filename)
                resolved = resolve_mesh_path(reference, source_urdf)
                if resolved is None or not resolved.is_file():
                    raise ProxyInertialError(
                        "MESH_UNRESOLVED",
                        "mesh reference cannot be preserved in the prepared URDF",
                        link=link_name,
                        usage=usage,
                        filename=filename,
                    )
                absolute = str(resolved)
                if filename != absolute:
                    mesh.set("filename", absolute)
                    rewritten += 1
    return rewritten


def _make_inertial(
    *,
    center: np.ndarray,
    extents: np.ndarray,
    mass: float,
) -> tuple[ET.Element, dict[str, float]]:
    x, y, z = (float(value) for value in extents)
    inertia = {
        "ixx": mass * (y * y + z * z) / 12.0,
        "ixy": 0.0,
        "ixz": 0.0,
        "iyy": mass * (x * x + z * z) / 12.0,
        "iyz": 0.0,
        "izz": mass * (x * x + y * y) / 12.0,
    }
    element = ET.Element("inertial")
    ET.SubElement(
        element,
        "origin",
        xyz=" ".join(_number(value) for value in center),
        rpy="0 0 0",
    )
    ET.SubElement(element, "mass", value=_number(mass))
    ET.SubElement(element, "inertia", **{name: _number(value) for name, value in inertia.items()})
    return element, inertia


def prepare_proxy_inertials(
    source_urdf: Path,
    output_urdf: Path,
    *,
    density_kg_m3: float = 300.0,
    minimum_mass_kg: float = 0.02,
    maximum_mass_kg: float = 1000.0,
    minimum_extent_m: float = 1.0e-4,
) -> ProxyInertialResult:
    """Write a distinct URDF with deterministic proxies for missing inertials."""

    density = _validate_positive(density_kg_m3, "density_kg_m3")
    minimum_mass = _validate_positive(minimum_mass_kg, "minimum_mass_kg")
    maximum_mass = _validate_positive(maximum_mass_kg, "maximum_mass_kg")
    minimum_extent = _validate_positive(minimum_extent_m, "minimum_extent_m")
    if minimum_mass > maximum_mass:
        raise ProxyInertialError(
            "PARAMETER_INVALID",
            "minimum_mass_kg must not exceed maximum_mass_kg",
            minimum_mass_kg=minimum_mass,
            maximum_mass_kg=maximum_mass,
        )

    source = Path(source_urdf).expanduser().resolve()
    output_input = Path(output_urdf).expanduser()
    output = output_input.resolve(strict=False)
    if source == output:
        raise ProxyInertialError(
            "SOURCE_OVERWRITE_FORBIDDEN",
            "proxy inertials must be written to a distinct URDF",
            source=str(source),
        )
    if output_input.is_symlink() or output.is_symlink():
        raise ProxyInertialError(
            "OUTPUT_SYMLINK_FORBIDDEN", "prepared URDF must not be a symlink", output=str(output)
        )
    try:
        tree = ET.parse(source)
    except FileNotFoundError as exc:
        raise ProxyInertialError("URDF_NOT_FOUND", "source URDF does not exist", path=str(source)) from exc
    except (OSError, ET.ParseError) as exc:
        raise ProxyInertialError(
            "URDF_INVALID", "source URDF cannot be parsed", path=str(source), error=str(exc)
        ) from exc
    root = tree.getroot()
    if root.tag != "robot":
        raise ProxyInertialError("URDF_INVALID", "URDF root must be <robot>", path=str(source))

    generated: list[str] = []
    preserved: list[str] = []
    provenance: dict[str, Mapping[str, Any]] = {}
    seen: set[str] = set()
    for link in root.findall("link"):
        name = (link.get("name") or "").strip()
        if not name or name in seen:
            raise ProxyInertialError(
                "LINK_NAME_INVALID", "links must have unique non-empty names", link=name
            )
        seen.add(name)
        inertials = link.findall("inertial")
        if len(inertials) > 1:
            raise ProxyInertialError(
                "INERTIAL_MULTIPLE", "link contains more than one inertial", link=name
            )
        if inertials:
            preserved.append(name)
            continue
        try:
            geometry = extract_handle_geometry(
                source, name, primitive_radial_samples=32
            )
        except AffordanceError as exc:
            raise ProxyInertialError(
                "GEOMETRY_UNAVAILABLE",
                "link has no usable local geometry for a proxy inertial",
                link=name,
                geometry_failure=exc.to_dict(),
            ) from exc
        aabb_min = np.asarray(geometry.aabb_min, dtype=float)
        aabb_max = np.asarray(geometry.aabb_max, dtype=float)
        raw_extents = aabb_max - aabb_min
        if not np.isfinite(raw_extents).all() or np.any(raw_extents < 0.0):
            raise ProxyInertialError(
                "GEOMETRY_BOUNDS_INVALID", "geometry AABB is invalid", link=name
            )
        extents = np.maximum(raw_extents, minimum_extent)
        center = (aabb_min + aabb_max) / 2.0
        volume = float(np.prod(extents))
        mass = float(np.clip(density * volume, minimum_mass, maximum_mass))
        inertial, inertia = _make_inertial(center=center, extents=extents, mass=mass)
        link.insert(0, inertial)
        generated.append(name)
        provenance[name] = {
            "geometry_sources": list(geometry.sources),
            "aabb_min_m": [float(value) for value in aabb_min],
            "aabb_max_m": [float(value) for value in aabb_max],
            "effective_extents_m": [float(value) for value in extents],
            "center_of_mass_m": [float(value) for value in center],
            "envelope_volume_m3": volume,
            "mass_kg": mass,
            "inertia_kg_m2": inertia,
        }
    if not seen:
        raise ProxyInertialError("LINKS_MISSING", "URDF contains no links", path=str(source))

    rewritten = _rewrite_mesh_references(root, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        tree.write(temporary, encoding="utf-8", xml_declaration=True)
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return ProxyInertialResult(
        source_urdf=source,
        output_urdf=output.resolve(),
        source_sha256=_sha256(source),
        output_sha256=_sha256(output),
        density_kg_m3=density,
        minimum_mass_kg=minimum_mass,
        maximum_mass_kg=maximum_mass,
        generated_links=tuple(generated),
        preserved_links=tuple(preserved),
        rewritten_mesh_references=rewritten,
        links=provenance,
    )


__all__ = ["ProxyInertialError", "ProxyInertialResult", "prepare_proxy_inertials"]
