#!/usr/bin/env python3
"""Inject checked-in inertial specifications into one compiled URDF.

The input JSON is the sole source of mass, inertial-frame, and inertia-matrix
values.  A pristine URDF must have exactly the links named by the specification
and all of those links must lack ``<inertial>``.  A fully postprocessed URDF is
accepted only when every existing value exactly matches the specification;
partially postprocessed files fail closed.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET


class InertialSpecificationError(ValueError):
    """The checked-in specification or compiled URDF is unsafe to apply."""


_ROOT_KEYS = {
    "schema_version",
    "state",
    "source_urdf_sha256",
    "record",
    "units",
    "links",
    "notes",
}
_RECORD_KEYS = {"id", "revision", "data_commit", "model_url"}
_UNIT_VALUES = {
    "mass": "kg",
    "length": "m",
    "angle": "rad",
    "inertia": "kg*m^2",
}
_LINK_KEYS = {"mass_kg", "origin_xyz_m", "origin_rpy_rad", "inertia_kg_m2"}
_INERTIA_KEYS = {"ixx", "ixy", "ixz", "iyy", "iyz", "izz"}
SIDECAR_NAME = "agentpre_inertial_completion.json"


def _reject_json_constant(value: str) -> None:
    raise InertialSpecificationError(
        f"inertial specification contains a non-finite JSON constant: {value}"
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InertialSpecificationError(
                f"inertial specification contains a duplicate key: {key!r}"
            )
        result[key] = value
    return result


def _require_regular_file(path: Path, description: str) -> Path:
    absolute = Path(os.path.abspath(os.path.expanduser(str(path))))
    if absolute.is_symlink():
        raise InertialSpecificationError(f"{description} must not be a symlink: {absolute}")
    try:
        mode = absolute.stat().st_mode
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{description} is missing: {absolute}") from exc
    if not stat.S_ISREG(mode):
        raise InertialSpecificationError(
            f"{description} is not a regular file: {absolute}"
        )
    return absolute.resolve()


def _read_specification(path: Path) -> dict[str, Any]:
    path = _require_regular_file(path, "inertial specification")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise InertialSpecificationError(
            f"inertial specification is not valid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise InertialSpecificationError("inertial specification root must be an object")
    return payload


def _exact_keys(value: Any, expected: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise InertialSpecificationError(f"{where} must be an object")
    actual = set(value)
    if actual != expected:
        raise InertialSpecificationError(
            f"{where} keys do not match the schema: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    return value


def _finite_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InertialSpecificationError(f"{where} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise InertialSpecificationError(f"{where} must be finite")
    return result


def _vector3(value: Any, where: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise InertialSpecificationError(f"{where} must be an array of three numbers")
    return tuple(  # type: ignore[return-value]
        _finite_number(item, f"{where}[{index}]")
        for index, item in enumerate(value)
    )


def _validate_positive_definite(inertia: Mapping[str, float], where: str) -> None:
    ixx = inertia["ixx"]
    ixy = inertia["ixy"]
    ixz = inertia["ixz"]
    iyy = inertia["iyy"]
    iyz = inertia["iyz"]
    izz = inertia["izz"]
    leading_minor_2 = ixx * iyy - ixy * ixy
    determinant = (
        ixx * (iyy * izz - iyz * iyz)
        - ixy * (ixy * izz - iyz * ixz)
        + ixz * (ixy * iyz - iyy * ixz)
    )
    if not math.isfinite(leading_minor_2) or not math.isfinite(determinant):
        raise InertialSpecificationError(
            f"{where} positive-definiteness calculation is non-finite"
        )
    if ixx <= 0.0 or leading_minor_2 <= 0.0 or determinant <= 0.0:
        raise InertialSpecificationError(
            f"{where} must be a symmetric positive-definite inertia matrix"
        )


def _validate_physical_inertia(inertia: Mapping[str, float], where: str) -> None:
    """Require the equivalent second-moment matrix to be positive semidefinite."""

    ixx = inertia["ixx"]
    ixy = inertia["ixy"]
    ixz = inertia["ixz"]
    iyy = inertia["iyy"]
    iyz = inertia["iyz"]
    izz = inertia["izz"]
    half_trace = 0.5 * (ixx + iyy + izz)
    # For any rigid body, P = trace(I)/2 * identity - I is the integral of
    # r r^T dm and therefore PSD.  Checking every principal minor is necessary
    # and sufficient for a real symmetric 3x3 matrix to be PSD.
    pxx = half_trace - ixx
    pyy = half_trace - iyy
    pzz = half_trace - izz
    pxy = -ixy
    pxz = -ixz
    pyz = -iyz
    scale = max(abs(value) for value in inertia.values())
    tolerance_1 = 1.0e-12 * scale
    tolerance_2 = 1.0e-12 * scale * scale
    tolerance_3 = 1.0e-12 * scale * scale * scale
    diagonal = (pxx, pyy, pzz)
    minors_2 = (
        pxx * pyy - pxy * pxy,
        pxx * pzz - pxz * pxz,
        pyy * pzz - pyz * pyz,
    )
    determinant = (
        pxx * (pyy * pzz - pyz * pyz)
        - pxy * (pxy * pzz - pyz * pxz)
        + pxz * (pxy * pyz - pyy * pxz)
    )
    if (
        any(value < -tolerance_1 for value in diagonal)
        or any(value < -tolerance_2 for value in minors_2)
        or determinant < -tolerance_3
    ):
        raise InertialSpecificationError(
            f"{where} is positive-definite but not physically realizable"
        )


def load_specification(path: Path) -> dict[str, Any]:
    """Load and strictly validate a finalized inertial specification."""

    payload = _read_specification(path)
    root = _exact_keys(payload, _ROOT_KEYS, "specification")
    if root["schema_version"] != 1:
        raise InertialSpecificationError("specification.schema_version must equal 1")
    source_urdf_sha256 = root["source_urdf_sha256"]
    if (
        not isinstance(source_urdf_sha256, str)
        or len(source_urdf_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source_urdf_sha256
        )
    ):
        raise InertialSpecificationError(
            "specification.source_urdf_sha256 must be a lowercase 64-character SHA-256"
        )
    state = root["state"]
    if state == "todo":
        raise InertialSpecificationError(
            "inertial specification is still TODO; replace every null placeholder "
            "with measured/reviewed values and set state to 'ready'"
        )
    if state != "ready":
        raise InertialSpecificationError(
            "specification.state must be exactly 'todo' or 'ready'"
        )

    record = _exact_keys(root["record"], _RECORD_KEYS, "specification.record")
    for key in sorted(_RECORD_KEYS):
        if not isinstance(record[key], str) or not record[key]:
            raise InertialSpecificationError(
                f"specification.record.{key} must be a non-empty string"
            )
    data_commit = record["data_commit"]
    if len(data_commit) != 40 or any(
        character not in "0123456789abcdef" for character in data_commit
    ):
        raise InertialSpecificationError(
            "specification.record.data_commit must be a lowercase 40-character SHA"
        )

    units = _exact_keys(root["units"], set(_UNIT_VALUES), "specification.units")
    if dict(units) != _UNIT_VALUES:
        raise InertialSpecificationError(
            f"specification.units must equal {_UNIT_VALUES}"
        )
    notes = root["notes"]
    if not isinstance(notes, list) or not all(
        isinstance(note, str) and note for note in notes
    ):
        raise InertialSpecificationError(
            "specification.notes must be an array of non-empty strings"
        )

    raw_links = root["links"]
    if not isinstance(raw_links, dict) or not raw_links:
        raise InertialSpecificationError(
            "specification.links must be a non-empty object"
        )
    normalized_links: dict[str, Any] = {}
    for link_name, raw_inertial in raw_links.items():
        if not isinstance(link_name, str) or not link_name:
            raise InertialSpecificationError(
                "specification link names must be non-empty strings"
            )
        inertial = _exact_keys(
            raw_inertial, _LINK_KEYS, f"specification.links.{link_name}"
        )
        mass = _finite_number(
            inertial["mass_kg"], f"specification.links.{link_name}.mass_kg"
        )
        if mass <= 0.0:
            raise InertialSpecificationError(
                f"specification.links.{link_name}.mass_kg must be positive"
            )
        origin_xyz = _vector3(
            inertial["origin_xyz_m"],
            f"specification.links.{link_name}.origin_xyz_m",
        )
        origin_rpy = _vector3(
            inertial["origin_rpy_rad"],
            f"specification.links.{link_name}.origin_rpy_rad",
        )
        raw_inertia = _exact_keys(
            inertial["inertia_kg_m2"],
            _INERTIA_KEYS,
            f"specification.links.{link_name}.inertia_kg_m2",
        )
        inertia = {
            component: _finite_number(
                raw_inertia[component],
                f"specification.links.{link_name}.inertia_kg_m2.{component}",
            )
            for component in sorted(_INERTIA_KEYS)
        }
        _validate_positive_definite(
            inertia, f"specification.links.{link_name}.inertia_kg_m2"
        )
        _validate_physical_inertia(
            inertia, f"specification.links.{link_name}.inertia_kg_m2"
        )
        normalized_links[link_name] = {
            "mass_kg": mass,
            "origin_xyz_m": origin_xyz,
            "origin_rpy_rad": origin_rpy,
            "inertia_kg_m2": inertia,
        }

    return {
        "schema_version": 1,
        "state": "ready",
        "source_urdf_sha256": source_urdf_sha256,
        "record": dict(record),
        "units": dict(units),
        "links": normalized_links,
        "notes": list(notes),
    }


def _format_number(value: float) -> str:
    if value == 0.0:
        return "0"
    return format(value, ".17g")


def _format_vector(values: Sequence[float]) -> str:
    return " ".join(_format_number(value) for value in values)


def _xml_float(value: str | None, where: str) -> float:
    if value is None:
        raise InertialSpecificationError(f"existing {where} is missing")
    try:
        result = float(value)
    except ValueError as exc:
        raise InertialSpecificationError(
            f"existing {where} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(result):
        raise InertialSpecificationError(f"existing {where} is non-finite")
    return result


def _xml_vector(value: str | None, where: str) -> tuple[float, float, float]:
    if value is None:
        return (0.0, 0.0, 0.0)
    parts = value.split()
    if len(parts) != 3:
        raise InertialSpecificationError(
            f"existing {where} must contain three numbers"
        )
    return tuple(  # type: ignore[return-value]
        _xml_float(part, f"{where}[{index}]")
        for index, part in enumerate(parts)
    )


def _assert_existing_matches(
    link_name: str, element: ET.Element, expected: Mapping[str, Any]
) -> None:
    origins = element.findall("origin")
    masses = element.findall("mass")
    inertias = element.findall("inertia")
    if len(origins) != 1 or len(masses) != 1 or len(inertias) != 1:
        raise InertialSpecificationError(
            f"existing inertial for link {link_name!r} has an invalid structure"
        )
    expected_attributes = {
        "inertial": set(),
        "origin": {"xyz", "rpy"},
        "mass": {"value"},
        "inertia": set(_INERTIA_KEYS),
    }
    actual_attributes = {
        "inertial": set(element.attrib),
        "origin": set(origins[0].attrib),
        "mass": set(masses[0].attrib),
        "inertia": set(inertias[0].attrib),
    }
    if actual_attributes != expected_attributes:
        raise InertialSpecificationError(
            f"existing inertial for link {link_name!r} has noncanonical attributes: "
            f"existing={actual_attributes}, expected={expected_attributes}"
        )
    allowed_children = {"origin", "mass", "inertia"}
    unexpected_children = [child.tag for child in element if child.tag not in allowed_children]
    if unexpected_children:
        raise InertialSpecificationError(
            f"existing inertial for link {link_name!r} has unexpected children: "
            f"{unexpected_children}"
        )
    origin_xyz = _xml_vector(
        origins[0].get("xyz"),
        f"inertial origin xyz for link {link_name!r}",
    )
    origin_rpy = _xml_vector(
        origins[0].get("rpy"),
        f"inertial origin rpy for link {link_name!r}",
    )
    mass = _xml_float(
        masses[0].get("value"), f"inertial mass for link {link_name!r}"
    )
    inertia = {
        component: _xml_float(
            inertias[0].get(component),
            f"inertial {component} for link {link_name!r}",
        )
        for component in sorted(_INERTIA_KEYS)
    }
    actual = {
        "mass_kg": mass,
        "origin_xyz_m": origin_xyz,
        "origin_rpy_rad": origin_rpy,
        "inertia_kg_m2": inertia,
    }
    if actual != expected:
        raise InertialSpecificationError(
            f"existing inertial for link {link_name!r} does not match the specification: "
            f"existing={actual}, expected={dict(expected)}"
        )


def _make_inertial(expected: Mapping[str, Any]) -> ET.Element:
    element = ET.Element("inertial")
    ET.SubElement(
        element,
        "origin",
        {
            "xyz": _format_vector(expected["origin_xyz_m"]),
            "rpy": _format_vector(expected["origin_rpy_rad"]),
        },
    )
    ET.SubElement(element, "mass", {"value": _format_number(expected["mass_kg"])})
    ET.SubElement(
        element,
        "inertia",
        {
            component: _format_number(expected["inertia_kg_m2"][component])
            for component in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
        },
    )
    return element


def _write_atomic(
    path: Path,
    payload: bytes,
    mode: int,
    expected_original: bytes | None,
    description: str,
) -> None:
    if path.is_symlink():
        raise RuntimeError(f"{description} must not be a symlink: {path}")
    if expected_original is None:
        if path.exists():
            raise RuntimeError(f"{description} appeared while it was being prepared: {path}")
    else:
        if not path.is_file() or path.read_bytes() != expected_original:
            raise RuntimeError(f"{description} changed while it was being prepared: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, stat.S_IMODE(mode))
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some filesystems do not support syncing directory descriptors.
            pass
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _existing_sidecar_bytes(path: Path) -> bytes | None:
    if path.is_symlink():
        raise InertialSpecificationError(
            f"inertial completion sidecar must not be a symlink: {path}"
        )
    if not path.exists():
        return None
    if not path.is_file():
        raise InertialSpecificationError(
            f"inertial completion sidecar is not a regular file: {path}"
        )
    return path.read_bytes()


def _completion_payload(
    *,
    urdf_path: Path,
    spec_path: Path,
    spec_sha256: str,
    pre_urdf_sha256: str,
    post_urdf_sha256: str,
    links: set[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "specification": {
            "path": str(spec_path),
            "sha256": spec_sha256,
        },
        "urdf": {
            "path": str(urdf_path),
            "pre_sha256": pre_urdf_sha256,
            "post_sha256": post_urdf_sha256,
        },
        "injected_links": sorted(links),
    }


def _validate_completion_sidecar(
    sidecar_path: Path,
    sidecar_bytes: bytes | None,
    *,
    urdf_path: Path,
    urdf_sha256: str,
    spec_path: Path,
    spec_sha256: str,
    source_urdf_sha256: str,
    links: set[str],
) -> tuple[dict[str, Any], str]:
    if sidecar_bytes is None:
        raise InertialSpecificationError(
            "URDF already contains matching inertials but the completion sidecar is "
            f"missing; recompile the pristine Articraft record before retrying: {sidecar_path}"
        )
    try:
        payload = json.loads(
            sidecar_bytes.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InertialSpecificationError(
            f"inertial completion sidecar is invalid: {sidecar_path}"
        ) from exc
    root = _exact_keys(
        payload,
        {"schema_version", "specification", "urdf", "injected_links"},
        "completion sidecar",
    )
    specification = _exact_keys(
        root["specification"], {"path", "sha256"}, "completion sidecar.specification"
    )
    urdf = _exact_keys(
        root["urdf"],
        {"path", "pre_sha256", "post_sha256"},
        "completion sidecar.urdf",
    )
    expected_fixed = {
        "schema_version": 1,
        "specification": {"path": str(spec_path), "sha256": spec_sha256},
        "urdf_path": str(urdf_path),
        "post_sha256": urdf_sha256,
        "injected_links": sorted(links),
    }
    actual_fixed = {
        "schema_version": root["schema_version"],
        "specification": dict(specification),
        "urdf_path": urdf["path"],
        "post_sha256": urdf["post_sha256"],
        "injected_links": root["injected_links"],
    }
    if actual_fixed != expected_fixed:
        raise InertialSpecificationError(
            "inertial completion sidecar does not match the current specification/URDF: "
            f"existing={actual_fixed}, expected={expected_fixed}"
        )
    pre_sha256 = urdf["pre_sha256"]
    if (
        not isinstance(pre_sha256, str)
        or len(pre_sha256) != 64
        or any(character not in "0123456789abcdef" for character in pre_sha256)
        or pre_sha256 != source_urdf_sha256
        or pre_sha256 == urdf_sha256
    ):
        raise InertialSpecificationError(
            "completion sidecar.urdf.pre_sha256 must equal the specification's "
            "distinct source_urdf_sha256"
        )
    return dict(root), _sha256_bytes(sidecar_bytes)


def _process_inertials_locked(
    urdf_path: Path,
    spec_path: Path,
    sidecar_path: Path | None,
    *,
    allow_injection: bool,
) -> dict[str, Any]:
    """Process one URDF while the caller holds its transaction lock."""

    urdf_path = _require_regular_file(urdf_path, "compiled URDF")
    spec_path = _require_regular_file(spec_path, "inertial specification")
    if sidecar_path is None:
        sidecar_path = urdf_path.parent / SIDECAR_NAME
    sidecar_input = Path(os.path.abspath(os.path.expanduser(str(sidecar_path))))
    if sidecar_input.is_symlink():
        raise InertialSpecificationError(
            f"inertial completion sidecar must not be a symlink: {sidecar_input}"
        )
    sidecar_path = sidecar_input.resolve()
    if sidecar_path in {urdf_path, spec_path}:
        raise InertialSpecificationError(
            "completion sidecar path must differ from the URDF and specification"
        )
    if not sidecar_path.parent.is_dir():
        raise FileNotFoundError(
            f"completion sidecar parent directory is missing: {sidecar_path.parent}"
        )
    sidecar_original = _existing_sidecar_bytes(sidecar_path)
    spec_sha256 = _sha256(spec_path)
    specification = load_specification(spec_path)
    original = urdf_path.read_bytes()
    original_sha256 = _sha256_bytes(original)
    mode = urdf_path.stat().st_mode
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    try:
        root = ET.fromstring(original, parser=parser)
    except ET.ParseError as exc:
        raise InertialSpecificationError(
            f"compiled URDF is not valid XML: {urdf_path}: {exc}"
        ) from exc
    if root.tag != "robot":
        raise InertialSpecificationError("compiled URDF root must be <robot>")

    link_elements: dict[str, ET.Element] = {}
    for link in root.findall("link"):
        name = link.get("name")
        if not name:
            raise InertialSpecificationError("compiled URDF contains an unnamed link")
        if name in link_elements:
            raise InertialSpecificationError(
                f"compiled URDF contains duplicate link name: {name!r}"
            )
        link_elements[name] = link
    specification_links = specification["links"]
    actual_names = set(link_elements)
    specified_names = set(specification_links)
    if actual_names != specified_names:
        raise InertialSpecificationError(
            "specification link set must exactly match the compiled URDF link set: "
            f"missing_from_spec={sorted(actual_names - specified_names)}, "
            f"unknown_or_extra_in_spec={sorted(specified_names - actual_names)}"
        )

    existing: dict[str, ET.Element] = {}
    missing: set[str] = set()
    for name, link in link_elements.items():
        inertials = link.findall("inertial")
        if len(inertials) > 1:
            raise InertialSpecificationError(
                f"compiled URDF link {name!r} contains multiple <inertial> elements"
            )
        if inertials:
            existing[name] = inertials[0]
        else:
            missing.add(name)

    if not missing:
        for name in sorted(specified_names):
            _assert_existing_matches(name, existing[name], specification_links[name])
        completion, sidecar_sha256 = _validate_completion_sidecar(
            sidecar_path,
            sidecar_original,
            urdf_path=urdf_path,
            urdf_sha256=original_sha256,
            spec_path=spec_path,
            spec_sha256=spec_sha256,
            source_urdf_sha256=specification["source_urdf_sha256"],
            links=specified_names,
        )
        return {
            "modified": False,
            "urdf": str(urdf_path),
            "specification": str(spec_path),
            "specification_sha256": spec_sha256,
            "links": sorted(specified_names),
            "completion_sidecar": str(sidecar_path),
            "completion_sidecar_sha256": sidecar_sha256,
            "pre_urdf_sha256": completion["urdf"]["pre_sha256"],
            "post_urdf_sha256": original_sha256,
        }

    if missing != specified_names:
        for name in sorted(existing):
            _assert_existing_matches(name, existing[name], specification_links[name])
        raise InertialSpecificationError(
            "refusing a partially postprocessed URDF: specification link set must "
            "exactly match the missing-inertial link set; "
            f"specified={sorted(specified_names)}, missing={sorted(missing)}, "
            f"already_present={sorted(existing)}"
        )

    if original_sha256 != specification["source_urdf_sha256"]:
        raise InertialSpecificationError(
            "pristine compiled URDF SHA-256 does not match "
            "specification.source_urdf_sha256: "
            f"actual={original_sha256}, "
            f"expected={specification['source_urdf_sha256']}"
        )
    if not allow_injection:
        raise InertialSpecificationError(
            "completed-inertial validation requires every URDF link to already "
            "contain its matching inertial"
        )

    for name, link in link_elements.items():
        link.insert(0, _make_inertial(specification_links[name]))
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    rendered = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    if not rendered.endswith(b"\n"):
        rendered += b"\n"
    post_sha256 = _sha256_bytes(rendered)
    completion = _completion_payload(
        urdf_path=urdf_path,
        spec_path=spec_path,
        spec_sha256=spec_sha256,
        pre_urdf_sha256=original_sha256,
        post_urdf_sha256=post_sha256,
        links=specified_names,
    )
    completion_bytes = (
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if _sha256(spec_path) != spec_sha256:
        raise RuntimeError(
            f"inertial specification changed while inertials were prepared: {spec_path}"
        )
    # Publish the future-state sidecar first.  If the process stops between the
    # two atomic replacements, the pristine URDF plus sidecar can be retried:
    # the sidecar is replaced again and the URDF is then completed.  Writing
    # the URDF first would leave a completed URDF without its required sidecar,
    # which intentionally cannot be reconstructed from an untrusted pre-hash.
    _write_atomic(
        sidecar_path,
        completion_bytes,
        0o644,
        sidecar_original,
        "inertial completion sidecar",
    )
    _write_atomic(urdf_path, rendered, mode, original, "compiled URDF")
    return {
        "modified": True,
        "urdf": str(urdf_path),
        "specification": str(spec_path),
        "specification_sha256": spec_sha256,
        "links": sorted(specified_names),
        "completion_sidecar": str(sidecar_path),
        "completion_sidecar_sha256": _sha256_bytes(completion_bytes),
        "pre_urdf_sha256": original_sha256,
        "post_urdf_sha256": post_sha256,
    }


@contextmanager
def _inertial_transaction_lock(urdf_path: Path):
    absolute = Path(os.path.abspath(os.path.expanduser(str(urdf_path))))
    parent = absolute.parent.resolve()
    lock_path = parent / f".{absolute.name}.agentpre-inertial.lock"
    if lock_path.is_symlink():
        raise InertialSpecificationError(
            f"inertial transaction lock must not be a symlink: {lock_path}"
        )
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise InertialSpecificationError(
                f"inertial transaction lock is not a regular file: {lock_path}"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _process_inertials(
    urdf_path: Path,
    spec_path: Path,
    sidecar_path: Path | None,
    *,
    allow_injection: bool,
) -> dict[str, Any]:
    with _inertial_transaction_lock(urdf_path):
        return _process_inertials_locked(
            urdf_path,
            spec_path,
            sidecar_path,
            allow_injection=allow_injection,
        )


def apply_inertials(
    urdf_path: Path, spec_path: Path, sidecar_path: Path | None = None
) -> dict[str, Any]:
    """Inject all missing inertials with a locked, recoverable two-file commit."""

    return _process_inertials(
        urdf_path, spec_path, sidecar_path, allow_injection=True
    )


def validate_completed_inertials(
    urdf_path: Path, spec_path: Path, sidecar_path: Path | None = None
) -> dict[str, Any]:
    """Read-only validation for a completed URDF/specification/sidecar triple."""

    return _process_inertials(
        urdf_path, spec_path, sidecar_path, allow_injection=False
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument(
        "--sidecar",
        type=Path,
        help=f"stable completion record (default: URDF directory/{SIDECAR_NAME})",
    )
    args = parser.parse_args()
    result = apply_inertials(args.urdf, args.spec, args.sidecar)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
