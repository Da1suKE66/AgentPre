"""Small, dependency-light URDF model used by deterministic asset checks.

The parser intentionally addresses links and joints by their URDF names.  It
does not expose or rely on simulator body/joint indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET

import numpy as np


SUPPORTED_JOINT_TYPES = frozenset(
    {"fixed", "revolute", "continuous", "prismatic", "floating", "planar"}
)


class URDFModelError(ValueError):
    """Structured exception raised when a URDF cannot be represented safely."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": dict(self.context)}


@dataclass(frozen=True, slots=True)
class Origin:
    """URDF origin using XYZ translation and fixed-axis RPY rotation, in SI units."""

    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class Inertia:
    ixx: float | None
    ixy: float | None
    ixz: float | None
    iyy: float | None
    iyz: float | None
    izz: float | None

    @property
    def complete(self) -> bool:
        return all(value is not None for value in self.components.values())

    @property
    def components(self) -> Mapping[str, float | None]:
        return MappingProxyType(
            {
                "ixx": self.ixx,
                "ixy": self.ixy,
                "ixz": self.ixz,
                "iyy": self.iyy,
                "iyz": self.iyz,
                "izz": self.izz,
            }
        )

    def as_matrix(self) -> np.ndarray | None:
        """Return the symmetric inertia matrix, or ``None`` if fields are missing."""

        if not self.complete:
            return None
        assert self.ixx is not None
        assert self.ixy is not None
        assert self.ixz is not None
        assert self.iyy is not None
        assert self.iyz is not None
        assert self.izz is not None
        return np.asarray(
            [
                [self.ixx, self.ixy, self.ixz],
                [self.ixy, self.iyy, self.iyz],
                [self.ixz, self.iyz, self.izz],
            ],
            dtype=float,
        )


@dataclass(frozen=True, slots=True)
class Inertial:
    origin: Origin
    mass: float | None
    inertia: Inertia | None


@dataclass(frozen=True, slots=True)
class MeshReference:
    link_name: str
    usage: str
    filename: str
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True, slots=True)
class Link:
    name: str
    inertial: Inertial | None
    meshes: tuple[MeshReference, ...] = ()


@dataclass(frozen=True, slots=True)
class JointLimit:
    lower: float | None = None
    upper: float | None = None
    effort: float | None = None
    velocity: float | None = None


@dataclass(frozen=True, slots=True)
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: Origin
    axis: tuple[float, float, float]
    limit: JointLimit | None


@dataclass(frozen=True, slots=True)
class URDFModel:
    name: str
    path: Path
    links: Mapping[str, Link] = field(repr=False)
    joints: Mapping[str, Joint] = field(repr=False)

    @property
    def link_names(self) -> tuple[str, ...]:
        return tuple(self.links)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self.joints)

    @property
    def root_link_names(self) -> tuple[str, ...]:
        child_links = {joint.child for joint in self.joints.values()}
        return tuple(name for name in self.links if name not in child_links)

    def require_link(self, name: str) -> Link:
        try:
            return self.links[name]
        except KeyError as exc:
            raise URDFModelError(
                "LINK_NOT_FOUND", f"URDF link not found: {name}", name=name
            ) from exc

    def require_joint(self, name: str) -> Joint:
        try:
            return self.joints[name]
        except KeyError as exc:
            raise URDFModelError(
                "JOINT_NOT_FOUND", f"URDF joint not found: {name}", name=name
            ) from exc


def _finite_float(text: str, *, field_name: str, context: Mapping[str, Any]) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise URDFModelError(
            "INVALID_NUMBER",
            f"{field_name} must be numeric, got {text!r}",
            field=field_name,
            value=text,
            **context,
        ) from exc
    if not np.isfinite(value):
        raise URDFModelError(
            "NONFINITE_NUMBER",
            f"{field_name} must be finite, got {text!r}",
            field=field_name,
            value=text,
            **context,
        )
    return value


def _vector_attribute(
    element: ET.Element | None,
    attribute: str,
    *,
    default: tuple[float, float, float],
    field_name: str,
    context: Mapping[str, Any],
) -> tuple[float, float, float]:
    if element is None or element.get(attribute) is None:
        return default
    text = element.get(attribute, "")
    parts = text.split()
    if len(parts) != 3:
        raise URDFModelError(
            "INVALID_VECTOR",
            f"{field_name} must have exactly 3 values, got {text!r}",
            field=field_name,
            value=text,
            **context,
        )
    values = tuple(
        _finite_float(value, field_name=f"{field_name}[{index}]", context=context)
        for index, value in enumerate(parts)
    )
    return values  # type: ignore[return-value]


def _optional_float_attribute(
    element: ET.Element | None,
    attribute: str,
    *,
    field_name: str,
    context: Mapping[str, Any],
) -> float | None:
    if element is None or element.get(attribute) is None:
        return None
    return _finite_float(element.get(attribute, ""), field_name=field_name, context=context)


def _parse_origin(element: ET.Element | None, context: Mapping[str, Any]) -> Origin:
    return Origin(
        xyz=_vector_attribute(
            element,
            "xyz",
            default=(0.0, 0.0, 0.0),
            field_name="origin.xyz",
            context=context,
        ),
        rpy=_vector_attribute(
            element,
            "rpy",
            default=(0.0, 0.0, 0.0),
            field_name="origin.rpy",
            context=context,
        ),
    )


def _parse_inertial(element: ET.Element | None, link_name: str) -> Inertial | None:
    if element is None:
        return None
    context = {"link": link_name}
    mass_element = element.find("mass")
    mass = _optional_float_attribute(
        mass_element, "value", field_name="inertial.mass", context=context
    )
    inertia_element = element.find("inertia")
    inertia = None
    if inertia_element is not None:
        inertia = Inertia(
            **{
                name: _optional_float_attribute(
                    inertia_element,
                    name,
                    field_name=f"inertial.{name}",
                    context=context,
                )
                for name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
            }
        )
    return Inertial(origin=_parse_origin(element.find("origin"), context), mass=mass, inertia=inertia)


def _parse_meshes(link_element: ET.Element, link_name: str) -> tuple[MeshReference, ...]:
    meshes: list[MeshReference] = []
    for usage in ("visual", "collision"):
        for entry_index, entry in enumerate(link_element.findall(usage)):
            mesh_element = entry.find("geometry/mesh")
            if mesh_element is None:
                continue
            filename = (mesh_element.get("filename") or "").strip()
            context = {"link": link_name, "usage": usage, "entry_index": entry_index}
            scale = _vector_attribute(
                mesh_element,
                "scale",
                default=(1.0, 1.0, 1.0),
                field_name="mesh.scale",
                context=context,
            )
            meshes.append(
                MeshReference(
                    link_name=link_name,
                    usage=usage,
                    filename=filename,
                    scale=scale,
                )
            )
    return tuple(meshes)


def _parse_link(element: ET.Element) -> Link:
    name = (element.get("name") or "").strip()
    if not name:
        raise URDFModelError("LINK_NAME_MISSING", "URDF link is missing a non-empty name")
    return Link(
        name=name,
        inertial=_parse_inertial(element.find("inertial"), name),
        meshes=_parse_meshes(element, name),
    )


def _parse_joint(element: ET.Element) -> Joint:
    name = (element.get("name") or "").strip()
    if not name:
        raise URDFModelError("JOINT_NAME_MISSING", "URDF joint is missing a non-empty name")
    joint_type = (element.get("type") or "").strip()
    if not joint_type:
        raise URDFModelError(
            "JOINT_TYPE_MISSING", f"URDF joint {name!r} is missing its type", joint=name
        )
    parent_element = element.find("parent")
    child_element = element.find("child")
    parent = (parent_element.get("link") if parent_element is not None else "") or ""
    child = (child_element.get("link") if child_element is not None else "") or ""
    parent, child = parent.strip(), child.strip()
    if not parent or not child:
        raise URDFModelError(
            "JOINT_LINK_MISSING",
            f"URDF joint {name!r} must name both parent and child links",
            joint=name,
            parent=parent,
            child=child,
        )
    context = {"joint": name}
    axis = _vector_attribute(
        element.find("axis"),
        "xyz",
        default=(1.0, 0.0, 0.0),
        field_name="joint.axis",
        context=context,
    )
    limit_element = element.find("limit")
    limit = None
    if limit_element is not None:
        limit = JointLimit(
            lower=_optional_float_attribute(
                limit_element, "lower", field_name="joint.limit.lower", context=context
            ),
            upper=_optional_float_attribute(
                limit_element, "upper", field_name="joint.limit.upper", context=context
            ),
            effort=_optional_float_attribute(
                limit_element, "effort", field_name="joint.limit.effort", context=context
            ),
            velocity=_optional_float_attribute(
                limit_element, "velocity", field_name="joint.limit.velocity", context=context
            ),
        )
    return Joint(
        name=name,
        joint_type=joint_type,
        parent=parent,
        child=child,
        origin=_parse_origin(element.find("origin"), context),
        axis=axis,
        limit=limit,
    )


def load_urdf(path: str | Path) -> URDFModel:
    """Parse a URDF XML file into mappings keyed by link and joint names."""

    urdf_path = Path(path).expanduser()
    try:
        tree = ET.parse(urdf_path)
    except FileNotFoundError as exc:
        raise URDFModelError(
            "URDF_NOT_FOUND", f"URDF file does not exist: {urdf_path}", path=str(urdf_path)
        ) from exc
    except OSError as exc:
        raise URDFModelError(
            "URDF_READ_ERROR", f"Cannot read URDF file: {exc}", path=str(urdf_path)
        ) from exc
    except ET.ParseError as exc:
        raise URDFModelError(
            "URDF_XML_INVALID",
            f"Invalid URDF XML: {exc}",
            path=str(urdf_path),
            line=exc.position[0],
            column=exc.position[1],
        ) from exc

    root = tree.getroot()
    if root.tag != "robot":
        raise URDFModelError(
            "URDF_ROOT_INVALID",
            f"URDF root element must be <robot>, got <{root.tag}>",
            path=str(urdf_path),
        )
    robot_name = (root.get("name") or "").strip()
    if not robot_name:
        raise URDFModelError(
            "ROBOT_NAME_MISSING", "URDF <robot> is missing a non-empty name", path=str(urdf_path)
        )

    links: dict[str, Link] = {}
    for element in root.findall("link"):
        link = _parse_link(element)
        if link.name in links:
            raise URDFModelError(
                "DUPLICATE_LINK_NAME",
                f"Duplicate URDF link name: {link.name}",
                path=str(urdf_path),
                link=link.name,
            )
        links[link.name] = link

    joints: dict[str, Joint] = {}
    for element in root.findall("joint"):
        joint = _parse_joint(element)
        if joint.name in joints:
            raise URDFModelError(
                "DUPLICATE_JOINT_NAME",
                f"Duplicate URDF joint name: {joint.name}",
                path=str(urdf_path),
                joint=joint.name,
            )
        joints[joint.name] = joint

    if not links:
        raise URDFModelError(
            "LINKS_MISSING", "URDF must contain at least one link", path=str(urdf_path)
        )

    return URDFModel(
        name=robot_name,
        path=urdf_path.resolve(),
        links=MappingProxyType(links),
        joints=MappingProxyType(joints),
    )


def resolve_mesh_path(
    mesh: MeshReference,
    urdf_path: str | Path,
    package_paths: Mapping[str, str | Path] | None = None,
) -> Path | None:
    """Resolve a URDF mesh filename without accessing the network.

    Relative paths are resolved against the URDF directory. ``file://`` URLs
    and absolute paths are supported. A ``package://`` URL is resolved only if
    its package name is present in ``package_paths``; otherwise ``None`` is
    returned so the inspector can emit a structured unresolved-package issue.
    Other URI schemes also return ``None``.
    """

    filename = mesh.filename.strip()
    if not filename:
        return None
    parsed = urlparse(filename)
    if parsed.scheme == "package":
        package_name = parsed.netloc
        relative = unquote(parsed.path.lstrip("/"))
        if not package_name or not package_paths or package_name not in package_paths:
            return None
        return (Path(package_paths[package_name]).expanduser() / relative).resolve()
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).expanduser().resolve()
    if parsed.scheme:
        return None
    candidate = Path(unquote(filename)).expanduser()
    if not candidate.is_absolute():
        candidate = Path(urdf_path).expanduser().resolve().parent / candidate
    return candidate.resolve()
