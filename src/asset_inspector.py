"""Structured, name-based validation for articulated URDF assets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .urdf_model import (
    SUPPORTED_JOINT_TYPES,
    Joint,
    MeshReference,
    URDFModel,
    URDFModelError,
    load_urdf,
    resolve_mesh_path,
)


ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True, slots=True)
class InspectionIssue:
    """One machine-readable validation issue."""

    severity: str
    code: str
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "context": dict(self.context),
        }


@dataclass(frozen=True, slots=True)
class MeshInspection:
    link_name: str
    usage: str
    filename: str
    scale: tuple[float, float, float]
    resolved_path: str | None
    exists: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "link_name": self.link_name,
            "usage": self.usage,
            "filename": self.filename,
            "scale": list(self.scale),
            "resolved_path": self.resolved_path,
            "exists": self.exists,
        }


@dataclass(frozen=True, slots=True)
class InspectionReport:
    """Complete inspection result; ``ok`` is true only when no errors exist."""

    urdf_path: str
    robot_name: str | None
    link_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    door_joint: Mapping[str, Any] | None
    meshes: tuple[MeshInspection, ...]
    issues: tuple[InspectionIssue, ...]

    @property
    def errors(self) -> tuple[InspectionIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == ERROR)

    @property
    def warnings(self) -> tuple[InspectionIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == WARNING)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "urdf_path": self.urdf_path,
            "robot_name": self.robot_name,
            "links": list(self.link_names),
            "joints": list(self.joint_names),
            "door_joint": dict(self.door_joint) if self.door_joint is not None else None,
            "meshes": [mesh.to_dict() for mesh in self.meshes],
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


def _issue(
    issues: list[InspectionIssue],
    severity: str,
    code: str,
    message: str,
    **context: Any,
) -> None:
    issues.append(
        InspectionIssue(severity=severity, code=code, message=message, context=context)
    )


def _joint_summary(joint: Joint) -> dict[str, Any]:
    limit = None
    if joint.limit is not None:
        limit = {
            "lower": joint.limit.lower,
            "upper": joint.limit.upper,
            "effort": joint.limit.effort,
            "velocity": joint.limit.velocity,
        }
    return {
        "name": joint.name,
        "type": joint.joint_type,
        "parent": joint.parent,
        "child": joint.child,
        "origin": {"xyz": list(joint.origin.xyz), "rpy": list(joint.origin.rpy)},
        "axis": list(joint.axis),
        "limit": limit,
    }


def _check_link_inertials(model: URDFModel, issues: list[InspectionIssue]) -> None:
    for link in model.links.values():
        inertial = link.inertial
        if inertial is None:
            _issue(
                issues,
                ERROR,
                "LINK_INERTIAL_MISSING",
                "Link has no <inertial>; finite non-negative mass and inertia cannot be verified",
                link=link.name,
            )
            continue
        if inertial.mass is None:
            _issue(
                issues,
                ERROR,
                "MASS_MISSING",
                "Link <inertial> is missing <mass value=...>",
                link=link.name,
            )
        elif not math.isfinite(inertial.mass):
            # Non-finite XML values are normally rejected by the parser.  Keep
            # this guard so programmatically constructed models are also safe.
            _issue(
                issues,
                ERROR,
                "MASS_NONFINITE",
                "Link mass must be finite",
                link=link.name,
                mass=inertial.mass,
            )
        elif inertial.mass < 0.0:
            _issue(
                issues,
                ERROR,
                "MASS_NEGATIVE",
                "Link mass must be non-negative",
                link=link.name,
                mass=inertial.mass,
            )

        inertia = inertial.inertia
        if inertia is None:
            _issue(
                issues,
                ERROR,
                "INERTIA_MISSING",
                "Link <inertial> is missing its <inertia> matrix",
                link=link.name,
            )
            continue
        missing = [name for name, value in inertia.components.items() if value is None]
        if missing:
            _issue(
                issues,
                ERROR,
                "INERTIA_COMPONENT_MISSING",
                "Inertia matrix is missing required components",
                link=link.name,
                components=missing,
            )
            continue
        matrix = inertia.as_matrix()
        assert matrix is not None
        if not np.isfinite(matrix).all():
            _issue(
                issues,
                ERROR,
                "INERTIA_NONFINITE",
                "Inertia matrix must contain only finite values",
                link=link.name,
            )
            continue
        diagonal = np.diag(matrix)
        if np.any(diagonal < 0.0):
            _issue(
                issues,
                ERROR,
                "INERTIA_DIAGONAL_NEGATIVE",
                "Inertia diagonal entries must be non-negative",
                link=link.name,
                diagonal=diagonal.tolist(),
            )
        eigenvalues = np.linalg.eigvalsh(matrix)
        if float(np.min(eigenvalues)) < -1.0e-10:
            _issue(
                issues,
                ERROR,
                "INERTIA_NOT_POSITIVE_SEMIDEFINITE",
                "Inertia matrix must be positive semidefinite",
                link=link.name,
                eigenvalues=eigenvalues.tolist(),
            )


def _check_joint_limits(joint: Joint, issues: list[InspectionIssue]) -> None:
    if joint.joint_type in {"revolute", "prismatic"}:
        if joint.limit is None:
            _issue(
                issues,
                ERROR,
                "JOINT_LIMIT_MISSING",
                "Revolute and prismatic joints require a <limit>",
                joint=joint.name,
                joint_type=joint.joint_type,
            )
            return
        if joint.limit.lower is None or joint.limit.upper is None:
            _issue(
                issues,
                ERROR,
                "JOINT_POSITION_LIMIT_MISSING",
                "Joint requires finite lower and upper position limits",
                joint=joint.name,
                lower=joint.limit.lower,
                upper=joint.limit.upper,
            )
        elif joint.limit.lower > joint.limit.upper:
            _issue(
                issues,
                ERROR,
                "JOINT_LIMIT_ORDER_INVALID",
                "Joint lower limit must not exceed upper limit",
                joint=joint.name,
                lower=joint.limit.lower,
                upper=joint.limit.upper,
            )
    if joint.joint_type in {"revolute", "continuous", "prismatic"}:
        if joint.limit is None or joint.limit.effort is None:
            _issue(
                issues,
                WARNING,
                "JOINT_EFFORT_LIMIT_MISSING",
                "Movable joint has no effort limit",
                joint=joint.name,
            )
        elif joint.limit.effort < 0.0:
            _issue(
                issues,
                ERROR,
                "JOINT_EFFORT_LIMIT_NEGATIVE",
                "Joint effort limit must be non-negative",
                joint=joint.name,
                effort=joint.limit.effort,
            )
        if joint.limit is None or joint.limit.velocity is None:
            _issue(
                issues,
                WARNING,
                "JOINT_VELOCITY_LIMIT_MISSING",
                "Movable joint has no velocity limit",
                joint=joint.name,
            )
        elif joint.limit.velocity < 0.0:
            _issue(
                issues,
                ERROR,
                "JOINT_VELOCITY_LIMIT_NEGATIVE",
                "Joint velocity limit must be non-negative",
                joint=joint.name,
                velocity=joint.limit.velocity,
            )


def _check_joints(model: URDFModel, issues: list[InspectionIssue]) -> None:
    child_to_joints: dict[str, list[str]] = {}
    for joint in model.joints.values():
        if joint.joint_type not in SUPPORTED_JOINT_TYPES:
            _issue(
                issues,
                ERROR,
                "JOINT_TYPE_UNSUPPORTED",
                "Joint type is not a standard URDF joint type",
                joint=joint.name,
                joint_type=joint.joint_type,
            )
        if joint.parent not in model.links:
            _issue(
                issues,
                ERROR,
                "JOINT_PARENT_LINK_NOT_FOUND",
                "Joint parent link does not exist",
                joint=joint.name,
                parent=joint.parent,
            )
        if joint.child not in model.links:
            _issue(
                issues,
                ERROR,
                "JOINT_CHILD_LINK_NOT_FOUND",
                "Joint child link does not exist",
                joint=joint.name,
                child=joint.child,
            )
        if joint.parent == joint.child:
            _issue(
                issues,
                ERROR,
                "JOINT_SELF_LOOP",
                "Joint parent and child links must differ",
                joint=joint.name,
                link=joint.parent,
            )
        child_to_joints.setdefault(joint.child, []).append(joint.name)
        if joint.joint_type in {"revolute", "continuous", "prismatic", "planar"}:
            axis = np.asarray(joint.axis, dtype=float)
            norm = float(np.linalg.norm(axis))
            if not np.isfinite(axis).all():
                _issue(
                    issues,
                    ERROR,
                    "JOINT_AXIS_NONFINITE",
                    "Movable joint axis must be finite",
                    joint=joint.name,
                    axis=axis.tolist(),
                )
            elif norm <= 1.0e-12:
                _issue(
                    issues,
                    ERROR,
                    "JOINT_AXIS_ZERO",
                    "Movable joint axis must be non-zero",
                    joint=joint.name,
                    axis=axis.tolist(),
                )
            elif not math.isclose(norm, 1.0, abs_tol=1.0e-6):
                _issue(
                    issues,
                    WARNING,
                    "JOINT_AXIS_NOT_UNIT",
                    "Joint axis is valid but not normalized",
                    joint=joint.name,
                    axis=axis.tolist(),
                    norm=norm,
                )
        _check_joint_limits(joint, issues)

    for child, joint_names in child_to_joints.items():
        if len(joint_names) > 1:
            _issue(
                issues,
                ERROR,
                "LINK_MULTIPLE_PARENTS",
                "URDF link is the child of multiple joints",
                link=child,
                joints=joint_names,
            )

    roots = model.root_link_names
    if len(roots) != 1:
        _issue(
            issues,
            ERROR,
            "ROOT_LINK_COUNT_INVALID",
            "A connected URDF tree must have exactly one root link",
            root_links=list(roots),
        )

    children: dict[str, list[str]] = {}
    for joint in model.joints.values():
        if joint.parent in model.links and joint.child in model.links:
            children.setdefault(joint.parent, []).append(joint.child)
    state: dict[str, int] = {}

    def visit(link_name: str, stack: list[str]) -> None:
        if state.get(link_name) == 1:
            start = stack.index(link_name) if link_name in stack else 0
            _issue(
                issues,
                ERROR,
                "URDF_KINEMATIC_CYCLE",
                "URDF joint graph contains a cycle",
                cycle=stack[start:] + [link_name],
            )
            return
        if state.get(link_name) == 2:
            return
        state[link_name] = 1
        for child in children.get(link_name, []):
            visit(child, stack + [link_name])
        state[link_name] = 2

    for link_name in model.links:
        if state.get(link_name, 0) == 0:
            visit(link_name, [])


def _check_meshes(
    model: URDFModel,
    issues: list[InspectionIssue],
    *,
    package_paths: Mapping[str, str | Path] | None,
    require_mesh_files: bool,
) -> tuple[MeshInspection, ...]:
    inspected: list[MeshInspection] = []
    for link in model.links.values():
        for mesh in link.meshes:
            scale = np.asarray(mesh.scale, dtype=float)
            if np.any(scale <= 0.0):
                _issue(
                    issues,
                    ERROR,
                    "MESH_SCALE_NONPOSITIVE",
                    "Mesh scale values must be positive",
                    link=mesh.link_name,
                    usage=mesh.usage,
                    filename=mesh.filename,
                    scale=scale.tolist(),
                )
            resolved = resolve_mesh_path(mesh, model.path, package_paths)
            exists = bool(resolved is not None and resolved.is_file())
            if not mesh.filename:
                _issue(
                    issues,
                    ERROR,
                    "MESH_FILENAME_MISSING",
                    "Mesh element is missing its filename",
                    link=mesh.link_name,
                    usage=mesh.usage,
                )
            elif resolved is None:
                _issue(
                    issues,
                    ERROR if require_mesh_files else WARNING,
                    "MESH_URI_UNRESOLVED",
                    "Mesh URI cannot be resolved locally; provide a package path for package:// URIs",
                    link=mesh.link_name,
                    usage=mesh.usage,
                    filename=mesh.filename,
                )
            elif require_mesh_files and not exists:
                _issue(
                    issues,
                    ERROR,
                    "MESH_FILE_NOT_FOUND",
                    "Resolved mesh path does not name an existing file",
                    link=mesh.link_name,
                    usage=mesh.usage,
                    filename=mesh.filename,
                    resolved_path=str(resolved),
                )
            elif not require_mesh_files and not exists:
                _issue(
                    issues,
                    WARNING,
                    "MESH_FILE_NOT_FOUND",
                    "Resolved mesh path does not name an existing file",
                    link=mesh.link_name,
                    usage=mesh.usage,
                    filename=mesh.filename,
                    resolved_path=str(resolved),
                )
            inspected.append(
                MeshInspection(
                    link_name=mesh.link_name,
                    usage=mesh.usage,
                    filename=mesh.filename,
                    scale=mesh.scale,
                    resolved_path=str(resolved) if resolved is not None else None,
                    exists=exists,
                )
            )
    return tuple(inspected)


def _is_descendant(model: URDFModel, ancestor: str, candidate: str) -> bool:
    children: dict[str, list[str]] = {}
    for joint in model.joints.values():
        children.setdefault(joint.parent, []).append(joint.child)
    pending = [ancestor]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current == candidate:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(children.get(current, []))
    return False


def _check_named_task_frames(
    model: URDFModel,
    issues: list[InspectionIssue],
    *,
    door_joint_name: str,
    door_link_name: str,
    handle_link_name: str,
) -> Joint | None:
    door_link = model.links.get(door_link_name)
    if door_link is None:
        _issue(
            issues,
            ERROR,
            "DOOR_LINK_NOT_FOUND",
            "Configured door link does not exist in the URDF",
            door_link=door_link_name,
        )
    handle_link = model.links.get(handle_link_name)
    if handle_link is None:
        _issue(
            issues,
            ERROR,
            "HANDLE_LINK_NOT_FOUND",
            "Configured handle link does not exist in the URDF",
            handle_link=handle_link_name,
        )
    door_joint = model.joints.get(door_joint_name)
    if door_joint is None:
        _issue(
            issues,
            ERROR,
            "DOOR_JOINT_NOT_FOUND",
            "Configured door joint does not exist in the URDF",
            door_joint=door_joint_name,
        )
        return None
    if door_joint.joint_type != "revolute":
        _issue(
            issues,
            ERROR,
            "DOOR_JOINT_TYPE_INVALID",
            "Microwave door joint must be revolute so finite lower/upper limits can be verified",
            door_joint=door_joint.name,
            joint_type=door_joint.joint_type,
        )
    if door_joint.child != door_link_name:
        _issue(
            issues,
            ERROR,
            "DOOR_JOINT_CHILD_MISMATCH",
            "Configured door link must be the child of the configured door joint",
            door_joint=door_joint.name,
            expected_child=door_link_name,
            actual_child=door_joint.child,
        )
    if door_link is not None and handle_link is not None and not _is_descendant(
        model, door_link_name, handle_link_name
    ):
        _issue(
            issues,
            ERROR,
            "HANDLE_NOT_ATTACHED_TO_DOOR",
            "Configured handle link is not the door link or one of its descendants",
            door_link=door_link_name,
            handle_link=handle_link_name,
        )
    return door_joint


def inspect_asset(
    urdf_path: str | Path,
    *,
    door_joint_name: str,
    door_link_name: str,
    handle_link_name: str,
    package_paths: Mapping[str, str | Path] | None = None,
    require_mesh_files: bool = True,
) -> InspectionReport:
    """Inspect a URDF and configured task frames, returning structured issues.

    Door joint, door link, and handle link are mandatory name arguments.  This
    makes asset-specific naming configuration explicit and prevents callers
    from relying on simulator-specific body or joint indices.
    """

    issues: list[InspectionIssue] = []
    try:
        model = load_urdf(urdf_path)
    except URDFModelError as exc:
        issue = InspectionIssue(ERROR, exc.code, exc.message, exc.context)
        return InspectionReport(
            urdf_path=str(Path(urdf_path).expanduser()),
            robot_name=None,
            link_names=(),
            joint_names=(),
            door_joint=None,
            meshes=(),
            issues=(issue,),
        )

    _check_link_inertials(model, issues)
    _check_joints(model, issues)
    meshes = _check_meshes(
        model,
        issues,
        package_paths=package_paths,
        require_mesh_files=require_mesh_files,
    )
    door_joint = _check_named_task_frames(
        model,
        issues,
        door_joint_name=door_joint_name,
        door_link_name=door_link_name,
        handle_link_name=handle_link_name,
    )
    return InspectionReport(
        urdf_path=str(model.path),
        robot_name=model.name,
        link_names=model.link_names,
        joint_names=model.joint_names,
        door_joint=_joint_summary(door_joint) if door_joint is not None else None,
        meshes=meshes,
        issues=tuple(issues),
    )


def _package_path_argument(value: str) -> tuple[str, Path]:
    package, separator, path = value.partition("=")
    if not separator or not package.strip() or not path.strip():
        raise argparse.ArgumentTypeError("package paths must use NAME=/absolute/or/relative/path")
    return package.strip(), Path(path).expanduser()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf", type=Path)
    parser.add_argument("--door-joint", required=True)
    parser.add_argument("--door-link", required=True)
    parser.add_argument("--handle-link", required=True)
    parser.add_argument(
        "--package-path",
        action="append",
        default=[],
        type=_package_path_argument,
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--allow-missing-meshes",
        action="store_true",
        help="downgrade unresolved or missing mesh files to warnings",
    )
    args = parser.parse_args(argv)
    package_paths = dict(args.package_path)
    report = inspect_asset(
        args.urdf,
        door_joint_name=args.door_joint,
        door_link_name=args.door_link,
        handle_link_name=args.handle_link,
        package_paths=package_paths,
        require_mesh_files=not args.allow_missing_meshes,
    )
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
