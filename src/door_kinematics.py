"""Name-based URDF forward kinematics for the articulated object."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .errors import FailureCode, PipelineError
from .transforms import compose_transforms, rpy_to_matrix
from .urdf_model import Joint, URDFModel


def _origin_transform(joint: Joint) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rpy_to_matrix(joint.origin.rpy)
    transform[:3, 3] = np.asarray(joint.origin.xyz, dtype=float)
    return transform


def _axis_angle_transform(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or norm <= 1.0e-12:
        raise PipelineError(
            FailureCode.ASSET_INVALID,
            "movable joint has an invalid axis",
            stage="object_fk",
            details={"axis": vector.tolist()},
        )
    x, y, z = vector / norm
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    one_minus = 1.0 - cosine
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(
        [
            [cosine + x * x * one_minus, x * y * one_minus - z * sine, x * z * one_minus + y * sine],
            [y * x * one_minus + z * sine, cosine + y * y * one_minus, y * z * one_minus - x * sine],
            [z * x * one_minus - y * sine, z * y * one_minus + x * sine, cosine + z * z * one_minus],
        ],
        dtype=float,
    )
    return transform


def _joint_motion(joint: Joint, coordinate: float) -> np.ndarray:
    if joint.joint_type == "fixed":
        return np.eye(4, dtype=float)
    if joint.joint_type in {"revolute", "continuous"}:
        return _axis_angle_transform(joint.axis, coordinate)
    if joint.joint_type == "prismatic":
        axis = np.asarray(joint.axis, dtype=float)
        norm = float(np.linalg.norm(axis))
        if not np.isfinite(axis).all() or norm <= 1.0e-12:
            raise PipelineError(
                FailureCode.ASSET_INVALID,
                f"joint {joint.name!r} has an invalid axis",
                stage="object_fk",
                details={"joint": joint.name, "axis": axis.tolist()},
            )
        transform = np.eye(4, dtype=float)
        transform[:3, 3] = axis / norm * float(coordinate)
        return transform
    raise PipelineError(
        FailureCode.ASSET_INVALID,
        f"joint type {joint.joint_type!r} is not supported by deterministic object FK",
        stage="object_fk",
        details={"joint": joint.name, "joint_type": joint.joint_type},
    )


def forward_kinematics(
    model: URDFModel,
    root_world_transform: np.ndarray,
    joint_positions: Mapping[str, float],
) -> dict[str, np.ndarray]:
    """Return ``T_world_link`` for every link in a connected URDF tree."""

    roots = model.root_link_names
    if len(roots) != 1:
        raise PipelineError(
            FailureCode.ASSET_INVALID,
            "object URDF must have exactly one root link",
            stage="object_fk",
            details={"root_links": list(roots)},
        )
    children: dict[str, list[Joint]] = {}
    for joint in model.joints.values():
        children.setdefault(joint.parent, []).append(joint)
    for joints in children.values():
        joints.sort(key=lambda item: item.name)

    transforms = {roots[0]: np.asarray(root_world_transform, dtype=float).copy()}
    pending = [roots[0]]
    while pending:
        parent = pending.pop()
        for joint in children.get(parent, []):
            coordinate = float(joint_positions.get(joint.name, 0.0))
            if not math.isfinite(coordinate):
                raise PipelineError(
                    FailureCode.NUMERICAL_INSTABILITY,
                    f"joint {joint.name!r} coordinate is not finite",
                    stage="object_fk",
                    details={"joint": joint.name, "coordinate": coordinate},
                )
            transforms[joint.child] = compose_transforms(
                transforms[parent],
                _origin_transform(joint),
                _joint_motion(joint, coordinate),
            )
            pending.append(joint.child)

    missing = sorted(set(model.links) - set(transforms))
    if missing:
        raise PipelineError(
            FailureCode.ASSET_INVALID,
            "object URDF contains links unreachable from its root",
            stage="object_fk",
            details={"missing_links": missing},
        )
    return transforms


@dataclass(frozen=True)
class DoorKinematics:
    """Object FK wrapper whose asset-specific names come only from config."""

    model: URDFModel
    root_world_transform: np.ndarray
    door_joint_name: str
    door_link_name: str
    handle_link_name: str

    def __post_init__(self) -> None:
        door_joint = self.model.require_joint(self.door_joint_name)
        self.model.require_link(self.door_link_name)
        self.model.require_link(self.handle_link_name)
        if door_joint.joint_type != "revolute" or door_joint.child != self.door_link_name:
            raise PipelineError(
                FailureCode.ASSET_INVALID,
                "configured door joint/link do not form a revolute door",
                stage="object_fk",
                details={
                    "door_joint": self.door_joint_name,
                    "door_link": self.door_link_name,
                    "joint_type": door_joint.joint_type,
                    "actual_child": door_joint.child,
                },
            )

    @property
    def limits(self) -> tuple[float, float]:
        joint = self.model.require_joint(self.door_joint_name)
        if joint.limit is None or joint.limit.lower is None or joint.limit.upper is None:
            raise PipelineError(
                FailureCode.ASSET_INVALID,
                "configured door joint has no finite lower/upper limit",
                stage="object_fk",
                details={"door_joint": self.door_joint_name},
            )
        return float(joint.limit.lower), float(joint.limit.upper)

    def link_transforms(self, door_angle_rad: float) -> dict[str, np.ndarray]:
        lower, upper = self.limits
        tolerance = 1.0e-9
        if not lower - tolerance <= door_angle_rad <= upper + tolerance:
            raise PipelineError(
                FailureCode.JOINT_LIMIT,
                "requested door angle is outside the URDF limits",
                stage="object_fk",
                details={
                    "door_joint": self.door_joint_name,
                    "angle_rad": float(door_angle_rad),
                    "lower": lower,
                    "upper": upper,
                },
            )
        return forward_kinematics(
            self.model,
            self.root_world_transform,
            {self.door_joint_name: float(door_angle_rad)},
        )

    def handle_frame_transform(self, door_angle_rad: float, link_to_frame: np.ndarray) -> np.ndarray:
        link_world = self.link_transforms(door_angle_rad)[self.handle_link_name]
        return compose_transforms(link_world, link_to_frame)

