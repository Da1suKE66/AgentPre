"""Deterministic six-phase door-opening target generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .config import PHASE_ORDER
from .door_kinematics import DoorKinematics
from .transforms import (
    compose_transforms,
    decompose_pose,
    matrix_to_quaternion,
    pose_matrix,
    quaternion_to_matrix,
)


def _slerp_wxyz(start: np.ndarray, end: np.ndarray, amount: float) -> np.ndarray:
    first = np.asarray(start, dtype=float)
    second = np.asarray(end, dtype=float)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = first + amount * (second - first)
        return result / np.linalg.norm(result)
    angle = np.arccos(dot)
    sine = np.sin(angle)
    return (np.sin((1.0 - amount) * angle) / sine) * first + (np.sin(amount * angle) / sine) * second


def interpolate_transform(start: np.ndarray, end: np.ndarray, amount: float) -> np.ndarray:
    start_position, start_quaternion = decompose_pose(start)
    end_position, end_quaternion = decompose_pose(end)
    position = (1.0 - amount) * start_position + amount * end_position
    quaternion = _slerp_wxyz(start_quaternion, end_quaternion, float(amount))
    return pose_matrix(position, quaternion)


def _phase_amount(index: int, count: int) -> float:
    """Return C2-continuous progress for a phase sample.

    Phase samples include the phase endpoint but not its explicit start.  The
    preceding phase's final sample supplies that start, so evaluating the
    quintic smoothstep at ``(index + 1) / count`` preserves the shared endpoint
    while giving every phase zero velocity and acceleration at both ends.
    """

    linear_amount = float(index + 1) / float(count)
    linear_amount = float(np.clip(linear_amount, 0.0, 1.0))
    return linear_amount**3 * (
        10.0 + linear_amount * (-15.0 + 6.0 * linear_amount)
    )


@dataclass(frozen=True)
class TaskPlan:
    phase_names: np.ndarray
    phase_indices: np.ndarray
    time_s: np.ndarray
    door_angle_rad: np.ndarray
    handle_world: np.ndarray
    target_gripper_world: np.ndarray
    gripper_width_m: np.ndarray

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "phase_names": self.phase_names,
            "phase_indices": self.phase_indices,
            "time_s": self.time_s,
            "door_angle_rad": self.door_angle_rad,
            "handle_world": self.handle_world,
            "target_gripper_world": self.target_gripper_world,
            "gripper_width_m": self.gripper_width_m,
        }


def generate_task_plan(
    *,
    kinematics: DoorKinematics,
    link_to_handle_frame: np.ndarray,
    handle_approach_axis: np.ndarray,
    handle_to_gripper: np.ndarray,
    initial_gripper_world: np.ndarray,
    closed_angle_rad: float,
    goal_angle_rad: float,
    phase_samples: Mapping[str, int],
    pregrasp_distance_m: float,
    retreat_distance_m: float,
    open_gripper_width_m: float,
    closed_gripper_width_m: float,
    dt: float,
) -> TaskPlan:
    """Build pregrasp, approach, close, actuate, release, and retreat targets.

    ``handle_approach_axis`` is expressed in the handle frame and points in the
    direction travelled from pregrasp toward the handle. Therefore pregrasp
    and retreat offsets use its negative direction.
    """

    if tuple(phase_samples) != PHASE_ORDER:
        missing = [phase for phase in PHASE_ORDER if phase not in phase_samples]
        extra = [phase for phase in phase_samples if phase not in PHASE_ORDER]
        if missing or extra:
            raise ValueError(f"phase_samples must contain exactly {PHASE_ORDER}; missing={missing}, extra={extra}")
    axis = np.asarray(handle_approach_axis, dtype=float)
    axis /= np.linalg.norm(axis)

    phase_names: list[str] = []
    door_angles: list[float] = []
    handle_world: list[np.ndarray] = []
    gripper_world: list[np.ndarray] = []
    widths: list[float] = []

    def add(phase: str, angle: float, target: np.ndarray, width: float) -> None:
        phase_names.append(phase)
        door_angles.append(float(angle))
        handle_world.append(kinematics.handle_frame_transform(float(angle), link_to_handle_frame))
        gripper_world.append(np.asarray(target, dtype=float).copy())
        widths.append(float(width))

    closed_handle = kinematics.handle_frame_transform(closed_angle_rad, link_to_handle_frame)
    closed_grasp = compose_transforms(closed_handle, handle_to_gripper)
    closed_axis_world = closed_handle[:3, :3] @ axis
    pregrasp = closed_grasp.copy()
    pregrasp[:3, 3] -= closed_axis_world * float(pregrasp_distance_m)

    count = int(phase_samples["pregrasp"])
    for index in range(count):
        amount = _phase_amount(index, count)
        add(
            "pregrasp",
            closed_angle_rad,
            interpolate_transform(initial_gripper_world, pregrasp, amount),
            open_gripper_width_m,
        )

    count = int(phase_samples["approach"])
    for index in range(count):
        amount = _phase_amount(index, count)
        add(
            "approach",
            closed_angle_rad,
            interpolate_transform(pregrasp, closed_grasp, amount),
            open_gripper_width_m,
        )

    count = int(phase_samples["close"])
    for index in range(count):
        amount = _phase_amount(index, count)
        width = (1.0 - amount) * open_gripper_width_m + amount * closed_gripper_width_m
        add("close", closed_angle_rad, closed_grasp, width)

    count = int(phase_samples["actuate"])
    for index in range(count):
        amount = _phase_amount(index, count)
        angle = (1.0 - amount) * closed_angle_rad + amount * goal_angle_rad
        handle = kinematics.handle_frame_transform(angle, link_to_handle_frame)
        add("actuate", angle, compose_transforms(handle, handle_to_gripper), closed_gripper_width_m)

    final_handle = kinematics.handle_frame_transform(goal_angle_rad, link_to_handle_frame)
    final_grasp = compose_transforms(final_handle, handle_to_gripper)

    count = int(phase_samples["release"])
    for index in range(count):
        amount = _phase_amount(index, count)
        width = (1.0 - amount) * closed_gripper_width_m + amount * open_gripper_width_m
        add("release", goal_angle_rad, final_grasp, width)

    final_axis_world = final_handle[:3, :3] @ axis
    retreat = final_grasp.copy()
    retreat[:3, 3] -= final_axis_world * float(retreat_distance_m)
    count = int(phase_samples["retreat"])
    for index in range(count):
        amount = _phase_amount(index, count)
        add(
            "retreat",
            goal_angle_rad,
            interpolate_transform(final_grasp, retreat, amount),
            open_gripper_width_m,
        )

    names = np.asarray(phase_names, dtype="U16")
    indices = np.asarray([PHASE_ORDER.index(name) for name in phase_names], dtype=np.int16)
    return TaskPlan(
        phase_names=names,
        phase_indices=indices,
        # Every stored target is the right endpoint of one control interval;
        # the explicit t=0 state is the configured nominal robot pose.
        time_s=(np.arange(len(names), dtype=float) + 1.0) * float(dt),
        door_angle_rad=np.asarray(door_angles, dtype=float),
        handle_world=np.stack(handle_world),
        target_gripper_world=np.stack(gripper_world),
        gripper_width_m=np.asarray(widths, dtype=float),
    )
