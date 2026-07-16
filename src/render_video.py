"""Render measured Newton body poses as a real-geometry H.264 video.

Unlike :mod:`src.animation`, this exporter keeps every URDF visual shape and
uses Newton's CUDA ray-traced tiled camera.  The physics trajectory remains the
sole motion source: no FK reconstruction or keyframe interpolation is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

import numpy as np


class VideoRenderError(RuntimeError):
    """A deterministic mesh-video render failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VideoRenderError(f"cannot read JSON object: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise VideoRenderError(f"JSON root must be an object: {path}")
    return payload


def _strings(values: np.ndarray, *, require_unique: bool) -> list[str]:
    result: list[str] = []
    for value in np.asarray(values).tolist():
        if isinstance(value, bytes):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    if any(not value for value in result):
        raise VideoRenderError("string arrays must contain non-empty values")
    if require_unique and len(set(result)) != len(result):
        raise VideoRenderError("body_labels must be unique")
    return result


def sample_frame_indices(
    frame_count: int,
    source_dt_s: float,
    output_fps: float,
) -> tuple[np.ndarray, int]:
    """Return a deterministic integer-stride downsample and its source stride."""

    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
        raise VideoRenderError("frame_count must be a positive integer")
    if not math.isfinite(source_dt_s) or source_dt_s <= 0.0:
        raise VideoRenderError("source_dt_s must be finite and positive")
    if not math.isfinite(output_fps) or output_fps <= 0.0:
        raise VideoRenderError("output_fps must be finite and positive")
    source_fps = 1.0 / float(source_dt_s)
    stride = max(1, int(round(source_fps / float(output_fps))))
    actual_fps = source_fps / stride
    if not math.isclose(actual_fps, output_fps, rel_tol=0.01, abs_tol=0.01):
        raise VideoRenderError(
            f"output_fps={output_fps:g} is not an integer-stride sampling of "
            f"source_fps={source_fps:g}"
        )
    indices = np.arange(0, frame_count, stride, dtype=np.int64)
    if indices.size == 0:
        raise AssertionError("positive frame count produced no sampled frames")
    return indices, stride


def unpack_rgb(packed_rgba: np.ndarray) -> np.ndarray:
    """Convert Newton's uint32 packed RGBA image into contiguous RGB24."""

    packed = np.asarray(packed_rgba)
    if packed.ndim != 2 or packed.dtype != np.uint32:
        raise VideoRenderError(
            f"packed camera output must be a 2D uint32 array, got {packed.shape}/{packed.dtype}"
        )
    rgb = np.empty(packed.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = (packed & 0xFF).astype(np.uint8)
    rgb[..., 1] = ((packed >> 8) & 0xFF).astype(np.uint8)
    rgb[..., 2] = ((packed >> 16) & 0xFF).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def _pose_transform(wp: Any, pose: dict[str, Any]) -> Any:
    position = np.asarray(pose["position"], dtype=float)
    quaternion = np.asarray(pose["orientation_wxyz"], dtype=float)
    if position.shape != (3,) or quaternion.shape != (4,):
        raise VideoRenderError("asset world pose must contain xyz and wxyz")
    return wp.transform(
        wp.vec3(*position.tolist()),
        wp.quat(quaternion[1], quaternion[2], quaternion[3], quaternion[0]),
    )


def _camera_array(
    wp: Any,
    *,
    eye: np.ndarray,
    target: np.ndarray,
    device: str,
) -> Any:
    from .transforms import matrix_to_quaternion

    forward = target - eye
    norm = float(np.linalg.norm(forward))
    if norm <= 1.0e-9:
        raise VideoRenderError("camera eye and target must differ")
    forward /= norm
    zaxis = -forward
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=float)
    xaxis = np.cross(world_up, zaxis)
    xnorm = float(np.linalg.norm(xaxis))
    if xnorm <= 1.0e-9:
        raise VideoRenderError("camera direction must not be parallel to world up")
    xaxis /= xnorm
    yaxis = np.cross(zaxis, xaxis)
    quaternion_wxyz = matrix_to_quaternion(
        np.column_stack((xaxis, yaxis, zaxis))
    )
    camera = wp.transformf(
        wp.vec3f(*eye.tolist()),
        wp.quatf(
            quaternion_wxyz[1],
            quaternion_wxyz[2],
            quaternion_wxyz[3],
            quaternion_wxyz[0],
        ),
    )
    return wp.array([[camera]], dtype=wp.transformf, device=device)


def _ffmpeg_process(
    *, output: Path, width: int, height: int, fps: float, ffmpeg: str
) -> subprocess.Popen[bytes]:
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        format(float(fps), ".12g"),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    try:
        return subprocess.Popen(command, stdin=subprocess.PIPE)
    except OSError as exc:
        raise VideoRenderError(f"cannot start ffmpeg: {exc}") from exc


def _write_png(
    rgb: np.ndarray,
    path: Path,
    *,
    ffmpeg: str,
) -> None:
    height, width, channels = rgb.shape
    if channels != 3:
        raise VideoRenderError("preview RGB image must have three channels")
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-i",
        "-",
        "-frames:v",
        "1",
        str(path),
    ]
    completed = subprocess.run(command, input=rgb.tobytes(order="C"), check=False)
    if completed.returncode != 0 or not path.is_file():
        raise VideoRenderError(f"ffmpeg failed to write preview PNG: {path}")


def _contact_sheet(
    first: Path,
    second: Path,
    output: Path,
    *,
    ffmpeg: str,
) -> None:
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(first),
        "-i",
        str(second),
        "-filter_complex",
        "[0:v]scale=640:-2[a];[1:v]scale=640:-2[b];[a][b]hstack=inputs=2",
        "-frames:v",
        "1",
        str(output),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0 or not output.is_file():
        raise VideoRenderError(f"ffmpeg failed to write contact sheet: {output}")


def render_video(
    *,
    workspace: Path,
    output: Path,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    device: str = "cuda:0",
    eye: Sequence[float] = (1.55, -1.85, 1.35),
    target: Sequence[float] = (0.35, -0.05, 0.55),
    vertical_fov_deg: float = 45.0,
    keyframes: Sequence[int] = (0, 391, 1191, 1335),
    ffmpeg: str = "ffmpeg",
) -> dict[str, Any]:
    """Render the frozen physics trajectory and return auditable metadata."""

    if width < 64 or height < 64 or width % 2 or height % 2:
        raise VideoRenderError("video width and height must be even integers >= 64")
    if not math.isfinite(vertical_fov_deg) or not 1.0 < vertical_fov_deg < 179.0:
        raise VideoRenderError("vertical_fov_deg must be between 1 and 179 degrees")

    workspace = workspace.expanduser().resolve()
    output = output.expanduser().resolve()
    config_path = workspace / "agent_config.json"
    trajectory_path = (
        workspace
        / "attempts"
        / "kinematic_000"
        / "physics_assisted"
        / "trajectory.npz"
    )
    if not trajectory_path.is_file():
        candidates = sorted(workspace.glob("attempts/*/physics_assisted/trajectory.npz"))
        if len(candidates) != 1:
            raise VideoRenderError(
                "workspace must contain exactly one physics-assisted trajectory"
            )
        trajectory_path = candidates[0]
    config = _read_json(config_path)

    with np.load(trajectory_path, allow_pickle=False) as archive:
        required = {"body_pose_wxyz", "body_labels", "time_s"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise VideoRenderError(f"trajectory is missing arrays: {missing}")
        body_poses = np.asarray(archive["body_pose_wxyz"], dtype=np.float64)
        recorded_labels = _strings(archive["body_labels"], require_unique=True)
        time_s = np.asarray(archive["time_s"], dtype=np.float64)
        phase_names = (
            _strings(archive["phase_names"], require_unique=False)
            if "phase_names" in archive.files
            else ["trajectory"] * len(time_s)
        )
        door_angle = (
            np.asarray(archive["door_angle_rad"], dtype=np.float64)
            if "door_angle_rad" in archive.files
            else None
        )
    if body_poses.ndim != 3 or body_poses.shape[1:] != (len(recorded_labels), 7):
        raise VideoRenderError(
            "body_pose_wxyz must have shape (frames, body_labels, 7)"
        )
    if time_s.shape != (body_poses.shape[0],) or len(phase_names) != body_poses.shape[0]:
        raise VideoRenderError("trajectory time/phase arrays do not match body poses")
    if body_poses.shape[0] < 2 or not np.isfinite(body_poses).all():
        raise VideoRenderError("trajectory must contain at least two finite frames")
    differences = np.diff(time_s)
    if not np.isfinite(differences).all() or np.any(differences <= 0.0):
        raise VideoRenderError("time_s must be finite and strictly increasing")
    source_dt_s = float(np.median(differences))
    sampled, stride = sample_frame_indices(body_poses.shape[0], source_dt_s, fps)

    # src.__init__ keeps simulation entrypoints CPU-safe.  Restore only the
    # explicitly selected render device before Warp/Newton initialize.
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get(
        "AGENTPRE_CUDA_VISIBLE_DEVICES", "0"
    )
    import newton
    import warp as wp
    from newton.sensors import SensorTiledCamera

    from .physics_assisted import plan_massless_fixed_joint_collapse

    wp.set_device(device)
    assets = config["assets"]
    robot = assets["robot"]
    object_asset = assets["object"]
    builder = newton.ModelBuilder()
    builder.add_urdf(
        robot["urdf"],
        xform=_pose_transform(wp, robot["world_pose"]),
        floating=False,
        hide_visuals=False,
        parse_visuals_as_colliders=False,
        enable_self_collisions=False,
        collapse_fixed_joints=False,
        collapse_massless_fixed_root=False,
        override_root_xform=True,
    )
    collapsed, joints_to_keep = plan_massless_fixed_joint_collapse(
        builder.joint_label,
        builder.joint_type,
        builder.joint_parent,
        builder.joint_child,
        builder.body_mass,
        fixed_joint_type=int(newton.JointType.FIXED),
    )
    if collapsed:
        builder.collapse_fixed_joints(joints_to_keep=joints_to_keep)
    builder.add_urdf(
        object_asset["urdf"],
        xform=_pose_transform(wp, object_asset["world_pose"]),
        floating=False,
        hide_visuals=False,
        parse_visuals_as_colliders=False,
        enable_self_collisions=False,
        collapse_fixed_joints=False,
        collapse_massless_fixed_root=False,
        override_root_xform=True,
    )
    builder.add_ground_plane(color=(0.28, 0.30, 0.34))
    render_labels = [str(value) for value in builder.body_label]
    if set(render_labels) != set(recorded_labels) or len(render_labels) != len(recorded_labels):
        raise VideoRenderError(
            "render-model body labels do not exactly match the measured trajectory"
        )
    record_index = {name: index for index, name in enumerate(recorded_labels)}

    model = builder.finalize(device=device)
    state = model.state()
    camera_eye = np.asarray(eye, dtype=float)
    camera_target = np.asarray(target, dtype=float)
    if camera_eye.shape != (3,) or camera_target.shape != (3,):
        raise VideoRenderError("camera eye and target must each contain three values")
    cameras = _camera_array(
        wp, eye=camera_eye, target=camera_target, device=device
    )
    sensor = SensorTiledCamera(model=model)
    sensor.utils.create_default_light(enable_shadows=True)
    rays = sensor.utils.compute_pinhole_camera_rays(
        width, height, math.radians(vertical_fov_deg)
    )
    color = sensor.utils.create_color_image_output(width, height, camera_count=1)

    def set_frame(frame_index: int) -> None:
        poses = body_poses[frame_index]
        rows = []
        for name in render_labels:
            pose = poses[record_index[name]]
            rows.append(
                wp.transformf(
                    wp.vec3f(*pose[:3].tolist()),
                    wp.quatf(pose[4], pose[5], pose[6], pose[3]),
                )
            )
        state.body_q.assign(wp.array(rows, dtype=wp.transformf, device=device))

    def render_frame(frame_index: int) -> np.ndarray:
        set_frame(frame_index)
        # Shape BVHs are built for the finalize-time state.  Refit after every
        # measured body-pose injection or moving links render at stale poses.
        model.bvh_refit_shapes(state)
        sensor.update(state, cameras, rays, color_image=color)
        packed = np.asarray(color.numpy()[0, 0], dtype=np.uint32)
        return unpack_rgb(packed)

    output.parent.mkdir(parents=True, exist_ok=True)
    process = _ffmpeg_process(
        output=output, width=width, height=height, fps=fps, ffmpeg=ffmpeg
    )
    assert process.stdin is not None
    try:
        for rendered_index, frame_index in enumerate(sampled.tolist()):
            process.stdin.write(render_frame(int(frame_index)).tobytes(order="C"))
            if rendered_index % 60 == 0:
                print(
                    json.dumps(
                        {
                            "event": "render_progress",
                            "rendered": rendered_index + 1,
                            "total": int(sampled.size),
                            "source_frame": int(frame_index),
                        }
                    ),
                    flush=True,
                )
    except Exception:
        process.stdin.close()
        process.wait()
        raise
    process.stdin.close()
    return_code = process.wait()
    if return_code != 0 or not output.is_file() or output.stat().st_size == 0:
        raise VideoRenderError(f"ffmpeg failed with exit code {return_code}")

    preview_dir = output.parent / f"{output.stem}_frames"
    rendered_keyframes: list[dict[str, Any]] = []
    for requested in keyframes:
        index = int(requested)
        if not 0 <= index < body_poses.shape[0]:
            raise VideoRenderError(f"keyframe is outside the trajectory: {index}")
        preview_path = preview_dir / f"frame_{index:04d}.png"
        _write_png(render_frame(index), preview_path, ffmpeg=ffmpeg)
        rendered_keyframes.append(
            {
                "frame": index,
                "time_s": float(time_s[index]),
                "phase": phase_names[index],
                "door_angle_deg": (
                    None if door_angle is None else float(np.rad2deg(door_angle[index]))
                ),
                "path": str(preview_path),
                "sha256": _sha256(preview_path),
            }
        )
    if len({item["sha256"] for item in rendered_keyframes}) < 2:
        raise VideoRenderError(
            "all rendered keyframes are pixel-identical; moving-shape replay is stale"
        )
    contact_sheet = output.parent / f"{output.stem}_closed_open.png"
    _contact_sheet(
        Path(rendered_keyframes[0]["path"]),
        Path(rendered_keyframes[-2]["path"]),
        contact_sheet,
        ffmpeg=ffmpeg,
    )

    metadata = {
        "schema_version": 1,
        "renderer": "newton.SensorTiledCamera",
        "motion_source": "measured body_pose_wxyz; no FK reconstruction or interpolation",
        "workspace": str(workspace),
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "trajectory": {
            "path": str(trajectory_path),
            "sha256": _sha256(trajectory_path),
            "source_frame_count": int(body_poses.shape[0]),
            "source_dt_s": source_dt_s,
            "source_fps": 1.0 / source_dt_s,
            "body_count": len(recorded_labels),
            "body_labels": recorded_labels,
        },
        "video": {
            "path": str(output),
            "sha256": _sha256(output),
            "codec": "H.264/libx264",
            "pixel_format": "yuv420p",
            "width": width,
            "height": height,
            "fps": float(fps),
            "frame_stride": stride,
            "rendered_frame_count": int(sampled.size),
            "duration_s": float(sampled.size / fps),
        },
        "camera": {
            "eye": camera_eye.tolist(),
            "target": camera_target.tolist(),
            "vertical_fov_deg": float(vertical_fov_deg),
        },
        "keyframes": rendered_keyframes,
        "closed_open_contact_sheet": {
            "path": str(contact_sheet),
            "sha256": _sha256(contact_sheet),
        },
    }
    metadata_path = output.with_suffix(".render.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata["metadata_path"] = str(metadata_path)
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def _vector3(text: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(value.strip()) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated x,y,z") from exc
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("expected three finite comma-separated values")
    return values  # type: ignore[return-value]


def _integer_list(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated frame indices") from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one keyframe is required")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eye", type=_vector3, default=(1.55, -1.85, 1.35))
    parser.add_argument("--target", type=_vector3, default=(0.35, -0.05, 0.55))
    parser.add_argument("--vertical-fov-deg", type=float, default=45.0)
    parser.add_argument("--keyframes", type=_integer_list, default=(0, 391, 1191, 1335))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        render_video(
            workspace=args.workdir,
            output=args.output,
            width=args.width,
            height=args.height,
            fps=args.fps,
            device=args.device,
            eye=args.eye,
            target=args.target,
            vertical_fov_deg=args.vertical_fov_deg,
            keyframes=args.keyframes,
            ffmpeg=args.ffmpeg,
        )
    except VideoRenderError as exc:
        print(json.dumps({"success": False, "error": str(exc)}), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VideoRenderError",
    "build_parser",
    "main",
    "render_video",
    "sample_frame_indices",
    "unpack_rgb",
]
