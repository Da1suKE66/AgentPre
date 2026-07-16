"""Export an AgentPre trajectory as a self-contained interactive animation.

The exporter deliberately depends only on NumPy and the Python standard library.
It consumes either a physics trajectory (``body_pose_wxyz`` plus
``body_labels``) or a kinematic trajectory containing TCP/handle transforms and
door angles.  The generated document embeds all data and needs no web server.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class AnimationExportError(ValueError):
    """Raised when a trajectory cannot be represented by the exporter."""


def _finite_array(name: str, value: Any, *, ndim: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise AnimationExportError(f"{name} must be numeric") from exc
    if ndim is not None and array.ndim != ndim:
        raise AnimationExportError(
            f"{name} must have {ndim} dimensions, got shape {array.shape}"
        )
    if not np.isfinite(array).all():
        raise AnimationExportError(f"{name} contains NaN or Infinity")
    return array


def _pose_positions(name: str, value: Any) -> np.ndarray:
    """Return xyz rows from either xyz_wxyz poses or 4x4 transforms."""

    array = _finite_array(name, value)
    if array.ndim == 2 and array.shape[1] == 7:
        return array[:, :3]
    if array.ndim == 3 and array.shape[1:] == (4, 4):
        return array[:, :3, 3]
    raise AnimationExportError(
        f"{name} must have shape (frames, 7) or (frames, 4, 4), got {array.shape}"
    )


def _labels(value: Any, count: int) -> list[str]:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != count:
        raise AnimationExportError(
            f"body_labels must contain one label per body (expected {count})"
        )
    labels: list[str] = []
    for raw in array.tolist():
        if isinstance(raw, bytes):
            try:
                label = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise AnimationExportError("body_labels must be UTF-8") from exc
        else:
            label = str(raw)
        if not label:
            raise AnimationExportError("body_labels cannot contain empty values")
        labels.append(label)
    return labels


def _project_xyz(points: np.ndarray) -> np.ndarray:
    """Use a stable isometric projection for a compact, dependency-free view."""

    projected = np.empty(points.shape[:-1] + (2,), dtype=float)
    projected[..., 0] = (points[..., 0] - points[..., 1]) * 0.7071067811865476
    projected[..., 1] = points[..., 2] - 0.35 * (
        points[..., 0] + points[..., 1]
    )
    return projected


def _rounded(value: np.ndarray, digits: int = 5) -> list[Any]:
    return np.round(np.asarray(value, dtype=float), digits).tolist()


def _classify_bodies(labels: Sequence[str]) -> dict[str, list[int]]:
    robot: list[int] = []
    object_bodies: list[int] = []
    highlighted: list[int] = []
    for index, label in enumerate(labels):
        lowered = label.lower()
        if any(token in lowered for token in ("robot/", "franka", "panda")):
            robot.append(index)
        else:
            object_bodies.append(index)
        if any(
            token in lowered
            for token in ("handle", "grip", "hand", "door", "wrist")
        ):
            highlighted.append(index)
    return {
        "robot": robot,
        "object": object_bodies,
        "highlighted": highlighted,
    }


def load_animation_data(
    trajectory_path: str | Path,
    *,
    fallback_fps: float = 60.0,
) -> dict[str, Any]:
    """Load and compact a trajectory for browser playback.

    Physics body poses are preferred whenever both ``body_pose_wxyz`` and
    ``body_labels`` are available.  Kinematic archives fall back to
    ``achieved_gripper_world``/``target_gripper_world``, ``handle_world``, and
    ``door_angle_rad``.
    """

    path = Path(trajectory_path)
    if not np.isfinite(fallback_fps) or fallback_fps <= 0.0:
        raise AnimationExportError("fallback_fps must be finite and positive")
    try:
        archive_context = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise AnimationExportError(f"cannot read trajectory archive: {path}") from exc

    with archive_context as archive:
        names = set(archive.files)
        has_body_pose = "body_pose_wxyz" in names
        has_body_labels = "body_labels" in names
        if has_body_pose != has_body_labels:
            raise AnimationExportError(
                "physics animation requires body_pose_wxyz and body_labels together"
            )

        body_xyz: np.ndarray | None = None
        body_labels: list[str] = []
        if has_body_pose:
            poses = _finite_array("body_pose_wxyz", archive["body_pose_wxyz"], ndim=3)
            if poses.shape[0] < 1 or poses.shape[1] < 1 or poses.shape[2] != 7:
                raise AnimationExportError(
                    "body_pose_wxyz must have shape (frames, bodies, 7)"
                )
            body_xyz = poses[..., :3]
            body_labels = _labels(archive["body_labels"], poses.shape[1])

        tcp_xyz: np.ndarray | None = None
        tcp_name: str | None = None
        for candidate in (
            "ee_pose_wxyz",
            "achieved_gripper_world",
            "target_gripper_world",
        ):
            if candidate in names:
                tcp_xyz = _pose_positions(candidate, archive[candidate])
                tcp_name = candidate
                break

        handle_xyz: np.ndarray | None = None
        handle_name: str | None = None
        for candidate in ("handle_link_pose_wxyz", "handle_world"):
            if candidate in names:
                handle_xyz = _pose_positions(candidate, archive[candidate])
                handle_name = candidate
                break

        door_angle: np.ndarray | None = None
        if "door_angle_rad" in names:
            door_angle = _finite_array(
                "door_angle_rad", archive["door_angle_rad"], ndim=1
            )

        frame_sources = [
            value
            for value in (body_xyz, tcp_xyz, handle_xyz, door_angle)
            if value is not None
        ]
        if not frame_sources:
            raise AnimationExportError(
                "trajectory has no physics bodies, TCP, handle, or door-angle data"
            )
        frame_count = int(frame_sources[0].shape[0])
        if frame_count < 1 or any(value.shape[0] != frame_count for value in frame_sources):
            raise AnimationExportError("trajectory arrays have inconsistent frame counts")

        if "time_s" in names:
            time_s = _finite_array("time_s", archive["time_s"], ndim=1)
            if time_s.shape != (frame_count,):
                raise AnimationExportError("time_s does not match the trajectory frames")
            if np.any(np.diff(time_s) < 0.0):
                raise AnimationExportError("time_s must be monotonically nondecreasing")
            time_s = time_s - time_s[0]
        else:
            time_s = np.arange(frame_count, dtype=float) / float(fallback_fps)

        if "phase_names" in names:
            raw_phases = np.asarray(archive["phase_names"])
            if raw_phases.shape != (frame_count,):
                raise AnimationExportError("phase_names does not match the trajectory frames")
            phases = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in raw_phases.tolist()
            ]
        else:
            phases = ["trajectory"] * frame_count

    body_xy = _project_xyz(body_xyz) if body_xyz is not None else None
    tcp_xy = _project_xyz(tcp_xyz) if tcp_xyz is not None else None
    handle_xy = _project_xyz(handle_xyz) if handle_xyz is not None else None

    visible_arrays = [
        value.reshape(-1, 2)
        for value in (body_xy, tcp_xy, handle_xy)
        if value is not None
    ]
    if visible_arrays:
        all_points = np.concatenate(visible_arrays, axis=0)
        lower = np.min(all_points, axis=0)
        upper = np.max(all_points, axis=0)
        extent = np.maximum(upper - lower, 0.1)
        lower -= 0.08 * extent
        upper += 0.08 * extent
    else:
        lower = np.asarray([-1.0, -1.0])
        upper = np.asarray([1.0, 1.0])

    tcp_handle_cm: np.ndarray | None = None
    if tcp_xyz is not None and handle_xyz is not None:
        tcp_handle_cm = 100.0 * np.linalg.norm(tcp_xyz - handle_xyz, axis=1)

    return {
        "version": 1,
        "source": "physics_body_pose" if body_xyz is not None else "kinematic",
        "frameCount": frame_count,
        "durationS": round(float(time_s[-1]), 6),
        "timeS": _rounded(time_s, 6),
        "phases": phases,
        "bodyLabels": body_labels,
        "bodyGroups": _classify_bodies(body_labels),
        "bodyXY": _rounded(body_xy) if body_xy is not None else None,
        "tcpXY": _rounded(tcp_xy) if tcp_xy is not None else None,
        "handleXY": _rounded(handle_xy) if handle_xy is not None else None,
        "doorAngleDeg": (
            _rounded(np.rad2deg(door_angle), 4) if door_angle is not None else None
        ),
        "tcpHandleCm": _rounded(tcp_handle_cm, 4) if tcp_handle_cm is not None else None,
        "bounds": _rounded(np.stack((lower, upper)), 5),
        "tcpSource": tcp_name,
        "handleSource": handle_name,
    }


def _safe_json(value: Any) -> str:
    # Escaping angle brackets prevents labels from terminating the data script.
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _document(data: dict[str, Any], title: str) -> str:
    escaped_title = html.escape(title, quote=True)
    payload = _safe_json(data)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escaped_title}</title>
<style>
:root {{ color-scheme: light dark; --bg:#f7f7f9; --panel:#ffffff; --fg:#18181b; --muted:#62626b; --line:#d4d4d8; --grid:#e4e4e7; --robot:#2563eb; --object:#71717a; --tcp:#059669; --handle:#dc2626; --door:#7c3aed; --focus:#2563eb; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#111113; --panel:#1c1c1f; --fg:#f4f4f5; --muted:#a1a1aa; --line:#3f3f46; --grid:#2b2b30; --robot:#60a5fa; --object:#a1a1aa; --tcp:#34d399; --handle:#f87171; --door:#a78bfa; --focus:#60a5fa; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:20px; background:var(--bg); color:var(--fg); font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ width:100%; max-width:1100px; margin:0 auto; }}
h1 {{ margin:0 0 14px; font-size:clamp(1.2rem,3vw,1.65rem); font-weight:600; }}
.surface {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px; }}
.scene {{ width:100%; aspect-ratio:16/8; display:block; }}
.chart {{ width:100%; aspect-ratio:16/3.5; display:block; margin-top:10px; }}
.controls {{ display:grid; grid-template-columns:auto minmax(120px,1fr); gap:12px; align-items:center; margin-top:14px; }}
button {{ appearance:none; border:1px solid var(--line); border-radius:8px; background:var(--fg); color:var(--bg); padding:8px 16px; font:inherit; cursor:pointer; }}
button:focus-visible,input:focus-visible {{ outline:3px solid var(--focus); outline-offset:2px; }}
input[type="range"] {{ width:100%; accent-color:var(--robot); }}
.readout {{ display:flex; gap:12px; flex-wrap:wrap; margin-top:10px; color:var(--muted); font-variant-numeric:tabular-nums; }}
.readout strong {{ color:var(--fg); font-weight:600; }}
.legend {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:10px; color:var(--muted); font-size:.9rem; }}
.key::before {{ content:""; display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; background:var(--key); }}
@media (max-width:520px) {{ body {{ padding:10px; }} .surface {{ padding:10px; }} .controls {{ grid-template-columns:1fr; }} button {{ width:100%; }} .scene {{ aspect-ratio:4/3; }} .chart {{ aspect-ratio:3/1.4; }} }}
</style>
</head>
<body>
<main>
<h1>{escaped_title}</h1>
<section class="surface" aria-label="Trajectory player">
<canvas id="scene" class="scene" role="img" aria-label="Projected robot and microwave body motion with TCP and handle paths">Your browser does not support canvas.</canvas>
<canvas id="chart" class="chart" role="img" aria-label="Door angle and TCP-to-handle distance over time">Trajectory curve unavailable.</canvas>
<div class="legend" aria-hidden="true">
<span class="key" style="--key:var(--robot)">robot</span><span class="key" style="--key:var(--object)">object</span><span class="key" style="--key:var(--tcp)">TCP</span><span class="key" style="--key:var(--handle)">handle</span><span class="key" style="--key:var(--door)">door angle</span>
</div>
<div class="controls">
<button id="toggle" type="button" aria-label="Play trajectory">Play</button>
<label><span class="sr-only">Frame</span><input id="frame" type="range" min="0" value="0" step="1"></label>
</div>
<div class="readout" id="status" aria-live="polite">
<span>frame <strong id="frame-value">1</strong></span><span>time <strong id="time-value">0.000 s</strong></span><span>phase <strong id="phase-value">trajectory</strong></span><span id="door-wrap">door <strong id="door-value">—</strong></span><span id="gap-wrap">TCP gap <strong id="gap-value">—</strong></span>
</div>
</section>
</main>
<script id="agentpre-data" type="application/json">{payload}</script>
<script>
(() => {{
  "use strict";
  const data = JSON.parse(document.getElementById("agentpre-data").textContent);
  const scene = document.getElementById("scene");
  const chart = document.getElementById("chart");
  const slider = document.getElementById("frame");
  const toggle = document.getElementById("toggle");
  const frameValue = document.getElementById("frame-value");
  const timeValue = document.getElementById("time-value");
  const phaseValue = document.getElementById("phase-value");
  const doorValue = document.getElementById("door-value");
  const gapValue = document.getElementById("gap-value");
  slider.max = String(Math.max(0, data.frameCount - 1));
  let frame = 0;
  let playing = false;
  let previousTimestamp = 0;
  let playTime = 0;

  const color = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const resizeCanvas = canvas => {{
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
    const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
    if (canvas.width !== width || canvas.height !== height) {{ canvas.width = width; canvas.height = height; }}
    return {{ width, height, ratio }};
  }};
  const transformFor = (width, height, pad) => {{
    const lo = data.bounds[0], hi = data.bounds[1];
    const sx = (width - 2 * pad) / Math.max(hi[0] - lo[0], 1e-6);
    const sy = (height - 2 * pad) / Math.max(hi[1] - lo[1], 1e-6);
    const scale = Math.min(sx, sy);
    const usedW = (hi[0] - lo[0]) * scale, usedH = (hi[1] - lo[1]) * scale;
    return point => [pad + (width - 2 * pad - usedW) / 2 + (point[0] - lo[0]) * scale, height - pad - (height - 2 * pad - usedH) / 2 - (point[1] - lo[1]) * scale];
  }};
  const drawPath = (ctx, rows, upto, map, stroke, width) => {{
    if (!rows || rows.length === 0) return;
    ctx.beginPath();
    for (let i = 0; i <= upto; i += 1) {{ const p = map(rows[i]); if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); }}
    ctx.strokeStyle = stroke; ctx.lineWidth = width; ctx.stroke();
  }};
  const point = (ctx, p, fill, radius) => {{ ctx.beginPath(); ctx.arc(p[0], p[1], radius, 0, Math.PI * 2); ctx.fillStyle = fill; ctx.fill(); }};

  function drawScene() {{
    const size = resizeCanvas(scene), ctx = scene.getContext("2d");
    const pad = 24 * size.ratio, map = transformFor(size.width, size.height, pad);
    ctx.clearRect(0, 0, size.width, size.height);
    ctx.strokeStyle = color("--grid"); ctx.lineWidth = size.ratio;
    for (let i = 1; i < 5; i += 1) {{ const y = i * size.height / 5; ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(size.width - pad, y); ctx.stroke(); }}
    drawPath(ctx, data.handleXY, frame, map, color("--handle"), 1.5 * size.ratio);
    drawPath(ctx, data.tcpXY, frame, map, color("--tcp"), 1.5 * size.ratio);
    if (data.bodyXY) {{
      const row = data.bodyXY[frame];
      const robot = data.bodyGroups.robot;
      if (robot.length > 1) {{ ctx.beginPath(); robot.forEach((index, order) => {{ const p = map(row[index]); if (order === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); }}); ctx.strokeStyle = color("--robot"); ctx.lineWidth = 2 * size.ratio; ctx.stroke(); }}
      data.bodyGroups.object.forEach(index => point(ctx, map(row[index]), color("--object"), 3 * size.ratio));
      robot.forEach(index => point(ctx, map(row[index]), color("--robot"), 3.5 * size.ratio));
      data.bodyGroups.highlighted.forEach(index => {{
        const p = map(row[index]); point(ctx, p, color("--handle"), 4.5 * size.ratio);
        ctx.fillStyle = color("--fg"); ctx.font = `${{11 * size.ratio}}px system-ui`; ctx.fillText(data.bodyLabels[index].split("/").pop(), p[0] + 6 * size.ratio, p[1] - 6 * size.ratio);
      }});
    }}
    if (data.handleXY) point(ctx, map(data.handleXY[frame]), color("--handle"), 5 * size.ratio);
    if (data.tcpXY) point(ctx, map(data.tcpXY[frame]), color("--tcp"), 5 * size.ratio);
  }}

  const seriesRange = values => {{
    if (!values) return null;
    let lo = Math.min(...values), hi = Math.max(...values);
    if (Math.abs(hi - lo) < 1e-8) {{ lo -= 1; hi += 1; }}
    const padding = (hi - lo) * 0.08; return [lo - padding, hi + padding];
  }};
  function drawChart() {{
    const size = resizeCanvas(chart), ctx = chart.getContext("2d");
    const left = 42 * size.ratio, right = 42 * size.ratio, top = 14 * size.ratio, bottom = 25 * size.ratio;
    const w = size.width - left - right, h = size.height - top - bottom;
    ctx.clearRect(0, 0, size.width, size.height);
    ctx.strokeStyle = color("--grid"); ctx.lineWidth = size.ratio;
    for (let i = 0; i <= 4; i += 1) {{ const y = top + i * h / 4; ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + w, y); ctx.stroke(); }}
    const x = index => left + (data.frameCount <= 1 ? 0 : index * w / (data.frameCount - 1));
    const drawSeries = (values, range, stroke) => {{
      if (!values || !range) return;
      ctx.beginPath(); values.forEach((value, index) => {{ const y = top + (range[1] - value) * h / (range[1] - range[0]); if (index === 0) ctx.moveTo(x(index), y); else ctx.lineTo(x(index), y); }}); ctx.strokeStyle = stroke; ctx.lineWidth = 1.8 * size.ratio; ctx.stroke();
    }};
    const doorRange = seriesRange(data.doorAngleDeg), gapRange = seriesRange(data.tcpHandleCm);
    drawSeries(data.doorAngleDeg, doorRange, color("--door"));
    drawSeries(data.tcpHandleCm, gapRange, color("--tcp"));
    const cursorX = x(frame); ctx.beginPath(); ctx.moveTo(cursorX, top); ctx.lineTo(cursorX, top + h); ctx.strokeStyle = color("--fg"); ctx.lineWidth = size.ratio; ctx.stroke();
    ctx.fillStyle = color("--muted"); ctx.font = `${{11 * size.ratio}}px system-ui`;
    ctx.textAlign = "left"; ctx.fillText("0 s", left, size.height - 5 * size.ratio);
    ctx.textAlign = "right"; ctx.fillText(`${{data.durationS.toFixed(2)}} s`, left + w, size.height - 5 * size.ratio);
    if (doorRange) {{ ctx.textAlign = "left"; ctx.fillStyle = color("--door"); ctx.fillText("door °", left, top + 11 * size.ratio); }}
    if (gapRange) {{ ctx.textAlign = "right"; ctx.fillStyle = color("--tcp"); ctx.fillText("gap cm", left + w, top + 11 * size.ratio); }}
  }}

  function update() {{
    frame = Math.max(0, Math.min(data.frameCount - 1, Math.round(frame)));
    slider.value = String(frame); frameValue.textContent = `${{frame + 1}} / ${{data.frameCount}}`;
    timeValue.textContent = `${{data.timeS[frame].toFixed(3)}} s`; phaseValue.textContent = data.phases[frame];
    doorValue.textContent = data.doorAngleDeg ? `${{data.doorAngleDeg[frame].toFixed(2)}}°` : "—";
    gapValue.textContent = data.tcpHandleCm ? `${{data.tcpHandleCm[frame].toFixed(2)}} cm` : "—";
    document.getElementById("door-wrap").hidden = !data.doorAngleDeg;
    document.getElementById("gap-wrap").hidden = !data.tcpHandleCm;
    drawScene(); drawChart();
  }}
  function seekTime(seconds) {{
    let lo = 0, hi = data.frameCount - 1;
    while (lo < hi) {{ const mid = Math.floor((lo + hi + 1) / 2); if (data.timeS[mid] <= seconds) lo = mid; else hi = mid - 1; }}
    frame = lo;
  }}
  function animate(timestamp) {{
    if (!playing) return;
    if (!previousTimestamp) previousTimestamp = timestamp;
    playTime += Math.min((timestamp - previousTimestamp) / 1000, 0.1); previousTimestamp = timestamp;
    if (playTime > data.durationS) {{ playTime = 0; frame = 0; }} else seekTime(playTime);
    update(); requestAnimationFrame(animate);
  }}
  function stop() {{ playing = false; toggle.textContent = "Play"; toggle.setAttribute("aria-label", "Play trajectory"); previousTimestamp = 0; }}
  toggle.addEventListener("click", () => {{
    if (playing) {{ stop(); return; }}
    if (frame >= data.frameCount - 1) {{ frame = 0; playTime = 0; }} else playTime = data.timeS[frame];
    playing = true; toggle.textContent = "Pause"; toggle.setAttribute("aria-label", "Pause trajectory"); previousTimestamp = 0; requestAnimationFrame(animate);
  }});
  slider.addEventListener("input", () => {{ stop(); frame = Number(slider.value); playTime = data.timeS[frame]; update(); }});
  new ResizeObserver(update).observe(document.querySelector(".surface"));
  update();
}})();
</script>
</body>
</html>
"""


def write_animation_html(
    trajectory_path: str | Path,
    output_path: str | Path = "animation.html",
    *,
    title: str = "AgentPre trajectory",
    fallback_fps: float = 60.0,
) -> Path:
    """Write a standalone HTML trajectory player and return its path."""

    destination = Path(output_path)
    data = load_animation_data(trajectory_path, fallback_fps=fallback_fps)
    document = _document(data, title)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(document, encoding="utf-8")
        temporary.replace(destination)
    except OSError as exc:
        raise AnimationExportError(f"cannot write animation: {destination}") from exc
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export AgentPre trajectory.npz as a standalone HTML animation"
    )
    parser.add_argument("trajectory", type=Path, help="input trajectory.npz")
    parser.add_argument(
        "output", type=Path, nargs="?", default=Path("animation.html")
    )
    parser.add_argument("--title", default="AgentPre trajectory")
    parser.add_argument("--fallback-fps", type=float, default=60.0)
    arguments = parser.parse_args(argv)
    written = write_animation_html(
        arguments.trajectory,
        arguments.output,
        title=arguments.title,
        fallback_fps=arguments.fallback_fps,
    )
    print(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
