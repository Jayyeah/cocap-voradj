#!/usr/bin/env python3
"""Render CoCap coverage rollout GIFs with clipped Voronoi cells."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
import numpy as np
from PIL import Image


def xy_array(trajectory: list) -> np.ndarray:
    array = np.asarray(trajectory, dtype=float)
    if array.ndim != 2 or len(array) == 0:
        return np.zeros((0, 2), dtype=float)
    return array[:, :2]


def clip_half_plane(polygon: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    if len(polygon) == 0:
        return polygon
    output: List[np.ndarray] = []
    eps = 1e-9
    for index, current in enumerate(polygon):
        previous = polygon[index - 1]
        current_value = float(np.dot(normal, current) - offset)
        previous_value = float(np.dot(normal, previous) - offset)
        current_inside = current_value <= eps
        previous_inside = previous_value <= eps
        if current_inside != previous_inside:
            direction = current - previous
            denominator = float(np.dot(normal, direction))
            if abs(denominator) > eps:
                t = float((offset - np.dot(normal, previous)) / denominator)
                output.append(previous + np.clip(t, 0.0, 1.0) * direction)
        if current_inside:
            output.append(current)
    return np.asarray(output, dtype=float)


def clipped_voronoi_cells(positions: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float) -> List[np.ndarray]:
    rectangle = np.asarray([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=float)
    cells: List[np.ndarray] = []
    for index, point in enumerate(positions):
        cell = rectangle.copy()
        for other_index, other in enumerate(positions):
            if other_index == index:
                continue
            normal = 2.0 * (other - point)
            offset = float(np.dot(other, other) - np.dot(point, point))
            cell = clip_half_plane(cell, normal, offset)
            if len(cell) == 0:
                break
        cells.append(cell)
    return cells


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def area_cv(cells: List[np.ndarray]) -> float:
    areas = np.asarray([polygon_area(cell) for cell in cells if len(cell) >= 3], dtype=float)
    if len(areas) == 0:
        return float("nan")
    mean_area = float(np.mean(areas))
    if mean_area <= 1e-9:
        return float("nan")
    return float(np.std(areas) / mean_area)


def plot_faded_trail(axis, trajectory: np.ndarray, upto: int, color: str, trail_window: int) -> None:
    start = max(0, upto - trail_window)
    recent = trajectory[start:upto]
    if len(recent) < 2:
        return
    denominator = max(1, len(recent) - 2)
    for index in range(len(recent) - 1):
        alpha = 0.10 + 0.58 * (index / denominator)
        axis.plot(recent[index:index + 2, 0], recent[index:index + 2, 1], color=color, linewidth=1.8, alpha=alpha, solid_capstyle="round", zorder=3)


def frame_ids(max_len: int, max_frames: int) -> List[int]:
    if max_len <= 1:
        return [0]
    if max_len <= max_frames:
        return list(range(max_len))
    return sorted(set(np.linspace(0, max_len - 1, max_frames, dtype=int).tolist()))


def render_episode(episode: dict, output_path: Path, trail_window: int, frame_duration_ms: int, max_frames: int) -> dict:
    env = episode["env"]
    rollout = episode["rollout"]
    trajectories = [xy_array(item) for item in episode.get("pursuers", {}).get("trajectory", [])]
    max_len = max([len(item) for item in trajectories] or [1])
    width = float(env.get("width", 55.0))
    height = float(env.get("height", width))
    x_min = float(env.get("inner_x_min", 0.0))
    x_max = float(env.get("inner_x_max", width))
    y_min = float(env.get("inner_y_min", 0.0))
    y_max = float(env.get("inner_y_max", height))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf", "#bcbd22"]
    images = []
    for frame_index in frame_ids(max_len, int(max_frames)):
        figure, axis = plt.subplots(figsize=(6.2, 6.2), dpi=110)
        axis.set_xlim(0.0, width)
        axis.set_ylim(0.0, height)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, linewidth=0.25, color="#d0d0d0", zorder=0)
        axis.add_patch(Rectangle((0.0, 0.0), width, height, fill=False, edgecolor="#222222", linewidth=1.4, zorder=5))
        axis.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor="#555555", linestyle="--", linewidth=1.0, zorder=5))
        for position, radius in zip(env.get("obstacles", {}).get("positions", []), env.get("obstacles", {}).get("r", [])):
            axis.add_patch(Circle((position[0], position[1]), radius, color="#555555", alpha=0.4, zorder=4))
        alive_positions = []
        for index, trajectory in enumerate(trajectories):
            if len(trajectory) == 0:
                continue
            dead_at_frame = frame_index >= len(trajectory)
            if not dead_at_frame:
                alive_positions.append(trajectory[min(frame_index, len(trajectory) - 1)])
        alive_np = np.asarray(alive_positions, dtype=float) if alive_positions else np.zeros((0, 2), dtype=float)
        if len(alive_np) >= 2:
            cells = clipped_voronoi_cells(alive_np, x_min, x_max, y_min, y_max)
            live_cv = area_cv(cells)
            cell_idx = 0
            for index, trajectory in enumerate(trajectories):
                if len(trajectory) == 0:
                    continue
                dead_at_frame = frame_index >= len(trajectory)
                if dead_at_frame:
                    continue
                if cell_idx < len(cells):
                    cell = cells[cell_idx]
                    if len(cell) >= 3:
                        color = colors[index % len(colors)]
                        axis.add_patch(Polygon(cell, closed=True, facecolor=color, edgecolor=color, alpha=0.09, linewidth=1.5, zorder=1))
                        closed = np.vstack([cell, cell[0]])
                        axis.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.2, alpha=0.85, zorder=2)
                cell_idx += 1
        else:
            live_cv = float("nan")
        for index, trajectory in enumerate(trajectories):
            if len(trajectory) == 0:
                continue
            upto = min(frame_index + 1, len(trajectory))
            color = colors[index % len(colors)]
            dead_at_frame = frame_index >= len(trajectory)
            plot_faded_trail(axis, trajectory, upto, color, trail_window)
            point = trajectory[-1] if dead_at_frame else trajectory[upto - 1]
            if dead_at_frame:
                axis.scatter(point[0], point[1], s=64, color="red", marker="x", linewidth=2.0, zorder=8)
            else:
                axis.scatter(point[0], point[1], s=48, color=color, edgecolor="white", linewidth=0.9, zorder=6)
                axis.text(point[0] + 1.0, point[1] + 1.0, f"P{index}", fontsize=8, color=color, weight="bold", zorder=7)
        status = "SUCCESS" if rollout.get("coverage_training_success") else "FAIL"
        axis.set_title(f"task coverage | {status} | collision {bool(rollout.get('collision_event'))} | step {frame_index}/{max_len - 1} | cv {live_cv:.3f}", fontsize=8.5)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        figure.tight_layout(pad=0.5)
        figure.canvas.draw()
        images.append(np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(figure)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save(output_path, save_all=True, append_images=pil_images[1:], duration=max(10, int(frame_duration_ms)), loop=0, optimize=False, disposal=2)
    return {"seed": int(rollout.get("seed", -1)), "output": str(output_path), "frames": len(images), "success": bool(rollout.get("coverage_training_success")), "loose": bool(rollout.get("coverage_loose_success")), "strict": bool(rollout.get("coverage_strict_success")), "collision": bool(rollout.get("collision_event"))}


def episode_paths(input_root: Path, seeds: Iterable[int]) -> List[Path]:
    return [input_root / f"episode_seed_{int(seed)}.json" for seed in seeds]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--trail-window", type=int, default=60)
    parser.add_argument("--frame-duration-ms", type=int, default=100)
    parser.add_argument("--max-gif-frames", type=int, default=300)
    args = parser.parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    records = []
    for episode_path in episode_paths(input_root, args.seeds):
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        seed = int(episode["rollout"]["seed"])
        records.append(render_episode(episode, output_root / f"rollout_seed_{seed}.gif", int(args.trail_window), int(args.frame_duration_ms), int(args.max_gif_frames)))
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"input_root": str(input_root), "output_root": str(output_root), "records": records}
    (output_root / "render_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"rendered": len(records), "output_root": str(output_root)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
