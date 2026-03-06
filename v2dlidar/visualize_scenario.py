"""
Simple CLI tool to visualize generated scenarios.

This module lets you inspect a scenario after it has been generated
by the dataset CLI, using only the contents of a scenario directory:

- It reads `scenario_meta.json` to recover the ScenarioSpec (including the RNG seed).
- It regenerates the wall segments via `generate_scenario`.
- It optionally overlays the first LiDAR pose from `scans_long.csv`.
- It can also overlay the ideal path for a specific scan from `scan_paths/scan_XXXX.json`.

Example (from the folder that contains the `v2dlidar` package):

    python -m v2dlidar.visualize_scenario --root ./dataset_10x1 --scenario_id 0
    python -m v2dlidar.visualize_scenario --root ./dataset_10x1 --scenario_id 0 --scan_id 0
"""

import argparse
import csv
import json
import math
import os
from typing import Optional, Tuple, List, Dict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .mapgen import ScenarioSpec, generate_scenario, get_apartment_layout_data


def load_scenario_spec(scenario_dir: str) -> ScenarioSpec:
    """Load ScenarioSpec from `scenario_meta.json` in a scenario folder."""
    meta_path = os.path.join(scenario_dir, "scenario_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    spec_dict = meta.get("spec", {})
    return ScenarioSpec(**spec_dict)


def load_first_pose(scenario_dir: str) -> Optional[Tuple[float, float, float]]:
    """Load the first LiDAR pose (x, y, yaw) from `scans_long.csv` if available."""
    csv_path = os.path.join(scenario_dir, "scans_long.csv")
    if not os.path.exists(csv_path):
        return None

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return None

        # Expect columns: scenario_id,scan_id,t,pose_x,pose_y,pose_yaw,...
        try:
            ix_x = header.index("pose_x")
            ix_y = header.index("pose_y")
            ix_yaw = header.index("pose_yaw")
        except ValueError:
            return None

        for row in reader:
            if not row or len(row) <= max(ix_x, ix_y, ix_yaw):
                continue
            try:
                x = float(row[ix_x])
                y = float(row[ix_y])
                yaw = float(row[ix_yaw])
                return (x, y, yaw)
            except ValueError:
                continue

    return None


def load_scan_path(
    scenario_dir: str, scan_id: int
) -> Optional[Dict[str, object]]:
    """Load the ideal path for a given scan_id from scan_paths/scan_XXXX.json if present."""
    path_dir = os.path.join(scenario_dir, "scan_paths")
    path_path = os.path.join(path_dir, f"scan_{scan_id:04d}.json")
    if not os.path.exists(path_path):
        return None
    with open(path_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_scenario(
    scen_spec: ScenarioSpec,
    show_pose: Optional[Tuple[float, float, float]] = None,
    scan_path: Optional[Dict[str, object]] = None,
) -> None:
    """Plot the scenario walls and optionally a LiDAR pose and ideal path."""
    segments, spec, _, _ = generate_scenario(scen_spec)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot walls: apartment layout uses rooms + doors; union uses segments
    layout = getattr(scen_spec, "layout", "union")
    if layout == "apartment":
        layout_data = get_apartment_layout_data(scen_spec)
        if layout_data is not None:
            rooms, doors = layout_data
            door_width = 0.8
            half_door = door_width / 2.0
            for i, r in enumerate(rooms):
                rect = mpatches.Rectangle(
                    (r.x, r.y), r.width, r.height,
                    linewidth=3, edgecolor="#333333", facecolor="#f9f9f9"
                )
                ax.add_patch(rect)
                ax.text(
                    r.x + r.width / 2, r.y + r.height / 2, f"Room {i + 1}",
                    ha="center", va="center", fontweight="bold", color="#555555"
                )
            for dx, dy, orientation in doors:
                if orientation == "vertical":
                    ax.plot(
                        [dx, dx], [dy - half_door, dy + half_door],
                        color="white", linewidth=4, zorder=3
                    )
                    arc = mpatches.Arc(
                        (dx, dy - half_door), door_width * 2, door_width * 2,
                        theta1=0, theta2=90, color="blue", linewidth=1.5, zorder=4
                    )
                    ax.add_patch(arc)
                else:
                    ax.plot(
                        [dx - half_door, dx + half_door], [dy, dy],
                        color="white", linewidth=4, zorder=3
                    )
                    arc = mpatches.Arc(
                        (dx - half_door, dy), door_width * 2, door_width * 2,
                        theta1=270, theta2=360, color="blue", linewidth=1.5, zorder=4
                    )
                    ax.add_patch(arc)
        else:
            for s in segments:
                ax.plot([s.x1, s.x2], [s.y1, s.y2], color="black", linewidth=1)
    else:
        for s in segments:
            ax.plot([s.x1, s.x2], [s.y1, s.y2], color="black", linewidth=1)

    # Plot ideal path for a specific scan, if provided
    if scan_path is not None:
        pts = scan_path.get("path") or []
        if pts:
            xs = [float(p["x"]) for p in pts]
            ys = [float(p["y"]) for p in pts]
            ax.plot(xs, ys, color="blue", linewidth=2, label="ideal path")
        # Overlay start/goal markers from the path info
        start = scan_path.get("start")
        goal = scan_path.get("goal")
        if start is not None:
            sx = float(start["x"])
            sy = float(start["y"])
            ax.plot(sx, sy, marker="o", color="green", markersize=6, label="start")
        if goal is not None:
            gx = float(goal["x"])
            gy = float(goal["y"])
            ax.plot(gx, gy, marker="x", color="magenta", markersize=7, label="goal")

    # Plot pose if provided
    if show_pose is not None:
        x, y, yaw = show_pose
        ax.plot(x, y, marker="o", color="red", markersize=6)
        # Draw a short heading arrow
        arrow_len = min(spec.width, spec.height) * 0.05
        ax.arrow(
            x,
            y,
            arrow_len * math.cos(yaw),
            arrow_len * math.sin(yaw),
            head_width=arrow_len * 0.3,
            head_length=arrow_len * 0.4,
            color="red",
        )

    ax.set_xlim(0, spec.width)
    ax.set_ylim(0, spec.height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Scenario visualization")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.invert_yaxis()  # match the GUI convention
    if scan_path is not None:
        ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Entry point for `python -m v2dlidar.visualize_scenario`."""
    ap = argparse.ArgumentParser(
        description="Visualize a generated 2D LiDAR scenario."
    )
    ap.add_argument(
        "--root",
        type=str,
        default="./dataset",
        help=(
            "Root dataset directory containing scenario_XXXX folders "
            "(default: ./dataset)."
        ),
    )
    ap.add_argument(
        "--scenario_id",
        type=int,
        required=True,
        help="Numeric scenario ID to visualize (e.g. 0 for scenario_0000).",
    )
    ap.add_argument(
        "--no_pose",
        action="store_true",
        help="Disable overlay of the first LiDAR pose from scans_long.csv.",
    )
    ap.add_argument(
        "--scan_id",
        type=int,
        default=None,
        help=(
            "Optional scan_id whose ideal path should be overlaid "
            "from scan_paths/scan_XXXX.json."
        ),
    )
    args = ap.parse_args()

    scenario_dir = os.path.join(args.root, f"scenario_{args.scenario_id:04d}")
    if not os.path.isdir(scenario_dir):
        raise SystemExit(f"Scenario directory not found: {scenario_dir}")

    scen_spec = load_scenario_spec(scenario_dir)
    scan_path = None
    if args.scan_id is not None:
        scan_path = load_scan_path(scenario_dir, args.scan_id)
    pose = None if args.no_pose else load_first_pose(scenario_dir)
    # If a scan_id is provided, prefer its stored start pose for the overlay
    if scan_path is not None and not args.no_pose:
        start = scan_path.get("start")
        if start is not None:
            pose = (
                float(start["x"]),
                float(start["y"]),
                float(start.get("yaw", 0.0)),
            )
    plot_scenario(scen_spec, show_pose=pose, scan_path=scan_path)


if __name__ == "__main__":
    main()
