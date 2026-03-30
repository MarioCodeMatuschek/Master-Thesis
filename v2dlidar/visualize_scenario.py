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

def _round_xy_key(x: float, y: float, ndigits: int = 3) -> Tuple[float, float]:
    return (round(float(x), ndigits), round(float(y), ndigits))


def load_all_scan_paths(scenario_dir: str) -> List[Dict[str, object]]:
    """Load all scan_paths/scan_XXXX.json files in a scenario folder."""
    path_dir = os.path.join(scenario_dir, "scan_paths")
    if not os.path.isdir(path_dir):
        return []
    out: List[Dict[str, object]] = []
    for fn in sorted(os.listdir(path_dir)):
        if not (fn.startswith("scan_") and fn.endswith(".json")):
            continue
        fp = os.path.join(path_dir, fn)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                out.append(json.load(f))
        except Exception:
            continue
    return out


def build_full_route_from_scan(
    scenario_dir: str,
    scan_id: int,
    max_legs: int = 10,
) -> Optional[Dict[str, object]]:
    """
    Reconstruct a multi-leg route by chaining scan_paths entries:
      leg i goal == leg i+1 start (x,y)

    This is intended for datasets where each leg is stored as a separate scan_path file
    (e.g., start->wp1, wp1->wp2, wp2->final).
    """
    first = load_scan_path(scenario_dir, scan_id)
    if first is None:
        return None

    all_paths = load_all_scan_paths(scenario_dir)
    by_start: Dict[Tuple[float, float], List[Dict[str, object]]] = {}
    for sp in all_paths:
        st = sp.get("start") or {}
        try:
            k = _round_xy_key(st["x"], st["y"])
        except Exception:
            continue
        by_start.setdefault(k, []).append(sp)
    for k in list(by_start.keys()):
        # Prefer deterministic chaining: smallest scan_id first.
        by_start[k].sort(key=lambda d: int(d.get("scan_id", 0)))

    # Walk the chain.
    chain: List[Dict[str, object]] = [first]
    visited = set()
    cur = first
    for _ in range(max_legs - 1):
        goal = cur.get("goal") or {}
        try:
            gk = _round_xy_key(goal["x"], goal["y"])
        except Exception:
            break
        if gk in visited:
            break
        visited.add(gk)

        nxt_candidates = by_start.get(gk, [])
        if not nxt_candidates:
            break
        # Pick the first candidate whose goal differs (avoid trivial self-loops).
        nxt = None
        for cand in nxt_candidates:
            cgoal = cand.get("goal") or {}
            try:
                ck = _round_xy_key(cgoal["x"], cgoal["y"])
            except Exception:
                continue
            if ck != gk:
                nxt = cand
                break
        if nxt is None:
            break
        chain.append(nxt)
        cur = nxt

    # Combine paths and collect waypoints (intermediate goals).
    combined_pts: List[Dict[str, float]] = []
    waypoints: List[Dict[str, float]] = []
    for i, leg in enumerate(chain):
        pts = leg.get("path") or []
        if not isinstance(pts, list) or not pts:
            continue
        if i == 0:
            combined_pts.extend(pts)
        else:
            combined_pts.extend(pts[1:])  # avoid duplicate join point
        if i < len(chain) - 1:
            g = leg.get("goal") or {}
            try:
                waypoints.append({"x": float(g["x"]), "y": float(g["y"])})
            except Exception:
                pass

    start = (first.get("start") or {}).copy()
    final_goal = (chain[-1].get("goal") or {}).copy()
    return {
        "scenario_id": first.get("scenario_id"),
        "scan_id": first.get("scan_id"),
        "start": start,
        "goal": final_goal,
        "waypoints": waypoints,
        "path": combined_pts,
    }


def draw_scenario_on_ax(
    ax,
    scen_spec: ScenarioSpec,
    segments,
    show_pose: Optional[Tuple[float, float, float]] = None,
    scan_path: Optional[Dict[str, object]] = None,
    title_extra: Optional[str] = None,
) -> None:
    """Draw scenario walls (and optionally pose + ideal path) onto an existing Axes."""
    layout = getattr(scen_spec, "layout", "union")
    if layout == "apartment":
        layout_data = get_apartment_layout_data(scen_spec)
        if layout_data is not None:
            rooms, doors = layout_data
            for r in rooms:
                rect = mpatches.Rectangle(
                    (r.x, r.y),
                    r.width,
                    r.height,
                    linewidth=3,
                    edgecolor="#333333",
                    facecolor="#f9f9f9",
                )
                ax.add_patch(rect)
            # Doors are (dx, dy, orientation, width_used); draw gaps using
            # width_used so visual door openings match the planner geometry.
            for dx, dy, orientation, width_used in doors:
                half_door = width_used / 2.0
                if orientation == "vertical":
                    ax.plot(
                        [dx, dx],
                        [dy - half_door, dy + half_door],
                        color="white",
                        linewidth=4,
                        zorder=3,
                    )
                else:
                    ax.plot(
                        [dx - half_door, dx + half_door],
                        [dy, dy],
                        color="white",
                        linewidth=4,
                        zorder=3,
                    )
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
        # Optional: show intermediate waypoints (multi-leg route)
        wps = scan_path.get("waypoints") or []
        if wps:
            try:
                wx = [float(p["x"]) for p in wps]
                wy = [float(p["y"]) for p in wps]
                ax.scatter(
                    wx,
                    wy,
                    marker="s",
                    s=40,
                    c="orange",
                    edgecolors="black",
                    linewidths=0.5,
                    label="waypoints",
                    zorder=4,
                )
            except Exception:
                pass
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
        arrow_len = min(scen_spec.width, scen_spec.height) * 0.05
        ax.arrow(
            x,
            y,
            arrow_len * math.cos(yaw),
            arrow_len * math.sin(yaw),
            head_width=arrow_len * 0.3,
            head_length=arrow_len * 0.4,
            color="red",
        )

    ax.set_xlim(0, scen_spec.width)
    ax.set_ylim(0, scen_spec.height)
    ax.set_aspect("equal", adjustable="box")
    title = "Scenario visualization"
    if title_extra:
        title = f"{title} — {title_extra}"
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.invert_yaxis()  # match the GUI convention
    if scan_path is not None:
        ax.legend(loc="best")


def plot_scenario(
    scen_spec: ScenarioSpec,
    show_pose: Optional[Tuple[float, float, float]] = None,
    scan_path: Optional[Dict[str, object]] = None,
    save_path: Optional[str] = None,
    title_extra: Optional[str] = None,
) -> None:
    """Plot the scenario walls and optionally a LiDAR pose and ideal path.
    If save_path is set, save the figure to that path. If title_extra is set, append it to the title."""
    segments, _, _, _, _ = generate_scenario(scen_spec)

    fig, ax = plt.subplots(figsize=(6, 6))
    draw_scenario_on_ax(
        ax,
        scen_spec,
        segments,
        show_pose=show_pose,
        scan_path=scan_path,
        title_extra=title_extra,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if save_path is None:
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
    ap.add_argument(
        "--full_route",
        action="store_true",
        help=(
            "If set (and --scan_id is provided), attempt to reconstruct and plot the full multi-leg "
            "route (start + waypoints + final goal) by chaining scan_paths files."
        ),
    )
    args = ap.parse_args()

    scenario_dir = os.path.join(args.root, f"scenario_{args.scenario_id:04d}")
    if not os.path.isdir(scenario_dir):
        raise SystemExit(f"Scenario directory not found: {scenario_dir}")

    scen_spec = load_scenario_spec(scenario_dir)
    scan_path = None
    if args.scan_id is not None:
        scan_path = (
            build_full_route_from_scan(scenario_dir, args.scan_id)
            if args.full_route
            else load_scan_path(scenario_dir, args.scan_id)
        )
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
