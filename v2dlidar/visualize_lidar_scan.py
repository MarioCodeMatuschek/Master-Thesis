"""
CLI tool to visualize a single LiDAR scan (with noise and dropout)
from an existing dataset scenario.

This module builds on `visualize_scenario`:

- It reads `scenario_meta.json` to recover both ScenarioSpec and LidarSpec.
- It reconstructs the scenario geometry via `generate_scenario`.
- It reads one scan from `scans_long.csv` (all rays for a given scan_id).
- It optionally overlays the ideal path from `scan_paths/scan_XXXX.json`.
- It shows two linked views:
  - World/layout view with walls, pose, path, and rays (valid; dropout as red lines to noise-free range; no wall as orange dashed to r_m).
  - Polar LiDAR view (angle vs. range) with noise-free baseline and dropout markers.
"""

import argparse
import csv
import json
import math
import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from .mapgen import ScenarioSpec, generate_scenario
from .lidar import LidarSpec
from .visualize_scenario import (
    load_scenario_spec,
    load_scan_path,
    build_full_route_from_scan,
    draw_scenario_on_ax,
)


def load_lidar_spec(scenario_dir: str) -> LidarSpec:
    """Load LidarSpec from the `lidar` block in scenario_meta.json."""
    meta_path = os.path.join(scenario_dir, "scenario_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    lidar_dict = meta.get("lidar", {})
    return LidarSpec(**lidar_dict)


def load_scan_from_csv(
    scenario_dir: str,
    scan_id: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Load one scan (all rays) from scans_long.csv.

    If scan_id is None, a random available scan_id is chosen.
    Returns a dict with:
      - scan_id: int
      - pose: (x, y, yaw)
      - goal: (x, y)
      - theta_deg, r_m, noise_free_r_m, hit_x, hit_y, valid: np.ndarray
    """
    csv_path = os.path.join(scenario_dir, "scans_long.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"scans_long.csv not found in {scenario_dir}")

    # First pass: collect per-row data and available scan_ids
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError("Empty scans_long.csv")

        idx = {name: header.index(name) for name in header}

        rows = []
        scan_ids = set()
        for row in reader:
            if not row:
                continue
            sid = int(row[idx["scan_id"]])
            scan_ids.add(sid)
            rows.append(row)

    if not scan_ids:
        raise ValueError("No scans found in scans_long.csv")

    if rng is None:
        rng = np.random.default_rng()

    if scan_id is None:
        scan_id = int(rng.choice(sorted(list(scan_ids))))

    # Filter rows for the chosen scan_id
    scan_rows = [r for r in rows if int(r[idx["scan_id"]]) == scan_id]
    if not scan_rows:
        raise ValueError(f"No rows found for scan_id {scan_id}")

    # Pose / goal (constant for all rays of this scan)
    first = scan_rows[0]
    pose_x = float(first[idx["pose_x"]])
    pose_y = float(first[idx["pose_y"]])
    pose_yaw = float(first[idx["pose_yaw"]])
    goal_x = float(first[idx["goal_x"]])
    goal_y = float(first[idx["goal_y"]])

    theta_deg = []
    r_m = []
    noise_free_r_m = []
    hit_x = []
    hit_y = []
    valid = []

    for r in scan_rows:
        theta_deg.append(float(r[idx["theta_deg"]]))
        r_val = float(r[idx["r_m"]])
        r_m.append(r_val)
        nf = float(r[idx["noise_free_r_m"]])
        noise_free_r_m.append(nf)
        # hit_x/hit_y may be empty strings for invalid rays
        hx_str = r[idx["hit_x"]]
        hy_str = r[idx["hit_y"]]
        if hx_str == "" or hy_str == "":
            hit_x.append(np.nan)
            hit_y.append(np.nan)
        else:
            hit_x.append(float(hx_str))
            hit_y.append(float(hy_str))
        valid.append(int(r[idx["valid"]]))

    return {
        "scan_id": int(scan_id),
        "pose": (pose_x, pose_y, pose_yaw),
        "goal": (goal_x, goal_y),
        "theta_deg": np.asarray(theta_deg, dtype=np.float32),
        "r_m": np.asarray(r_m, dtype=np.float32),
        "noise_free_r_m": np.asarray(noise_free_r_m, dtype=np.float32),
        "hit_x": np.asarray(hit_x, dtype=np.float32),
        "hit_y": np.asarray(hit_y, dtype=np.float32),
        "valid": np.asarray(valid, dtype=np.int32),
    }


def draw_lidar_rays_on_ax(
    ax,
    pose: Tuple[float, float, float],
    scan_data: Dict[str, np.ndarray],
    lidar_spec: LidarSpec,
    show_ideal_hits: bool = True,
) -> None:
    """Overlay LiDAR rays for a single scan on the given Axes.

    Ray directions (valid, dropout, no-wall) and ideal hit dots are drawn flipped
    180° for display on the layout axes only; stored angles and the polar subplot
    are unchanged.

    Invalid rays are split for the map view: dropout (measured r≈0) as solid red
    lines to the noise-free range along each displayed beam, vs no wall hit as
    orange dashed to r_m. The polar subplot is unchanged by this helper.
    """
    x0, y0, _yaw = pose
    theta_deg = scan_data["theta_deg"]
    r_m = scan_data["r_m"]
    noise_free_r_m = scan_data["noise_free_r_m"]
    hit_x = scan_data["hit_x"]
    hit_y = scan_data["hit_y"]
    valid = scan_data["valid"].astype(bool)

    # Convert angles to radians in world coordinates: theta_deg is already absolute.
    theta_rad = np.radians(theta_deg)
    # Visualization-only: flip drawn ray directions by 180° on the layout axes (polar unchanged).
    cth = -np.cos(theta_rad)
    sth = -np.sin(theta_rad)

    dropout_eps = 1e-6
    r_max = float(lidar_spec.max_range)

    # Valid returns: line segments to hit points
    if np.any(valid):
        vx = hit_x[valid].astype(float)
        vy = hit_y[valid].astype(float)
        n = vx.size
        segs_valid = np.empty((n, 2, 2), dtype=float)
        segs_valid[:, 0, 0] = x0
        segs_valid[:, 0, 1] = y0
        segs_valid[:, 1, 0] = 2.0 * x0 - vx
        segs_valid[:, 1, 1] = 2.0 * y0 - vy
        lc_valid = LineCollection(
            segs_valid,
            colors="#1f77b4",
            linewidths=0.5,
            alpha=0.5,
            label="Valid return (noisy)",
        )
        ax.add_collection(lc_valid)

    invalid = ~valid
    dropout = invalid & (np.abs(r_m) < dropout_eps)
    no_wall = invalid & (np.abs(r_m) >= dropout_eps)

    if np.any(dropout):
        r_nf_d = np.clip(noise_free_r_m[dropout].astype(float), 0.0, r_max)
        # Avoid degenerate segments if noise-free is exactly zero (rare).
        r_nf_d = np.maximum(r_nf_d, 1e-9)
        n_d = r_nf_d.size
        x1 = x0 + r_nf_d * cth[dropout]
        y1 = y0 + r_nf_d * sth[dropout]
        segs_d = np.empty((n_d, 2, 2), dtype=float)
        segs_d[:, 0, 0] = x0
        segs_d[:, 0, 1] = y0
        segs_d[:, 1, 0] = x1
        segs_d[:, 1, 1] = y1
        lc_drop = LineCollection(
            segs_d,
            colors="red",
            linestyles="solid",
            linewidths=0.7,
            alpha=0.85,
            label="Dropout / no return",
            zorder=5,
        )
        ax.add_collection(lc_drop)

    if np.any(no_wall):
        r_nw = r_m[no_wall].astype(float)
        x_end = x0 + r_nw * cth[no_wall]
        y_end = y0 + r_nw * sth[no_wall]
        n_nw = int(r_nw.size)
        segs_nw = np.empty((n_nw, 2, 2), dtype=float)
        segs_nw[:, 0, 0] = x0
        segs_nw[:, 0, 1] = y0
        segs_nw[:, 1, 0] = x_end
        segs_nw[:, 1, 1] = y_end
        lc_nw = LineCollection(
            segs_nw,
            colors="#ff7f0e",
            linestyles="dashed",
            linewidths=0.65,
            alpha=0.55,
            label="No wall hit within max range (r≈r_max)",
            zorder=3,
        )
        ax.add_collection(lc_nw)

    # Optionally show ideal (noise-free) hit locations as small gray dots
    if show_ideal_hits:
        ideal_r = np.clip(noise_free_r_m, 0.0, lidar_spec.max_range)
        ix = x0 + ideal_r * cth
        iy = y0 + ideal_r * sth
        ax.scatter(
            ix,
            iy,
            s=4,
            c="gray",
            alpha=0.6,
            label="ideal (noise-free) hits",
        )

    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize="small")


def draw_polar_lidar(
    ax,
    scan_data: Dict[str, np.ndarray],
    lidar_spec: LidarSpec,
    show_target_hits: bool = False,
    target_hit_tol_m: float = 0.15,
) -> None:
    """Draw a robot-centric polar view of one LiDAR scan (angle vs. range).

    0° corresponds to the robot's forward direction (pose_yaw) for this scan.
    """
    theta_deg = scan_data["theta_deg"]
    r_m = scan_data["r_m"]
    noise_free_r_m = scan_data["noise_free_r_m"]
    valid = scan_data["valid"].astype(bool)
    hit_x = scan_data.get("hit_x", None)
    hit_y = scan_data.get("hit_y", None)
    goal = scan_data.get("goal", None)

    # Convert stored world-frame angles to robot-centric angles by subtracting pose_yaw.
    pose = scan_data.get("pose")
    if pose is not None and len(pose) == 3:
        _, _, pose_yaw = pose
    else:
        pose_yaw = 0.0

    theta_world = np.radians(theta_deg)
    theta_rel = np.mod(theta_world - pose_yaw, 2.0 * math.pi)

    # Noise-free baseline
    ax.scatter(
        theta_rel,
        noise_free_r_m,
        s=3,
        c="gray",
        alpha=0.7,
        label="noise-free range",
    )

    # Noisy valid returns
    if np.any(valid):
        ax.scatter(
            theta_rel[valid],
            r_m[valid],
            s=3,
            c="#1f77b4",
            alpha=0.8,
            label="noisy range (valid)",
        )

    # Target-hit overlay (opt-in so existing behavior is unchanged).
    if (
        show_target_hits
        and goal is not None
        and hit_x is not None
        and hit_y is not None
        and len(goal) == 2
    ):
        gx, gy = float(goal[0]), float(goal[1])
        hx = np.asarray(hit_x, dtype=np.float32)
        hy = np.asarray(hit_y, dtype=np.float32)
        dist = np.hypot(hx - gx, hy - gy)
        is_target_hit = valid & np.isfinite(dist) & (dist <= float(target_hit_tol_m))
        if np.any(is_target_hit):
            ax.scatter(
                theta_rel[is_target_hit],
                r_m[is_target_hit],
                s=30,
                c="limegreen",
                marker="*",
                edgecolors="black",
                linewidths=0.5,
                alpha=0.95,
                label="target hit",
                zorder=10,
            )

    # Dropouts / invalid
    invalid = ~valid
    if np.any(invalid):
        ax.scatter(
            theta_rel[invalid],
            r_m[invalid],
            s=8,
            c="red",
            marker="x",
            alpha=0.8,
            label="dropout / no return",
        )

    ax.set_rmax(lidar_spec.max_range)
    ax.set_title(
        "LiDAR polar view (robot-centric, 0° = forward)\n"
        f"fov={lidar_spec.fov_deg:.0f}°, rays={lidar_spec.num_rays}, "
        f"noise_std={lidar_spec.range_noise_std:.3f}, "
        f"dropout={lidar_spec.dropout_prob:.2f}"
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(135)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize="small")


def main() -> None:
    """Entry point for `python -m v2dlidar.visualize_lidar_scan`."""
    ap = argparse.ArgumentParser(
        description=(
            "Visualize a single LiDAR scan (with noise/dropout) "
            "for a generated scenario."
        )
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
        help="Numeric scenario ID (e.g. 0 for scenario_0000).",
    )
    ap.add_argument(
        "--scan_id",
        type=int,
        default=None,
        help=(
            "Scan ID to visualize. If omitted, a random scan from "
            "scans_long.csv is chosen."
        ),
    )
    ap.add_argument(
        "--no_path",
        action="store_true",
        help="Disable overlay of the ideal path from scan_paths/scan_XXXX.json.",
    )
    ap.add_argument(
        "--full_route",
        action="store_true",
        help=(
            "If set, attempt to reconstruct and overlay the full multi-leg route (start + waypoints + final goal) "
            "by chaining scan_paths files. Requires scan_paths to be present."
        ),
    )
    ap.add_argument(
        "--no_polar",
        action="store_true",
        help="Disable the polar LiDAR subplot (only show layout view).",
    )
    ap.add_argument(
        "--show_target_hits",
        action="store_true",
        help="Overlay rays whose hit point coincides with the scan goal (target hit) in the polar plot.",
    )
    ap.add_argument(
        "--target_hit_tol_m",
        type=float,
        default=0.15,
        help="Distance tolerance (m) used to classify a ray hit as a goal/target hit (default: 0.15).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed used when choosing a random scan_id.",
    )
    args = ap.parse_args()

    scenario_dir = os.path.join(args.root, f"scenario_{args.scenario_id:04d}")
    if not os.path.isdir(scenario_dir):
        raise SystemExit(f"Scenario directory not found: {scenario_dir}")

    # Load scenario + lidar specs from scenario_meta.json
    scen_spec = load_scenario_spec(scenario_dir)
    lidar_spec = load_lidar_spec(scenario_dir)

    # Rebuild scenario geometry
    segments, _, _, _, _ = generate_scenario(scen_spec)

    # Load scan data
    rng = np.random.default_rng(args.seed) if args.seed is not None else None
    scan = load_scan_from_csv(scenario_dir, scan_id=args.scan_id, rng=rng)
    if args.scan_id is None:
        print(f"Using randomly chosen scan_id={scan['scan_id']}")

    pose = scan["pose"]

    # Optional ideal path overlay
    scan_path: Optional[Dict[str, object]] = None
    if not args.no_path:
        scan_path = (
            build_full_route_from_scan(scenario_dir, scan["scan_id"])
            if args.full_route
            else load_scan_path(scenario_dir, scan["scan_id"])
        )

    # Build figure with layout + optional polar view
    if args.no_polar:
        fig, ax_layout = plt.subplots(figsize=(6, 6))
        draw_scenario_on_ax(
            ax_layout,
            scen_spec,
            segments,
            show_pose=pose,
            scan_path=scan_path,
            title_extra=f"scan_id={scan['scan_id']}",
        )
    else:
        fig = plt.figure(figsize=(12, 6))
        ax_layout = fig.add_subplot(1, 2, 1)
        ax_polar = fig.add_subplot(1, 2, 2, projection="polar")

        draw_scenario_on_ax(
            ax_layout,
            scen_spec,
            segments,
            show_pose=pose,
            scan_path=scan_path,
            title_extra=f"scan_id={scan['scan_id']}",
        )
        draw_lidar_rays_on_ax(ax_layout, pose, scan, lidar_spec)
        draw_polar_lidar(
            ax_polar,
            scan,
            lidar_spec,
            show_target_hits=bool(args.show_target_hits),
            target_hit_tol_m=float(args.target_hit_tol_m),
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

