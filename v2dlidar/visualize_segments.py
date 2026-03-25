"""
CLI tool to visualize raw wall segments for a generated scenario.

This is intended for debugging geometry used by the LiDAR simulator and
planner. It shows:

- All wall segments returned by `generate_scenario`.
- Optionally, the apartment rooms + doors overlay, to compare room
  rectangles with the actual segments used for LiDAR.
- Optionally, a side-by-side figure:
  - Left: existing room/door visualization (like visualize_scenario).
  - Right: raw segments only.
"""

import argparse
import os

import matplotlib.pyplot as plt

from .mapgen import ScenarioSpec, generate_scenario, get_apartment_layout_data
from .visualize_scenario import load_scenario_spec, draw_scenario_on_ax


def plot_segments_on_ax(ax, segments, spec: ScenarioSpec, title: str | None = None) -> None:
    """Draw all wall segments onto an existing Axes."""
    for s in segments:
        ax.plot([s.x1, s.x2], [s.y1, s.y2], color="black", linewidth=1)

    ax.set_xlim(0, spec.width)
    ax.set_ylim(0, spec.height)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    if title:
        ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def plot_segments_with_optional_rooms(
    ax,
    segments,
    spec: ScenarioSpec,
    overlay_rooms: bool = True,
    title: str | None = None,
) -> None:
    """Draw raw segments, optionally overlaying apartment rooms/doors."""
    # First optionally draw rooms/doors for apartment layout
    if overlay_rooms and getattr(spec, "layout", "union") == "apartment":
        layout_data = get_apartment_layout_data(spec)
        if layout_data is not None:
            rooms, doors = layout_data
            for r in rooms:
                ax.add_patch(
                    plt.Rectangle(
                        (r.x, r.y),
                        r.width,
                        r.height,
                        linewidth=1.5,
                        edgecolor="#666666",
                        facecolor="#f0f0f0",
                        alpha=0.4,
                    )
                )
            for dx, dy, orientation, width_used in doors:
                half_door = width_used / 2.0
                if orientation == "vertical":
                    ax.plot(
                        [dx, dx],
                        [dy - half_door, dy + half_door],
                        color="white",
                        linewidth=3,
                        zorder=3,
                    )
                else:
                    ax.plot(
                        [dx - half_door, dx + half_door],
                        [dy, dy],
                        color="white",
                        linewidth=3,
                        zorder=3,
                    )

    # Then draw segments on top
    for s in segments:
        ax.plot([s.x1, s.x2], [s.y1, s.y2], color="red", linewidth=1.0)

    ax.set_xlim(0, spec.width)
    ax.set_ylim(0, spec.height)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    if title:
        ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def main() -> None:
    """Entry point for `python -m v2dlidar.visualize_segments`."""
    ap = argparse.ArgumentParser(
        description=(
            "Visualize raw wall segments for a generated 2D LiDAR scenario "
            "(with optional room/door overlay and side-by-side comparison)."
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
        help="Numeric scenario ID to visualize (e.g. 0 for scenario_0000).",
    )
    ap.add_argument(
        "--segments_only",
        action="store_true",
        help="Show only raw segments (no room/door overlay, no side-by-side).",
    )
    ap.add_argument(
        "--no_rooms",
        action="store_true",
        help="Do not overlay rooms/doors on the segments view.",
    )
    ap.add_argument(
        "--side_by_side",
        action="store_true",
        help=(
            "Show side-by-side figure: left = existing room/door layout, "
            "right = raw segments."
        ),
    )
    args = ap.parse_args()

    scenario_dir = os.path.join(args.root, f"scenario_{args.scenario_id:04d}")
    if not os.path.isdir(scenario_dir):
        raise SystemExit(f"Scenario directory not found: {scenario_dir}")

    # Load spec and generate segments
    scen_spec = load_scenario_spec(scenario_dir)
    segments, _, _, _, _ = generate_scenario(scen_spec)

    if args.side_by_side and not args.segments_only:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
        # Left: existing scenario visualization (rooms/doors view)
        draw_scenario_on_ax(
            ax_left,
            scen_spec,
            segments,
            show_pose=None,
            scan_path=None,
            title_extra=f"scenario_id={args.scenario_id} (rooms/doors view)",
        )
        # Right: raw segments with optional room overlay
        overlay_rooms = not args.no_rooms
        plot_segments_with_optional_rooms(
            ax_right,
            segments,
            scen_spec,
            overlay_rooms=overlay_rooms,
            title=f"scenario_id={args.scenario_id} (raw segments)",
        )
        plt.tight_layout()
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        if args.segments_only:
            plot_segments_on_ax(
                ax,
                segments,
                scen_spec,
                title=f"scenario_id={args.scenario_id} (segments only)",
            )
        else:
            overlay_rooms = not args.no_rooms
            plot_segments_with_optional_rooms(
                ax,
                segments,
                scen_spec,
                overlay_rooms=overlay_rooms,
                title=f"scenario_id={args.scenario_id} (segments + rooms)",
            )
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

