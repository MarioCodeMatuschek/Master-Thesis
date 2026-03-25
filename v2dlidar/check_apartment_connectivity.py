"""
Generate a random apartment scenario, verify free-space and room-level
connectivity (every room has ≥2 connections when ≥2 adjacent rooms, else 1),
and save a visualization.

Run from the repo root (parent of v2dlidar):

    python -m v2dlidar.check_apartment_connectivity

Output: scenario_connectivity_check.png in the current working directory,
        with title showing connectivity; seed and room checks printed to stdout.
"""

import os
import random
import sys

# Use non-interactive backend when saving only (no display)
import matplotlib
matplotlib.use("Agg")

from .mapgen import ScenarioSpec, generate_scenario, get_apartment_layout_data, _room_has_door
from .planner import free_space_connected_components
from .visualize_scenario import plot_scenario


def main() -> None:
    # Random seed for a random scenario (30x30, apartment; 5 iterations => ~32 rooms for complex validation)
    seed = random.randint(0, 99999)
    spec = ScenarioSpec(
        width=30.0,
        height=30.0,
        layout="apartment",
        seed=seed,
        apt_iterations=5,
    )
    segments, scen_spec, interior, res, _ = generate_scenario(spec)
    n_comp, _ = free_space_connected_components(
        segments, scen_spec.width, scen_spec.height, res, interior=interior
    )

    rooms, doors = get_apartment_layout_data(scen_spec)
    all_rooms_connected = (
        all(_room_has_door(r, rooms, doors) for r in rooms)
        if rooms else True
    )

    out_dir = os.getcwd()
    save_path = os.path.join(out_dir, "scenario_connectivity_check.png")
    title_parts = [f"Free space: {n_comp} component(s)"]
    if all_rooms_connected:
        title_parts.append("every room ≥2 connections (or 1 if 1 neighbor)")
    else:
        title_parts.append("some room(s) missing connections!")
    title_extra = " · ".join(title_parts)
    plot_scenario(
        scen_spec,
        save_path=save_path,
        title_extra=title_extra,
    )

    print(f"Seed: {seed}")
    print(f"Rooms: {len(rooms)}, Doors: {len(doors)}")
    print(f"Free-space connectivity: {n_comp} component(s)")
    print(f"Every room has required connections (≥2 or 1): {all_rooms_connected}")
    print(f"Saved: {save_path}")
    sys.exit(0 if (n_comp == 1 and all_rooms_connected) else 1)


if __name__ == "__main__":
    main()
