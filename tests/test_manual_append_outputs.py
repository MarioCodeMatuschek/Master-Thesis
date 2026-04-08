import json

import pytest

from v2dlidar.dataset import AutoGenConfig, DatasetGenerator
from v2dlidar.lidar import LidarSpec
from v2dlidar.mapgen import ScenarioSpec, generate_scenario
from v2dlidar.planner import occupancy_from_segments


def _first_free_xy(occ, res):
    import numpy as np

    ys, xs = (occ == 0).nonzero()
    assert len(xs) > 0
    gx = int(xs[0])
    gy = int(ys[0])
    return (gx * res, gy * res)


def test_manual_append_writes_auto_style_outputs(tmp_path):
    out_dir = str(tmp_path)
    scen = ScenarioSpec(layout="apartment", width=30.0, height=30.0, seed=0, apt_iterations=2)
    lidar = LidarSpec()
    cfg = AutoGenConfig(seed=0)
    gen = DatasetGenerator(out_dir, lidar, scen, cfg)

    segments, spec, interior, res, exterior_goal = generate_scenario(scen)
    assert exterior_goal is not None
    occ_walls, _ = occupancy_from_segments(segments, spec.width, spec.height, res)
    occ = occ_walls.copy()
    occ[interior == 0] = 1

    start = _first_free_xy(occ, res)
    goal = exterior_goal

    out = gen.append_manual_scans_auto_style(0, start, goal, waypoints=[])
    assert out is not None

    scen_dir = tmp_path / "scenario_0000"
    assert (scen_dir / "scenario_meta.json").exists()
    assert (scen_dir / "scans_long.csv").exists()
    assert (scen_dir / "scan_paths" / "scan_0000.json").exists()
    assert (scen_dir / "scan_paths" / "scan_0001.json").exists()
    assert (scen_dir / "scan_paths" / "scan_0002.json").exists()

    # scans_long.csv: header + 3 sweeps * num_rays rows
    num_rays = lidar.num_rays
    with open(scen_dir / "scans_long.csv", "r", encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)
    assert n_lines == 1 + 3 * num_rays

    with open(scen_dir / "scan_paths" / "scan_0000.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert set(data.keys()) == {"scenario_id", "scan_id", "start", "goal", "path"}
    assert set(data["start"].keys()) == {"x", "y", "yaw"}
    assert set(data["goal"].keys()) == {"x", "y"}

