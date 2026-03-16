
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math, random, os
import numpy as np
from .mapgen import ScenarioSpec, generate_scenario
from .lidar import LidarSpec, LidarSimulator
from .planner import occupancy_from_segments, astar_path
from .utils import ensure_dir, write_scan_long_csv, write_json, save_npy

@dataclass
class AutoGenConfig:
    scenarios: int = 100
    scans_per_scenario: int = 2000
    poses_per_scan: int = 1  # leave at 1 unless you want multi-pose averaging
    headings_per_pose: int = 4
    seq_per_scenario: int = 50
    seq_steps: int = 100
    grid_res: float = 0.1
    clearance: float = 0.2
    seed: int = 0

class DatasetGenerator:
    def __init__(self, out_dir: str, lidar: LidarSpec, scen: ScenarioSpec, cfg: AutoGenConfig):
        self.out_dir = out_dir
        self.lidar_spec = lidar
        self.scen_spec = scen
        self.cfg = cfg
        ensure_dir(self.out_dir)

    def _random_free_pose(self, occ, res, rng):
        H, W = occ.shape
        while True:
            gx = rng.randrange(W); gy = rng.randrange(H)
            if occ[gy,gx] == 0:
                x, y = gx*res, gy*res
                yaw = rng.uniform(-math.pi, math.pi)
                return (x, y, yaw)

    def _sample_goal_far(self, occ, res, start_xy, rng):
        H, W = occ.shape
        for _ in range(1000):
            gx = rng.randrange(W); gy = rng.randrange(H)
            if occ[gy,gx]: continue
            x, y = gx*res, gy*res
            dx, dy = x - start_xy[0], y - start_xy[1]
            if dx*dx + dy*dy > (min(W,H)*res*0.25)**2:
                return (x, y)
        return (start_xy[0]+2.0, start_xy[1]+2.0)

    def generate_scenario_data(self, scenario_id: int, manual_start: Optional[Tuple[float,float]]=None, manual_goal: Optional[Tuple[float,float]]=None, seed_offset:int=0):
        rng = random.Random(self.cfg.seed + scenario_id + seed_offset)
        spec_kwargs = vars(self.scen_spec).copy()
        spec_kwargs["seed"] = self.scen_spec.seed + scenario_id
        # Generate geometry and interior mask (union of rectangles)
        segments, spec, interior, res = generate_scenario(ScenarioSpec(**spec_kwargs))
        # Occupancy from walls only
        occ_walls, _ = occupancy_from_segments(segments, spec.width, spec.height, res)
        # Navigation occupancy: 1 = forbidden (wall or outside union), 0 = free (inside union, off walls)
        occ = occ_walls.copy()
        occ[interior == 0] = 1
        # Randomize LiDAR noise parameters per scenario in a reproducible way
        lidar_spec = LidarSpec(
            fov_deg=self.lidar_spec.fov_deg,
            num_rays=self.lidar_spec.num_rays,
            max_range=self.lidar_spec.max_range,
            range_noise_std=rng.uniform(0.005, 0.05),
            dropout_prob=rng.uniform(0.01, 0.2),
        )
        lidar = LidarSimulator(lidar_spec, segments)
        scen_dir = os.path.join(self.out_dir, f"scenario_{scenario_id:04d}")
        ensure_dir(scen_dir)
        write_json(os.path.join(scen_dir, "scenario_meta.json"), {"spec": vars(spec), "lidar": vars(lidar_spec)})

        # Single-scan dataset
        scans_header = [
            "scenario_id",
            "scan_id",
            "t",
            "pose_x",
            "pose_y",
            "pose_yaw",
            "goal_x",
            "goal_y",
            "theta_deg",
            "r_m",
            "hit_x",
            "hit_y",
            "valid",
            "noise_free_r_m",
        ]
        scan_paths_dir = os.path.join(scen_dir, "scan_paths")
        ensure_dir(scan_paths_dir)

        def scan_rows():
            scan_id = 0
            # For each randomly chosen pose, we sample one goal and one path and
            # reuse them across all headings_per_pose. Only the yaw (and thus
            # the LiDAR scan) changes between headings at the same pose.
            for _ in range(self.cfg.scans_per_scenario):
                pose = self._random_free_pose(occ, res, rng)
                x, y, _ = pose
                start_xy = (x, y)
                # Sample a goal and ensure there is a valid path; retry a number
                # of times so we are likely to get at least one scan for this pose.
                goal = None
                path = None
                max_goal_tries = 50
                for _ in range(max_goal_tries):
                    candidate_goal = self._sample_goal_far(occ, res, start_xy, rng)
                    candidate_path = astar_path(occ, start_xy, candidate_goal, res)
                    if candidate_path is not None and len(candidate_path) >= 2:
                        goal = candidate_goal
                        path = candidate_path
                        break
                if goal is None or path is None:
                    # Could not find a valid path for this pose; skip it.
                    continue
                # Save ideal geometric path for this pose/goal pair once; it will
                # be reused across all headings at this pose.
                path_points = [{"x": float(px), "y": float(py)} for (px, py) in path]
                for _ in range(self.cfg.headings_per_pose):
                    yaw = rng.uniform(-math.pi, math.pi)
                    write_json(
                        os.path.join(scan_paths_dir, f"scan_{scan_id:04d}.json"),
                        {
                            "scenario_id": scenario_id,
                            "scan_id": int(scan_id),
                            "start": {"x": float(x), "y": float(y), "yaw": float(yaw)},
                            "goal": {"x": float(goal[0]), "y": float(goal[1])},
                            "path": path_points,
                        },
                    )
                    # Run LiDAR scans for this pose/yaw
                    s = lidar.scan((x, y, yaw), rng=rng, noise_free=False)
                    s_nf = lidar.scan((x, y, yaw), rng=rng, noise_free=True)
                    for i in range(len(s["theta_deg"])):
                        hit_x = s["hits"][i,0]; hit_y = s["hits"][i,1]
                        valid = int(np.isfinite(hit_x))
                        yield [
                            scenario_id,
                            scan_id,
                            0.0,
                            x,
                            y,
                            yaw,
                            float(goal[0]),
                            float(goal[1]),
                            float(s["theta_deg"][i]),
                            float(s["ranges"][i]),
                            float(hit_x) if valid else "",
                            float(hit_y) if valid else "",
                            valid,
                            float(s_nf["noise_free_ranges"][i]),
                        ]
                    scan_id += 1
        write_scan_long_csv(os.path.join(scen_dir, "scans_long.csv"), scans_header, scan_rows())

        # Sequences (automated or manual)
        seq_dir = os.path.join(scen_dir, "sequences")
        ensure_dir(seq_dir)
        for si in range(self.cfg.seq_per_scenario):
            if manual_start is not None and manual_goal is not None and si == 0:
                start = manual_start; goal = manual_goal
            else:
                s_pose = self._random_free_pose(occ, res, rng); start = (s_pose[0], s_pose[1])
                goal = self._sample_goal_far(occ, res, start, rng)
            path = astar_path(occ, start, goal, res)
            if path is None or len(path) < 2:
                continue
            # subsample path to seq_steps
            idx = np.linspace(0, len(path)-1, self.cfg.seq_steps).astype(int)
            path_seq = [path[i] for i in idx]
            seq_records = []
            for step, (x,y) in enumerate(path_seq):
                # approximate local yaw towards next waypoint
                if step < len(path_seq)-1:
                    nx, ny = path_seq[step+1]
                    yaw = math.atan2(ny-y, nx-x)
                else:
                    yaw = 0.0
                s = lidar.scan((x,y,yaw), rng=rng, noise_free=False)
                rec = {
                    "step": int(step),
                    "pose_x": float(x), "pose_y": float(y), "pose_yaw": float(yaw),
                    "theta_deg": s["theta_deg"].tolist(),
                    "ranges": s["ranges"].tolist()
                }
                seq_records.append(rec)
            write_json(os.path.join(seq_dir, f"seq_{si:04d}.json"), {
                "scenario_id": scenario_id,
                "start": {"x": start[0], "y": start[1]},
                "goal": {"x": goal[0], "y": goal[1]},
                "steps": seq_records
            })

    def generate_automated(self, start_scenario_id: int = 0, count: int = 10):
        for sid in range(start_scenario_id, start_scenario_id + count):
            self.generate_scenario_data(sid)

    def generate_manual(self, scenario_id: int, start: Tuple[float,float], goal: Tuple[float,float]):
        self.generate_scenario_data(scenario_id, manual_start=start, manual_goal=goal, seed_offset=999)

    def generate_single_capture(self, scenario_id: int, start: Tuple[float,float,float], goal: Tuple[float,float]):
        """Create a single-scan CSV with exactly 720 measurements at 0.5° (assuming LidarSpec matches).
        Output file: single_capture.csv with 720 rows containing:
          scenario_id, start_x, start_y, start_yaw, goal_x, goal_y, angle_deg, r_m, hit_x, hit_y, valid
        """
        import numpy as np, math

        spec_kwargs = vars(self.scen_spec).copy()
        spec_kwargs["seed"] = self.scen_spec.seed + scenario_id
        # Force a deterministic scenario using scenario_id-based seed
        segments, spec, interior, res = generate_scenario(ScenarioSpec(**spec_kwargs))
        # Ensure LiDAR has 360° and 720 rays (0.5° resolution) and randomize noise per scenario
        from .lidar import LidarSpec, LidarSimulator
        base_spec = self.lidar_spec
        if abs(base_spec.fov_deg - 360.0) > 1e-6 or base_spec.num_rays != 720:
            base_spec = LidarSpec(
                fov_deg=360.0,
                num_rays=720,
                max_range=self.lidar_spec.max_range,
                range_noise_std=self.lidar_spec.range_noise_std,
                dropout_prob=self.lidar_spec.dropout_prob,
            )
        rng = random.Random(self.cfg.seed + scenario_id)
        lidar_spec = LidarSpec(
            fov_deg=base_spec.fov_deg,
            num_rays=base_spec.num_rays,
            max_range=base_spec.max_range,
            range_noise_std=rng.uniform(0.005, 0.05),
            dropout_prob=rng.uniform(0.01, 0.2),
        )
        # Build interior-aware occupancy to validate start/goal
        occ_walls, _ = occupancy_from_segments(segments, spec.width, spec.height, res)
        occ = occ_walls.copy()
        occ[interior == 0] = 1
        # Validate that start and goal are inside free interior (not walls / outside)
        H, W = occ.shape
        sx, sy, syaw = start
        gx = int(np.clip(sx / res, 0, W - 1))
        gy = int(np.clip(sy / res, 0, H - 1))
        gx_goal = int(np.clip(goal[0] / res, 0, W - 1))
        gy_goal = int(np.clip(goal[1] / res, 0, H - 1))
        if occ[gy, gx] or occ[gy_goal, gx_goal]:
            raise ValueError("Start and goal must lie inside the interior of the scenario and not on walls.")
        lidar = LidarSimulator(lidar_spec, segments)
        # Run one scan from the provided start pose
        s = lidar.scan((sx, sy, syaw), noise_free=False)
        out_dir = os.path.join(self.out_dir, f"scenario_{scenario_id:04d}")
        from .utils import ensure_dir, write_scan_long_csv, write_json
        ensure_dir(out_dir)
        # Write metadata with scenario + start/goal used
        write_json(os.path.join(out_dir, "single_capture_meta.json"), {
            "scenario_id": scenario_id,
            "scenario_spec": vars(spec),
            "lidar_spec": vars(lidar_spec),
            "start": {"x": sx, "y": sy, "yaw": syaw},
            "goal": {"x": goal[0], "y": goal[1]}
        })
        # Compose rows
        header = ["scenario_id","start_x","start_y","start_yaw","goal_x","goal_y","angle_deg","r_m","hit_x","hit_y","valid"]
        def rows():
            for i in range(len(s["theta_deg"])):
                ang = float(s["theta_deg"][i])
                r = float(s["ranges"][i])
                hx = float(s["hits"][i,0]) if not np.isnan(s["hits"][i,0]) else ""
                hy = float(s["hits"][i,1]) if not np.isnan(s["hits"][i,1]) else ""
                valid = int(hx != "" and hy != "")
                yield [scenario_id, sx, sy, syaw, goal[0], goal[1], ang, r, hx, hy, valid]
        write_scan_long_csv(os.path.join(out_dir, "single_capture.csv"), header, rows())
        return os.path.join(out_dir, "single_capture.csv")

    def append_manual_sequence(self, scenario_id: int, start: Tuple[float,float], goal: Tuple[float,float], seq_steps: int = None):
        """Append a single manual sequence file to an existing or new scenario folder.
        It finds the next available sequence index and writes seq_XXXX_manual.json
        without regenerating random auto sequences. Also creates single_capture.csv for the
        chosen start pose with yaw toward first step.
        """
        import numpy as np, math, os
        if seq_steps is None:
            seq_steps = self.cfg.seq_steps
        # Recreate deterministic scenario and planner with interior-aware occupancy
        spec_kwargs = vars(self.scen_spec).copy()
        spec_kwargs["seed"] = self.scen_spec.seed + scenario_id
        segments, spec, interior, res = generate_scenario(ScenarioSpec(**spec_kwargs))
        occ_walls, _ = occupancy_from_segments(segments, spec.width, spec.height, res)
        occ = occ_walls.copy()
        occ[interior == 0] = 1
        # Plan path
        path = astar_path(occ, start, goal, res)
        scen_dir = os.path.join(self.out_dir, f"scenario_{scenario_id:04d}")
        os.makedirs(scen_dir, exist_ok=True)
        seq_dir = os.path.join(scen_dir, "sequences")
        os.makedirs(seq_dir, exist_ok=True)
        if path is None or len(path) < 2:
            # Still write a meta error file
            write_json(os.path.join(seq_dir, "last_manual_failed.json"), {
                "reason": "no_path",
                "scenario_id": scenario_id,
                "start": {"x": start[0], "y": start[1]},
                "goal": {"x": goal[0], "y": goal[1]}
            })
            return None
        # Subsample path
        idx = np.linspace(0, len(path)-1, seq_steps).astype(int)
        path_seq = [path[i] for i in idx]
        # Determine next index
        existing = [fn for fn in os.listdir(seq_dir) if fn.startswith("seq_") and fn.endswith(".json")]
        next_idx = 0
        if existing:
            nums = []
            for fn in existing:
                try:
                    n = int(fn.split("_")[1].split(".")[0])
                    nums.append(n)
                except:
                    pass
            if nums:
                next_idx = max(nums) + 1
        # Build lidar and records with per-scenario randomized noise (deterministic from cfg.seed and scenario_id)
        rng = random.Random(self.cfg.seed + scenario_id)
        lidar_spec = LidarSpec(
            fov_deg=self.lidar_spec.fov_deg,
            num_rays=self.lidar_spec.num_rays,
            max_range=self.lidar_spec.max_range,
            range_noise_std=rng.uniform(0.005, 0.05),
            dropout_prob=rng.uniform(0.01, 0.2),
        )
        lidar = LidarSimulator(lidar_spec, segments)
        steps = []
        start_yaw = 0.0
        for step, (x,y) in enumerate(path_seq):
            if step < len(path_seq)-1:
                nx, ny = path_seq[step+1]
                yaw = math.atan2(ny-y, nx-x)
                if step == 0: start_yaw = yaw
            else:
                yaw = 0.0
            s = lidar.scan((x,y,yaw), noise_free=False)
            steps.append({
                "step": int(step),
                "pose_x": float(x), "pose_y": float(y), "pose_yaw": float(yaw),
                "theta_deg": s["theta_deg"].tolist(),
                "ranges": s["ranges"].tolist()
            })
        # Write sequence
        out_path = os.path.join(seq_dir, f"seq_{next_idx:04d}_manual.json")
        write_json(out_path, {
            "scenario_id": scenario_id,
            "start": {"x": start[0], "y": start[1]},
            "goal": {"x": goal[0], "y": goal[1]},
            "manual": True,
            "steps": steps
        })
        # Also create/update single-capture file for this manual selection
        self.generate_single_capture(scenario_id, (start[0], start[1], start_yaw), goal)
        return out_path
