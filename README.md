
# Virtual 2D LiDAR Dataset Generator (Python)

This application generates synthetic 2D LiDAR scans for complex indoor‑like environments composed of overlapped rectangles (rooms/walls). It supports:
- **Fully automated** dataset creation (random scenarios, poses, and paths)
- **Manual** start/goal specification (CLI and GUI)
- **Single-scan CSV** (long format, one row per ray)
- **Sequence JSON** (for navigation/temporal models)
- **Scenario visualization** after dataset generation

All automatically sampled and manually chosen poses are constrained to lie
inside the union of rectangles (rooms) and not on any wall/segment.

## Quick start

```bash
# Example: automated generation of 10 scenarios
python -m v2dlidar.cli auto \
  --out ./dataset \
  --scenarios 10 \
  --scans_per_scenario 500 \
  --headings_per_pose 4 \
  --seq_per_scenario 10 \
  --seq_steps 50


# Example: manual one scenario with start/goal (auto-generated paths)
python -m v2dlidar.cli manual \
  --out ./dataset \
  --scenario_id 0 \
  --start 2 2 \
  --goal 25 25 \
  --scans_per_scenario 500 \
  --seq_steps 50


## Outputs and formats

For each scenario folder OUT_DIR/scenario_XXXX/:

### scenario_meta.json
Top-level keys:

spec (ScenarioSpec):

width, height: map size in meters.
n_rects: number of rectangles used to build the union.
rect_min, rect_max: ranges for rectangle width/height.
wall_thickness: used when rasterising walls.
seed: RNG seed that uniquely defines this scenario layout.
outline_res: grid resolution used for building the outline.

lidar (LidarSpec):
fov_deg: field of view in degrees (typically 360).
️ - num_rays: number of beams per scan (typically 720).
max_range: maximum measurable range in meters.
range_noise_std: standard deviation of Gaussian range noise.
dropout_prob: probability that a valid beam is turned into a miss.


### scans_long.csv
Long-format table where each row = one LiDAR ray.

Columns:

scenario_id: integer ID of the scenario (matches folder index).
scan_id: integer ID of the scan within this scenario.
t: time stamp (currently always 0.0, reserved for temporal use).
pose_x, pose_y: robot position in meters when this scan was taken.
pose_yaw: robot heading in radians.
theta_deg: angle of this individual ray in degrees (0–360).
r_m: measured range along the ray (with noise and dropout), meters.
hit_x, hit_y: world coordinates of the hit point, empty if no valid hit.
valid: 1 if there is a finite hit, 0 if not.
noise_free_r_m: ideal range without noise/dropout, meters.

For a given `(scenario_id, scan_id)`, `pose_*` and `goal_*` are constant across all 720 rows.

### sequences/seq_XXXX.json
Each file describes a navigation sequence following an A* path inside the scenario.

Top-level fields:

scenario_id: integer ID of the scenario.
start: { "x": float, "y": float } start position in meters.
goal: { "x": float, "y": float } goal position in meters.
manual: boolean, true for manually created sequences (GUI / manual CLI), false for automated ones.
steps: list of per-time-step records:
step: integer index (0..seq_steps-1).
pose_x, pose_y, pose_yaw: robot pose at this step.
theta_deg: list of per-ray angles for the scan at this step.
ranges: list of per-ray ranges (with noise and dropout) in meters.

### `scan_paths/scan_XXXX.json` (per-scan ideal paths)

- One file per scan with a valid path.
- Fields:
  - `scenario_id`, `scan_id`
  - `start`: `{ "x", "y", "yaw" }`
  - `goal`: `{ "x", "y" }`
  - `path`: list of `{ "x", "y" }` points along the ideal A* path.

## Recommended dataset sizes

- Single-scan tasks: ~1–5k scans per scenario, hundreds of scenarios.
- Temporal tasks: 50–200 sequences per scenario, 50–500 steps each.


## Single-capture (exactly 720 rows) --> currently not available
#Create one CSV with **720 measurements** (0.5° resolution over 360°) for a single scenario and a single start/goal:
#
#```bash
#python -m v2dlidar.cli single --out ./dataset --scenario_id 0 --start 5 5 0.0 --goal 25 25
#```
#
#Outputs `scenario_0000/single_capture.csv` with columns:
#
#scenario_id: integer scenario ID.
#start_x, start_y, start_yaw: start pose (position in meters, yaw in radians).
#goal_x, goal_y: goal position in meters.
#angle_deg: per-ray angle in degrees (0–360).
#r_m: measured range (with noise/dropout) in meters.
#hit_x, hit_y: per-ray hit coordinates in meters, empty if no valid hit.
#valid: 1 if a valid hit exists, 0 otherwise.

## Sequences and `seq_steps`

Two main data types:

- **Single scans** (`scans_long.csv`):
  - Each `scan_id` is one LiDAR sweep at a fixed pose.
  - Good for static tasks (range prediction, occupancy mapping, obstacle detection).

- **Sequences** (`sequences/seq_XXXX.json`):
  - Each file is a trajectory from a start to a goal inside a scenario.
  - The path is computed with A*, then subsampled to `seq_steps` points.
  - At each point, a LiDAR scan is simulated and stored as a `step`.

`seq_steps` controls **how many samples you take along a fixed path**:

- Larger `seq_steps` → more steps per sequence, finer sampling, longer sequences.
- Smaller `seq_steps` → fewer steps, coarser sampling, shorter sequences.

Use sequences when you care about how LiDAR observations evolve over time (navigation, temporal models).  
Use single scans when you only need independent snapshots.

## Manual GUI (pick start/goal and append to dataset)
You can launch a simple Tkinter + Matplotlib GUI to visualize the scenario complexity,
pick a **Start** (left click) and **Goal** (right click), and append the resulting
**manual sequence** and a **single-capture CSV** to the dataset folder.

```bash
python -m v2dlidar.gui_manual
```

- Use **Generate/Refresh** to change the scene (via seed / rectangle count / size).
- Complexity metrics shown: number of segments and wall density.
- After picking Start and Goal, click **Append to Dataset (Manual)**.
- Files are saved under `OUT_DIR/scenario_XXXX/`:
  - `sequences/seq_XXXX_manual.json` (next free index)
  - `single_capture.csv` (720 rows @ 0.5°)


### Path preview toggle
In the Manual GUI, enable/disable the **Preview planned path** checkbox to overlay the A* path
between your Start and Goal before saving. The preview recomputes automatically whenever you pick Start/Goal
or regenerate the scenario. All clicks for Start/Goal in the GUI, as well as automatically sampled
poses in the CLI, are constrained to the free interior of the environment:
inside the union of rectangles and away from walls/obstacles.


### Scenario Visualization

After generating a dataset you can visualize any scenario (walls, pose, and optional ideal path):

```bash
# Just visualize walls and the first scan pose
python -m v2dlidar.visualize_scenario --root ./dataset_10x1 --scenario_id 0

To visualize the ideal path and start/goal for a specific scan (as created during automatic dataset generation), use the --scan_id option:

# Walls + ideal path + start/goal for scan_id 0
python -m v2dlidar.visualize_scenario --root ./dataset_10x1 --scenario_id 0 --scan_id 0

If you only want the walls and ideal path (no pose arrow), add --no_pose:

python -m v2dlidar.visualize_scenario --root ./dataset_10x1 --scenario_id 0 --scan_id 0 --no_pose

--root: points to the dataset root containing scenario_XXXX folders (e.g. ./dataset_10x1).
--scenario_id: selects which scenario to inspect (e.g. 0 for scenario_0000).
--scan_id: selects which scan’s ideal path to overlay using scan_paths/scan_XXXX.json.



ToDO:


1. Lidar Similieren der aktuellen virtuellen Map
  hierbei auch berücksichtigen dass die measurements noisy (Gaussian) sein müssen
  potenziel muss auch noch vibration noise, motor harmonics berücksichtigt werden

(Siehe Codex suggestion left) 2. Die Szenarien "einfacher" machen (wie eine Art Floorplan einer Wohnung von oben)

3. Mit Joe real-life scenarios nachstellen

4. diese dann virtuell abbilden und dann im practical work abgleich
  vermutung - starke unterschiede (virtuell womöglich viel ausfall)

DONE 5. Metrik durchdenken für den abgleich physisch vs. virtuell