
import argparse, os, json
from .lidar import LidarSpec
from .mapgen import ScenarioSpec
from .dataset import DatasetGenerator, AutoGenConfig

def main():
    ap = argparse.ArgumentParser(description="Virtual 2D LiDAR Dataset Generator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--layout", type=str, default="union", choices=["union", "apartment"])
    common.add_argument("--apt_rows", type=int, default=2)
    common.add_argument("--apt_cols", type=int, default=3)
    common.add_argument("--apt_door_prob", type=float, default=0.8)
    common.add_argument("--out", type=str, default="out_dataset")
    common.add_argument("--width", type=float, default=30.0)
    common.add_argument("--height", type=float, default=30.0)
    common.add_argument("--n_rects", type=int, default=20)
    common.add_argument("--seed", type=int, default=0)

    lid = argparse.ArgumentParser(add_help=False)
    lid.add_argument("--fov_deg", type=float, default=360.0)
    lid.add_argument("--num_rays", type=int, default=720)
    lid.add_argument("--max_range", type=float, default=12.0)
    lid.add_argument("--noise_std", type=float, default=0.02)
    lid.add_argument("--dropout", type=float, default=0.01)

    auto = sub.add_parser("auto", parents=[common, lid])
    auto.add_argument("--scenarios", type=int, default=10)
    auto.add_argument("--scans_per_scenario", type=int, default=2000)
    auto.add_argument("--headings_per_pose", type=int, default=4)
    auto.add_argument("--seq_per_scenario", type=int, default=50)
    auto.add_argument("--seq_steps", type=int, default=100)
    auto.add_argument("--grid_res", type=float, default=0.1)

    manual = sub.add_parser("manual", parents=[common, lid])

    single = sub.add_parser("single", parents=[common, lid])
    single.add_argument("--scenario_id", type=int, default=0)
    single.add_argument("--start", type=float, nargs=3, metavar=("X","Y","YAW"), required=True, help="Start pose (x y yaw_rad)")
    single.add_argument("--goal", type=float, nargs=2, metavar=("X","Y"), required=True, help="Goal position (x y)")
    
    manual.add_argument("--scenario_id", type=int, required=True)
    manual.add_argument("--start", type=float, nargs=2, metavar=("X","Y"), required=True)
    manual.add_argument("--goal", type=float, nargs=2, metavar=("X","Y"), required=True)
    manual.add_argument("--scans_per_scenario", type=int, default=2000)
    manual.add_argument("--headings_per_pose", type=int, default=4)
    manual.add_argument("--seq_steps", type=int, default=100)
    manual.add_argument("--grid_res", type=float, default=0.1)

    args = ap.parse_args()
    scen = ScenarioSpec(width=args.width, height=args.height, n_rects=args.n_rects, seed=args.seed, layout=args.layout, apt_rows=args.apt_rows, apt_cols=args.apt_cols, apt_door_prob=args.apt_door_prob,)
    lidar = LidarSpec(fov_deg=args.fov_deg, num_rays=args.num_rays, max_range=args.max_range, range_noise_std=args.noise_std, dropout_prob=args.dropout)
    cfg = AutoGenConfig(
        scenarios=getattr(args, "scenarios", 1),
        scans_per_scenario=args.scans_per_scenario,
        headings_per_pose=args.headings_per_pose,
        seq_per_scenario=getattr(args, "seq_per_scenario", 1 if args.__dict__.get("scenario_id", None) is not None else 50),
        seq_steps=args.seq_steps,
        grid_res=args.grid_res,
        seed=args.seed
    )
    gen = DatasetGenerator(args.out, lidar, scen, cfg)
    if args.cmd == "auto":
        gen.generate_automated(0, args.scenarios)
    elif args.cmd == "manual":
        gen.generate_manual(args.scenario_id, tuple(args.start), tuple(args.goal))

if __name__ == "__main__":
    main()

#elif args.cmd == "single":
#    # Generate a scenario (or reuse) and produce exactly one scan with 720 measurements at 0.5°
#    gen.generate_single_capture(args.scenario_id, (args.start[0], args.start[1], args.start[2]), (args.goal[0], args.goal[1]))
