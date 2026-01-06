
from v2dlidar.cli import main

if __name__ == "__main__":
    import sys
    sys.argv = ["", "auto", "--out", "./demo_out", "--scenarios", "1", "--scans_per_scenario", "50", "--seq_per_scenario", "2", "--seq_steps", "20"]
    main()
