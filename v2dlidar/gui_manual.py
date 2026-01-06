import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np, math, os, random
from .mapgen import ScenarioSpec, generate_scenario
from .planner import occupancy_from_segments, astar_path
from .dataset import DatasetGenerator, AutoGenConfig
from .lidar import LidarSpec

class ManualGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Virtual 2D LiDAR - Manual Scenario GUI")
        # Parameters frame
        params = ttk.Frame(self)
        params.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        # Scenario controls
        ttk.Label(params, text="Output dir").pack(anchor="w")
        self.out_var = tk.StringVar(value="./dataset")
        ttk.Entry(params, textvariable=self.out_var, width=28).pack(anchor="w")
        ttk.Label(params, text="Scenario ID").pack(anchor="w")
        self.scen_id_var = tk.IntVar(value=0)
        ttk.Entry(params, textvariable=self.scen_id_var, width=10).pack(anchor="w")
        ttk.Label(params, text="Width, Height").pack(anchor="w")
        self.width_var = tk.DoubleVar(value=30.0)
        self.height_var = tk.DoubleVar(value=30.0)
        row = ttk.Frame(params)
        row.pack(anchor="w")
        ttk.Entry(row, textvariable=self.width_var, width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.height_var, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Label(params, text="# Rectangles").pack(anchor="w")
        self.nrect_var = tk.IntVar(value=20)
        ttk.Entry(params, textvariable=self.nrect_var, width=10).pack(anchor="w")
        ttk.Label(params, text="Seed").pack(anchor="w")
        self.seed_var = tk.IntVar(value=0)
        ttk.Entry(params, textvariable=self.seed_var, width=10).pack(anchor="w")
        ttk.Button(params, text="Generate/Refresh", command=self.refresh).pack(anchor="w", pady=6)
        ttk.Separator(params, orient="horizontal").pack(fill="x", pady=6)

        ttk.Label(params, text="Layout").pack(anchor="w")
        self.layout_var = tk.StringVar(value="union")
        ttk.Combobox(
            params,
            textvariable=self.layout_var,
            values=["union", "apartment"],
            state="readonly",
            width=12,
        ).pack(anchor="w")

        ttk.Label(params, text="Apartment rows, cols").pack(anchor="w")
        self.apt_rows_var = tk.IntVar(value=2)
        self.apt_cols_var = tk.IntVar(value=3)
        row_apt = ttk.Frame(params)
        row_apt.pack(anchor="w")
        ttk.Entry(row_apt, textvariable=self.apt_rows_var, width=5).pack(side=tk.LEFT)
        ttk.Entry(row_apt, textvariable=self.apt_cols_var, width=5).pack(
            side=tk.LEFT, padx=4
        )

        ttk.Label(params, text="Apartment door prob").pack(anchor="w")
        self.apt_door_prob_var = tk.DoubleVar(value=0.8)
        ttk.Entry(params, textvariable=self.apt_door_prob_var, width=10).pack(
            anchor="w"
        )


        # Lidar & sequence controls
        ttk.Label(params, text="Seq steps").pack(anchor="w")
        self.seq_steps_var = tk.IntVar(value=100)
        ttk.Entry(params, textvariable=self.seq_steps_var, width=10).pack(anchor="w")

        # Preview toggle
        self.preview_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            params,
            text="Preview planned path",
            variable=self.preview_var,
            command=self.draw_scene,
        ).pack(anchor="w")

        # Complexity metrics
        ttk.Separator(params, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(params, text="Complexity").pack(anchor="w")
        self.comp_text = tk.StringVar(value="?")
        ttk.Label(params, textvariable=self.comp_text, justify="left").pack(anchor="w")

        # Actions
        ttk.Separator(params, orient="horizontal").pack(fill="x", pady=6)
        ttk.Button(
            params, text="Append to Dataset (Manual)", command=self.append_dataset
        ).pack(anchor="w", pady=8)
        self.status = tk.StringVar(
            value="Click to set Start (left click) and Goal (right click)"
        )
        ttk.Label(params, textvariable=self.status, wraplength=220).pack(
            anchor="w", pady=6
        )

        # Plot
        fig = Figure(figsize=(6, 6), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_aspect("equal")
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        self.canvas.mpl_connect("button_press_event", self.on_click)

        # State
        self.segments = []
        self.spec = None
        self.interior = None
        self.occ = None
        self.grid_res = 0.1
        self.start = None
        self.goal = None
        self.preview_path = None
        self.refresh()

    def refresh(self):
        spec = ScenarioSpec(
            width=self.width_var.get(),
            height=self.height_var.get(),
            n_rects=self.nrect_var.get(),
            seed=self.seed_var.get(),
            layout=self.layout_var.get(),
            apt_rows=self.apt_rows_var.get(),
            apt_cols=self.apt_cols_var.get(),
            apt_door_prob=self.apt_door_prob_var.get(),
        )
        # Geometry + interior mask (union of rectangles)
        self.segments, self.spec, self.interior, self.grid_res = generate_scenario(spec)
        # Navigation occupancy: walls and outside-union are forbidden
        occ_walls, _ = occupancy_from_segments(
            self.segments, spec.width, spec.height, res=self.grid_res
        )
        self.occ = occ_walls.copy()
        self.occ[self.interior == 0] = 1
        # Recompute preview path if both points are available
        if self.start and self.goal:
            self.compute_preview_path()
        self.draw_scene()
        # Complexity: wall density from occupancy + segment count
        occ, res = occupancy_from_segments(
            self.segments, spec.width, spec.height, res=0.1
        )
        density = float(np.mean(occ))
        comp = (
            f"Segments: {len(self.segments)}\n"
            f"Wall density (0-1): {density:.3f}\n"
            f"Size: {spec.width:.1f} x {spec.height:.1f}"
        )
        self.comp_text.set(comp)
        self.start = None
        self.goal = None
        self.preview_path = None
        self.status.set("Pick Start (left click) and Goal (right click)")

    def draw_scene(self):
        self.ax.clear()
        if self.spec is not None:
            self.ax.set_xlim(0, self.spec.width)
            self.ax.set_ylim(0, self.spec.height)
        for s in self.segments:
            self.ax.plot([s.x1, s.x2], [s.y1, s.y2], linewidth=1)
        if self.preview_var.get() and self.preview_path:
            xs = [p[0] for p in self.preview_path]
            ys = [p[1] for p in self.preview_path]
            self.ax.plot(xs, ys, color="red", linewidth=1.5)
        if self.start:
            self.ax.plot(self.start[0], self.start[1], marker="o", markersize=8)
        if self.goal:
            self.ax.plot(self.goal[0], self.goal[1], marker="x", markersize=10)
        self.ax.invert_yaxis()  # visually like a map editor
        self.ax.set_title(f"Scenario {self.scen_id_var.get()}")
        self.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        x, y = float(event.xdata), float(event.ydata)
        # Enforce picking only inside union of rectangles and off walls
        if self.occ is not None and self.grid_res is not None:
            H, W = self.occ.shape
            gx = int(np.clip(x / self.grid_res, 0, W - 1))
            gy = int(np.clip(y / self.grid_res, 0, H - 1))
            if self.occ[gy, gx]:
                self.status.set("Please click inside free interior (not on walls or outside).")
                return
        if event.button == 1:  # left -> start
            self.start = (x, y)
            self.status.set(
                f"Start set to ({x:.2f}, {y:.2f}). Now right-click to set Goal."
            )
        elif event.button == 3:  # right -> goal
            self.goal = (x, y)
            self.status.set(
                f"Goal set to ({x:.2f}, {y:.2f}). Press 'Append to Dataset'."
            )
        # Recompute preview path if both points are available
        if self.start and self.goal:
            self.compute_preview_path()
        self.draw_scene()

    def compute_preview_path(self):
        if self.start is None or self.goal is None:
            self.preview_path = None
            return
        if self.occ is None:
            self.preview_path = None
            return
        self.preview_path = astar_path(self.occ, self.start, self.goal, self.grid_res)

    def append_dataset(self):
        if self.start is None or self.goal is None:
            messagebox.showwarning(
                "Missing points",
                "Please set both Start (left-click) and Goal (right-click).",
            )
            return
        # Build generator with chosen params; note: same seed+scenario_id used for determinism
        out_dir = self.out_var.get()
        scen = ScenarioSpec(
            width=self.spec.width,
            height=self.spec.height,
            n_rects=self.spec.n_rects,
            seed=self.spec.seed,
            layout=self.spec.layout,
            apt_rows=self.spec.apt_rows,
            apt_cols=self.spec.apt_cols,
            apt_door_prob=self.spec.apt_door_prob,
        )
        lidar = LidarSpec()  # defaults
        cfg = AutoGenConfig(
            grid_res=0.1,
            seq_steps=self.seq_steps_var.get(),
            scenarios=1,
            scans_per_scenario=0,
            seq_per_scenario=0,
            seed=self.spec.seed,
        )
        gen = DatasetGenerator(out_dir, lidar, scen, cfg)
        sid = int(self.scen_id_var.get())
        # Append manual sequence + single capture
        out = gen.append_manual_sequence(
            sid, self.start, self.goal, seq_steps=self.seq_steps_var.get()
        )
        if out is None:
            self.status.set(
                "No path found from Start to Goal. Try again or change seed/scene."
            )
            messagebox.showerror(
                "No path",
                "Planner couldn't find a path. Choose different points or regenerate.",
            )
        else:
            self.status.set(
                f"Appended manual sequence and single_capture to scenario_{sid:04d}."
            )
            messagebox.showinfo(
                "Saved", f"Appended:\n{out}\n...and updated single_capture.csv"
            )

def main():
    app = ManualGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
