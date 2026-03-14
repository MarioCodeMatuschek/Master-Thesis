import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import numpy as np, math, os, random
from .mapgen import ScenarioSpec, generate_scenario, get_apartment_layout_data
from .planner import occupancy_from_segments, astar_path
from .dataset import DatasetGenerator, AutoGenConfig
from .lidar import LidarSpec

class ManualGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Virtual 2D LiDAR - Manual Scenario GUI")
        params = ttk.Frame(self)
        params.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        # 1. General Settings
        gen_frame = ttk.LabelFrame(params, text="General Settings", padding=4)
        gen_frame.pack(fill="x", pady=(0, 6))
        ttk.Label(gen_frame, text="Output dir").pack(anchor="w")
        self.out_var = tk.StringVar(value="./dataset")
        ttk.Entry(gen_frame, textvariable=self.out_var, width=28).pack(anchor="w")
        ttk.Label(gen_frame, text="Scenario ID").pack(anchor="w")
        self.scen_id_var = tk.IntVar(value=0)
        ttk.Entry(gen_frame, textvariable=self.scen_id_var, width=10).pack(anchor="w")
        ttk.Label(gen_frame, text="Seed").pack(anchor="w")
        self.seed_var = tk.IntVar(value=0)
        ttk.Entry(gen_frame, textvariable=self.seed_var, width=10).pack(anchor="w")
        ttk.Label(gen_frame, text="Layout").pack(anchor="w")
        self.layout_var = tk.StringVar(value="union")
        ttk.Combobox(
            gen_frame,
            textvariable=self.layout_var,
            values=["union", "apartment"],
            state="readonly",
            width=12,
        ).pack(anchor="w")
        ttk.Label(gen_frame, text="Width, Height").pack(anchor="w")
        self.width_var = tk.DoubleVar(value=30.0)
        self.height_var = tk.DoubleVar(value=30.0)
        row_wh = ttk.Frame(gen_frame)
        row_wh.pack(anchor="w")
        ttk.Entry(row_wh, textvariable=self.width_var, width=10).pack(side=tk.LEFT)
        ttk.Entry(row_wh, textvariable=self.height_var, width=10).pack(side=tk.LEFT, padx=4)

        # 2. Union Layout Scenarios
        union_frame = ttk.LabelFrame(params, text="Union Layout Scenarios", padding=4)
        union_frame.pack(fill="x", pady=(0, 6))
        ttk.Label(union_frame, text="# Rectangles").pack(anchor="w")
        self.nrect_var = tk.IntVar(value=20)
        ttk.Entry(union_frame, textvariable=self.nrect_var, width=10).pack(anchor="w")

        # 3. Apartment Layout Scenarios
        apt_frame = ttk.LabelFrame(params, text="Apartment Layout Scenarios", padding=4)
        apt_frame.pack(fill="x", pady=(0, 6))
        ttk.Label(apt_frame, text="Apartment iterations").pack(anchor="w")
        self.apt_iterations_var = tk.IntVar(value=4)
        ttk.Spinbox(
            apt_frame, textvariable=self.apt_iterations_var, from_=1, to=8, width=10
        ).pack(anchor="w")

        # Generate/Refresh
        ttk.Button(params, text="Generate/Refresh Scenario", command=self.refresh).pack(
            anchor="w", pady=6
        )
        # Append to Dataset
        ttk.Button(
            params, text="Append to Dataset (Manual)", command=self.append_dataset
        ).pack(anchor="w", pady=(0, 8))

        # 4. Other Settings
        other_frame = ttk.LabelFrame(params, text="Other Settings", padding=4)
        other_frame.pack(fill="x", pady=(0, 6))
        ttk.Label(other_frame, text="Seq steps").pack(anchor="w")
        self.seq_steps_var = tk.IntVar(value=100)
        ttk.Entry(other_frame, textvariable=self.seq_steps_var, width=10).pack(anchor="w")
        self.preview_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            other_frame,
            text="Preview planned path",
            variable=self.preview_var,
            command=self.draw_scene,
        ).pack(anchor="w")
        ttk.Label(other_frame, text="Complexity").pack(anchor="w")
        self.comp_text = tk.StringVar(value="?")
        ttk.Label(other_frame, textvariable=self.comp_text, justify="left").pack(anchor="w")
        self.status = tk.StringVar(
            value="Set Start (left click), Goal (right-click or Control+click)"
        )
        ttk.Label(other_frame, textvariable=self.status, wraplength=220).pack(
            anchor="w", pady=4
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
            apt_iterations=self.apt_iterations_var.get(),
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
        self.status.set("Pick Start (left click), Goal (right-click or Control+click)")

    def draw_scene(self):
        self.ax.clear()
        if self.spec is not None:
            self.ax.set_xlim(0, self.spec.width)
            self.ax.set_ylim(0, self.spec.height)
        layout = getattr(self.spec, "layout", "union")
        if layout == "apartment":
            layout_data = get_apartment_layout_data(self.spec) if self.spec else None
            if layout_data is not None:
                rooms, doors = layout_data
                door_width = 0.8
                half_door = door_width / 2.0
                for r in rooms:
                    rect = mpatches.Rectangle(
                        (r.x, r.y), r.width, r.height,
                        linewidth=3, edgecolor="#333333", facecolor="#f9f9f9"
                    )
                    self.ax.add_patch(rect)
                for dx, dy, orientation, _ in doors:
                    if orientation == "vertical":
                        self.ax.plot(
                            [dx, dx], [dy - half_door, dy + half_door],
                            color="white", linewidth=4, zorder=3
                        )
                    else:
                        self.ax.plot(
                            [dx - half_door, dx + half_door], [dy, dy],
                            color="white", linewidth=4, zorder=3
                        )
            else:
                for s in self.segments:
                    self.ax.plot([s.x1, s.x2], [s.y1, s.y2], linewidth=1)
        else:
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
        # Primary click = Start; secondary click = Goal.
        # On macOS: right-click may be reported as button 2 or 3; Control+click is secondary.
        is_secondary = (
            event.button == 3
            or event.button == 2
            or (event.button == 1 and getattr(event, "key", None) == "control")
        )
        if event.button == 1 and not (
            getattr(event, "key", None) == "control"
        ):
            self.start = (x, y)
            self.status.set(
                f"Start set to ({x:.2f}, {y:.2f}). Right-click or Control+click to set Goal."
            )
        elif is_secondary:
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
                "Please set both Start (left-click) and Goal (right-click or Control+click).",
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
            apt_iterations=getattr(self.spec, "apt_iterations", 4),
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
