#!/usr/bin/env python3
"""
Standalone procedural apartment floorplan generator with dedicated UI.
Uses recursive space-splitting and places doors between adjacent rooms.
Visualization is done with Tkinter Canvas only (no matplotlib).
Does not depend on or modify any other project code.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import random


# --- Core generation logic (unchanged) ---

class Room:
    def __init__(self, x, y, width, height, name="Room"):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.name = name


def find_shared_wall(r1, r2):
    """
    Detects if two rooms share a wall and returns the midpoint for a door.
    """
    tol = 0.1

    # Vertical shared wall (r1 left of r2 or vice versa)
    if abs((r1.x + r1.width) - r2.x) < tol or abs((r2.x + r2.width) - r1.x) < tol:
        overlap_y_start = max(r1.y, r2.y)
        overlap_y_end = min(r1.y + r1.height, r2.y + r2.height)
        if overlap_y_start < overlap_y_end:
            mid_y = (overlap_y_start + overlap_y_end) / 2
            door_x = r2.x if abs((r1.x + r1.width) - r2.x) < tol else r1.x
            return (door_x, mid_y, "vertical")

    # Horizontal shared wall (r1 below r2 or vice versa)
    if abs((r1.y + r1.height) - r2.y) < tol or abs((r2.y + r2.height) - r1.y) < tol:
        overlap_x_start = max(r1.x, r2.x)
        overlap_x_end = min(r1.x + r1.width, r2.x + r2.width)
        if overlap_x_start < overlap_x_end:
            mid_x = (overlap_x_start + overlap_x_end) / 2
            door_y = r2.y if abs((r1.y + r1.height) - r2.y) < tol else r1.y
            return (mid_x, door_y, "horizontal")

    return None


def split_space(room, min_size=2.5):
    """
    Splits a rectangle into two smaller ones based on the longer axis.
    """
    split_vertically = room.width > room.height

    if split_vertically and room.width > min_size * 2:
        split_point = random.uniform(min_size, room.width - min_size)
        r1 = Room(room.x, room.y, split_point, room.height)
        r2 = Room(room.x + split_point, room.y, room.width - split_point, room.height)
        return [r1, r2]

    elif not split_vertically and room.height > min_size * 2:
        split_point = random.uniform(min_size, room.height - min_size)
        r1 = Room(room.x, room.y, room.width, split_point)
        r2 = Room(room.x, room.y + split_point, room.width, room.height - split_point)
        return [r1, r2]

    return [room]


def generate_apartment(width, height, iterations=4):
    """
    Main generator loop. Returns a list of rooms and a list of door coordinates.
    """
    rooms = [Room(0, 0, width, height, "Apartment")]
    doors = []

    for _ in range(iterations):
        new_rooms = []
        for r in rooms:
            res = split_space(r)
            if len(res) > 1:
                door = find_shared_wall(res[0], res[1])
                if door:
                    doors.append(door)
                new_rooms.extend(res)
            else:
                new_rooms.append(r)
        rooms = new_rooms

    return rooms, doors


# --- Tkinter Canvas drawing (world coords: x right, y up; canvas: x right, y down) ---

def _world_to_canvas(x, y, world_w, world_h, pad, draw_w, draw_h):
    """Map world (x, y) with y-up to canvas pixels. Fit (world_w, world_h) in (draw_w, draw_h)."""
    scale = min((draw_w - 2 * pad) / world_w, (draw_h - 2 * pad) / world_h)
    cx = pad + scale * x
    cy = pad + scale * (world_h - y)
    return cx, cy, scale


def draw_floorplan_on_canvas(canvas, rooms, doors, draw_width, draw_height):
    """
    Renders rooms and door swings on a Tkinter Canvas.
    Clears the canvas and draws everything in world coordinates (y-up), scaled to fit.
    """
    canvas.delete("all")

    if not rooms:
        canvas.create_text(
            draw_width // 2, draw_height // 2,
            text="Click Generate to create a floorplan",
            fill="#888"
        )
        return

    world_w = max(r.x + r.width for r in rooms)
    world_h = max(r.y + r.height for r in rooms)
    pad = 20
    _, _, scale = _world_to_canvas(0, 0, world_w, world_h, pad, draw_width, draw_height)
    door_width_world = 0.8

    def to_c(x, y):
        cx, cy, _ = _world_to_canvas(x, y, world_w, world_h, pad, draw_width, draw_height)
        return cx, cy

    # 1. Room rectangles (draw in order so borders overlap cleanly)
    for i, r in enumerate(rooms):
        x1, y1 = to_c(r.x, r.y)
        x2, y2 = to_c(r.x + r.width, r.y + r.height)
        # Canvas rect: (x1, y1) top-left, (x2, y2) bottom-right (y1 < y2 in canvas)
        canvas.create_rectangle(
            x1, y1, x2, y2,
            outline="#333333", fill="#f9f9f9", width=3
        )
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        canvas.create_text(cx, cy, text=f"Room {i + 1}", fill="#555555", font=("", 10, "bold"))

    # 2. Doors: white gap + blue swing arc
    for dx, dy, orientation in doors:
        if orientation == "vertical":
            half = door_width_world / 2
            gx1, gy1 = to_c(dx, dy - half)
            gx2, gy2 = to_c(dx, dy + half)
            canvas.create_line(gx1, gy1, gx2, gy2, fill="white", width=4)
            # Arc: door swing 0° to 90° from (dx, dy - half); in canvas y-down, so angles change
            ax1, ay1 = to_c(dx, dy - half)
            ax2, ay2 = to_c(dx, dy + half)
            # create_arc(x1,y1,x2,y2, start=angle, extent=angle) - angles in degrees, 0=3 o'clock
            # We want arc from (dx, dy-half) upward then right: in canvas that's from 12 o'clock going to 3
            span = 2 * half * scale
            canvas.create_arc(
                ax1 - span, ay1 - span, ax1 + span, ay1 + span,
                start=0, extent=90, outline="blue", width=2, style=tk.ARC
            )
        else:
            half = door_width_world / 2
            gx1, gy1 = to_c(dx - half, dy)
            gx2, gy2 = to_c(dx + half, dy)
            canvas.create_line(gx1, gy1, gx2, gy2, fill="white", width=4)
            ax1, ay1 = to_c(dx - half, dy)
            span = 2 * half * scale
            canvas.create_arc(
                ax1 - span, ay1 - span, ax1 + span, ay1 + span,
                start=270, extent=90, outline="blue", width=2, style=tk.ARC
            )

    # Title
    canvas.create_text(
        draw_width // 2, 12,
        text="Procedural Apartment Floorplan",
        fill="#333", font=("", 12, "bold")
    )


# --- Standalone UI ---

class ApartmentScenarioGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Apartment Scenario Generator")
        self.geometry("900x700")

        # Left panel: parameters
        params = ttk.Frame(self, padding=10)
        params.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(params, text="Width").pack(anchor="w")
        self.width_var = tk.DoubleVar(value=18.0)
        ttk.Entry(params, textvariable=self.width_var, width=12).pack(anchor="w")

        ttk.Label(params, text="Height").pack(anchor="w")
        self.height_var = tk.DoubleVar(value=12.0)
        ttk.Entry(params, textvariable=self.height_var, width=12).pack(anchor="w")

        ttk.Label(params, text="Iterations (splits)").pack(anchor="w")
        self.iterations_var = tk.IntVar(value=4)
        ttk.Spinbox(
            params, textvariable=self.iterations_var, from_=1, to=8, width=10
        ).pack(anchor="w")

        ttk.Label(params, text="Seed (0 = random)").pack(anchor="w")
        self.seed_var = tk.IntVar(value=0)
        ttk.Entry(params, textvariable=self.seed_var, width=12).pack(anchor="w")

        ttk.Separator(params, orient="horizontal").pack(fill="x", pady=8)

        ttk.Button(params, text="Generate", command=self.on_generate).pack(
            anchor="w", pady=4
        )
        ttk.Button(params, text="Save as PNG", command=self.on_save_png).pack(
            anchor="w", pady=2
        )
        ttk.Button(params, text="Save as PostScript", command=self.on_save_ps).pack(
            anchor="w", pady=2
        )

        # Right: Tkinter Canvas
        canvas_frame = ttk.Frame(self, padding=10)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.draw_width = 700
        self.draw_height = 600
        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.draw_width,
            height=self.draw_height,
            bg="white",
            highlightthickness=1,
            highlightbackground="#ccc"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._rooms = []
        self._doors = []

        # Initial placeholder
        draw_floorplan_on_canvas(
            self.canvas, self._rooms, self._doors,
            self.draw_width, self.draw_height
        )

    def on_generate(self):
        try:
            w = self.width_var.get()
            h = self.height_var.get()
            it = self.iterations_var.get()
            seed = self.seed_var.get()
            if w <= 0 or h <= 0:
                messagebox.showerror("Error", "Width and height must be positive.")
                return
            if it < 1 or it > 10:
                messagebox.showerror("Error", "Iterations should be between 1 and 10.")
                return
            if seed != 0:
                random.seed(seed)
            self._rooms, self._doors = generate_apartment(w, h, iterations=it)
            self.canvas.update_idletasks()
            draw_floorplan_on_canvas(
                self.canvas, self._rooms, self._doors,
                self.draw_width, self.draw_height
            )
        except tk.TclError as e:
            messagebox.showerror("Invalid input", str(e))

    def on_save_png(self):
        if not self._rooms:
            messagebox.showinfo("Info", "Generate a floorplan first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            messagebox.showinfo(
                "PNG export",
                "PNG save requires Pillow (pip install Pillow). Use Save as PostScript for built-in export."
            )
            return
        world_w = max(r.x + r.width for r in self._rooms)
        world_h = max(r.y + r.height for r in self._rooms)
        pad = 40
        scale = min((800 - 2 * pad) / world_w, (600 - 2 * pad) / world_h)
        img_w = int(2 * pad + world_w * scale)
        img_h = int(2 * pad + world_h * scale)
        img = Image.new("RGB", (img_w, img_h), "white")
        draw = ImageDraw.Draw(img)

        def to_px(x, y):
            px = pad + scale * x
            py = pad + scale * (world_h - y)
            return px, py

        for i, r in enumerate(self._rooms):
            x1, y1 = to_px(r.x, r.y)
            x2, y2 = to_px(r.x + r.width, r.y + r.height)
            draw.rectangle([x1, y1, x2, y2], outline="#333333", fill="#f9f9f9", width=3)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            draw.text((cx, cy), f"Room {i + 1}", fill="#555555")
        door_width = 0.8
        for dx, dy, orientation in self._doors:
            half = door_width / 2
            if orientation == "vertical":
                gx, gy1 = to_px(dx, dy - half)
                _, gy2 = to_px(dx, dy + half)
                draw.line([(gx, gy1), (gx, gy2)], fill="white", width=4)
            else:
                gx1, gy = to_px(dx - half, dy)
                gx2, _ = to_px(dx + half, dy)
                draw.line([(gx1, gy), (gx2, gy)], fill="white", width=4)
        img.save(path)
        messagebox.showinfo("Saved", f"Saved to {path}")

    def on_save_ps(self):
        if not self._rooms:
            messagebox.showinfo("Info", "Generate a floorplan first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".ps",
            filetypes=[("PostScript", "*.ps"), ("All files", "*.*")]
        )
        if path:
            self.canvas.postscript(file=path, colormode="color")
            messagebox.showinfo("Saved", f"Saved to {path}")


def main():
    app = ApartmentScenarioGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
