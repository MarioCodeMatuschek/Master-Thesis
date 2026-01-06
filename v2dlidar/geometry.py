
from dataclasses import dataclass
from typing import Tuple, Optional, List
import math

@dataclass
class Segment:
    x1: float; y1: float; x2: float; y2: float

    def as_tuple(self):
        return (self.x1, self.y1, self.x2, self.y2)

def ray_segment_intersection(px: float, py: float, dx: float, dy: float, seg: Segment) -> Optional[Tuple[float, float, float]]:
    """
    Ray: p + t*(d), t >= 0; Segment: s + u*(e), u in [0,1].
    Returns (t, ix, iy) for the closest intersection if exists, else None.
    """
    sx, sy, ex, ey = seg.x1, seg.y1, seg.x2, seg.y2
    rx, ry = dx, dy
    sxr, syr = ex - sx, ey - sy
    denom = (-rx * syr + ry * sxr)
    if abs(denom) < 1e-12:
        return None  # parallel or colinear; we ignore colinear-on-ray for simplicity
    t = (-(px - sx) * syr + (py - sy) * sxr) / denom
    u = (-rx * (py - sy) + ry * (px - sx)) / denom
    if t >= 0.0 and 0.0 <= u <= 1.0:
        ix = px + t * rx
        iy = py + t * ry
        return (t, ix, iy)
    return None

def bresenham_line(x0, y0, x1, y1):
    """Simple integer grid line for occupancy rasterization.
Returns list of (x,y) integer grid cells.
"""
    x0 = int(x0); y0 = int(y0); x1 = int(x1); y1 = int(y1)
    dx = abs(x1 - x0); sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0); sy = 1 if y0 < y1 else -1
    err = dx + dy
    cells = []
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy; x0 += sx
        if e2 <= dx:
            err += dx; y0 += sy
    return cells
