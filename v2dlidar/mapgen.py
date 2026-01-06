
from dataclasses import dataclass
from typing import List, Tuple
import random, math
import numpy as np
from .geometry import Segment

@dataclass
class ScenarioSpec:
    width: float = 30.0
    height: float = 30.0
    n_rects: int = 20
    rect_min: Tuple[float, float] = (2.0, 2.0)
    rect_max: Tuple[float, float] = (8.0, 6.0)
    wall_thickness: float = 0.1
    seed: int = 0
    outline_res: float = 0.1    # resolution used to build union outline
    layout: str = "union"   # "union" and "appartment" will be the sample layout choices (random rectangle overlap vs. clearn floorplan like)
    apt_rows: int = 2   # room grid rows
    apt_cols: int = 3   # room grid cols
    apt_door_prob: float = 0.8  #chance of doorway between adjacent rooms

def _rect_edges(cx, cy, w, h, yaw):
    c = math.cos(yaw); s = math.sin(yaw)
    # rectangle corners centered at (0,0)
    pts = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
    # rotate+translate
    world = [(cx + c*x - s*y, cy + s*x + c*y) for x,y in pts]
    edges = []
    for i in range(4):
        x1,y1 = world[i]
        x2,y2 = world[(i+1)%4]
        edges.append(Segment(x1,y1,x2,y2))
    return edges

def _paint_rect(grid: np.ndarray, cx: float, cy: float, w: float, h: float, res: float):
    H, W = grid.shape
    x0 = max(0, int(math.floor((cx - w/2) / res)))
    x1 = min(W - 1, int(math.floor((cx + w/2) / res)))
    y0 = max(0, int(math.floor((cy - h/2) / res)))
    y1 = min(H - 1, int(math.floor((cy + h/2) / res)))
    if x0 <= x1 and y0 <= y1:
        grid[y0 : y1 + 1, x0 : x1 + 1] = 1


def _extract_outline_segments(grid: np.ndarray, res: float) -> List[Segment]:
    H, W = grid.shape
    segments: List[Segment] = []
    # Horizontal edges (between cell (y-1, x) and (y, x)) for y in 0..H
    for y in range(H + 1):
        x = 0
        while x < W:
            # Determine if edge between (y-1,x) and (y,x) is boundary
            def occ(yy, xx):
                return 1 if (0 <= yy < H and 0 <= xx < W and grid[yy, xx]) else 0

            left = x
            # boundary present when occupancy differs across the edge
            is_boundary = occ(y - 1, x) != occ(y, x)
            if not is_boundary:
                x += 1
                continue
            # extend run
            x += 1
            while x < W and (occ(y - 1, x) != occ(y, x)):
                x += 1
            right = x
            # Convert to world coords; y-edge at y, spans [left, right)
            y_world = y * res
            x1w = left * res
            x2w = right * res
            segments.append(Segment(x1w, y_world, x2w, y_world))
    # Vertical edges (between cell (y,x-1) and (y,x)) for x in 0..W
    for x in range(W + 1):
        y = 0
        while y < H:
            def occ2(yy, xx):
                return 1 if (0 <= yy < H and 0 <= xx < W and grid[yy, xx]) else 0

            top = y
            is_boundary = occ2(y, x - 1) != occ2(y, x)
            if not is_boundary:
                y += 1
                continue
            y += 1
            while y < H and (occ2(y, x - 1) != occ2(y, x)):
                y += 1
            bottom = y
            x_world = x * res
            y1w = top * res
            y2w = bottom * res
            segments.append(Segment(x_world, y1w, x_world, y2w))
    return segments


def _generate_union_scenario(spec: ScenarioSpec):
    """
    Build a connected shape as the union of randomly sized axis-aligned rectangles.

    Returns
    -------
    segments : List[Segment]
        Outline (border) segments of the union plus an outer rectangular boundary
        around the map.
    spec : ScenarioSpec
        The (possibly updated) scenario specification used.
    interior : np.ndarray
        2D uint8 grid where 1 indicates cells inside the union of rectangles,
        0 indicates outside.
    res : float
        Resolution (meters per cell) of the interior grid.
    """
    random.seed(spec.seed)
    rng = random.Random(spec.seed)

    # Grid for union outline construction
    res = max(1e-3, float(spec.outline_res))
    W = int(math.ceil(spec.width / res))
    H = int(math.ceil(spec.height / res))
    union = np.zeros((H, W), dtype=np.uint8)

    rects: List[Tuple[float, float, float, float]] = []  # (cx, cy, w, h)

    # First rectangle anywhere within bounds
    w0 = rng.uniform(spec.rect_min[0], spec.rect_max[0])
    h0 = rng.uniform(spec.rect_min[1], spec.rect_max[1])
    cx0 = rng.uniform(w0 * 0.5, spec.width - w0 * 0.5)
    cy0 = rng.uniform(h0 * 0.5, spec.height - h0 * 0.5)
    rects.append((cx0, cy0, w0, h0))
    _paint_rect(union, cx0, cy0, w0, h0, res)

    # Subsequent rectangles must overlap (at least partially) with existing union
    for _ in range(spec.n_rects - 1):
        pw, ph = rng.uniform(spec.rect_min[0], spec.rect_max[0]), rng.uniform(
            spec.rect_min[1], spec.rect_max[1]
        )
        # pick an existing rectangle to overlap with
        pcx, pcy, pw0, ph0 = rects[rng.randrange(len(rects))]
        # sample dx, dy such that |dx| < (pw+pw0)/2 and |dy| < (ph+ph0)/2 (ensures overlap)
        dx_lim = 0.45 * (pw + pw0)  # 0.45 to bias to more-than-touching overlap
        dy_lim = 0.45 * (ph + ph0)
        tries = 0
        while True:
            dx = rng.uniform(-dx_lim, dx_lim)
            dy = rng.uniform(-dy_lim, dy_lim)
            cx = pcx + dx
            cy = pcy + dy
            # keep fully within map bounds; if not, resample a few times
            if (
                (pw * 0.5) <= cx <= (spec.width - pw * 0.5)
                and (ph * 0.5) <= cy <= (spec.height - ph * 0.5)
            ):
                break
            tries += 1
            if tries > 50:
                # fallback: clamp into bounds
                cx = min(max(cx, pw * 0.5), spec.width - pw * 0.5)
                cy = min(max(cy, ph * 0.5), spec.height - ph * 0.5)
                break
        rects.append((cx, cy, pw, ph))
        _paint_rect(union, cx, cy, pw, ph, res)

    # Build segments: outer boundary + outline of union
    segments: List[Segment] = []
    segments += _rect_edges(spec.width / 2, spec.height / 2, spec.width, spec.height, 0.0)
    segments += _extract_outline_segments(union, res)
    return segments, spec, union, res


def _generate_apartment_scenario(spec: ScenarioSpec):
    """
    Build an apartment-like floorplan inspired by apartment_floorplan.svg:

    - Outer rectangular boundary
    - One main vertical spine wall with a doorway near the middle
    - Additional vertical walls on left/right to form rooms
    - Two horizontal walls at different heights with door gaps
      (forming a corridor-like central region)
    - Doors are represented as gaps in otherwise continuous walls

    Returns (segments, spec, interior, res) just like _generate_union_scenario.
    """
    random.seed(spec.seed)
    rng = random.Random(spec.seed)

    res = max(1e-3, float(spec.outline_res))
    W = int(math.ceil(spec.width / res))
    H = int(math.ceil(spec.height / res))
    # Entire map interior is potentially free; walls will be added via segments
    interior = np.ones((H, W), dtype=np.uint8)

    segments: List[Segment] = []
    # Outer boundary walls
    segments += _rect_edges(
        spec.width / 2.0, spec.height / 2.0, spec.width, spec.height, 0.0
    )

    # Small helper for random jitter
    def jitter(base: float, span: float) -> float:
        return base + rng.uniform(-span, span)

    # --- Key structural lines (normalized from SVG, then jittered) ---

    # Main vertical spine ~ center
    x_main = jitter(0.5 * spec.width, 0.05 * spec.width)

    # Optional left/right vertical partitions (inspired by x=250, 550 in SVG)
    # Use apt_cols to decide how much structure we add
    cols = max(1, int(getattr(spec, "apt_cols", 3)))
    x_left = jitter(0.3 * spec.width, 0.05 * spec.width) if cols >= 2 else None
    x_right = jitter(0.7 * spec.width, 0.05 * spec.width) if cols >= 3 else None

    # Two horizontal “levels” (like y=250, y=350 in SVG)
    y_upper = jitter(0.4 * spec.height, 0.05 * spec.height)
    y_lower = jitter(0.6 * spec.height, 0.05 * spec.height)

    # Clamp everything inside the outer rectangle
    x_main = min(max(x_main, 0.2 * spec.width), 0.8 * spec.width)
    if x_left is not None:
        x_left = min(max(x_left, 0.1 * spec.width), x_main - 0.1 * spec.width)
    if x_right is not None:
        x_right = min(
            max(x_right, x_main + 0.1 * spec.width), 0.9 * spec.width
        )
    y_upper = min(max(y_upper, 0.2 * spec.height), 0.5 * spec.height)
    y_lower = min(max(y_lower, 0.5 * spec.height), 0.8 * spec.height)

    # --- Helpers to add walls with door gaps ---

    def add_vertical_wall(x: float, y0: float, y1: float,
                          door_center: float = None,
                          door_height: float = None):
        """Add a vertical wall at x from y0 to y1 with optional door gap."""
        eps = 1e-3
        if door_center is None or door_height is None:
            segments.append(Segment(x, y0, x, y1))
            return
        span = y1 - y0
        dh = min(door_height, 0.8 * span)
        dc = min(max(door_center, y0 + 0.1 * span), y1 - 0.1 * span)
        d0 = max(y0, dc - 0.5 * dh)
        d1 = min(y1, dc + 0.5 * dh)
        if d0 - y0 > eps:
            segments.append(Segment(x, y0, x, d0))
        if y1 - d1 > eps:
            segments.append(Segment(x, d1, x, y1))

    def add_horizontal_wall(y: float, x0: float, x1: float,
                            door_center: float = None,
                            door_width: float = None):
        """Add a horizontal wall at y from x0 to x1 with optional door gap."""
        eps = 1e-3
        if door_center is None or door_width is None:
            segments.append(Segment(x0, y, x1, y))
            return
        span = x1 - x0
        dw = min(door_width, 0.8 * span)
        dc = min(max(door_center, x0 + 0.1 * span), x1 - 0.1 * span)
        d0 = max(x0, dc - 0.5 * dw)
        d1 = min(x1, dc + 0.5 * dw)
        if d0 - x0 > eps:
            segments.append(Segment(x0, y, d0, y))
        if x1 - d1 > eps:
            segments.append(Segment(d1, y, x1, y))

    # --- Vertical walls (like x=400, 250, 550 in SVG) ---

    # Main spine: full height with a doorway roughly in the middle
    add_vertical_wall(
        x_main,
        0.0,
        spec.height,
        door_center=jitter(0.5 * spec.height, 0.05 * spec.height),
        door_height=0.12 * spec.height,
    )

    # Left partition: below upper level, with door near between upper/lower
    if x_left is not None:
        add_vertical_wall(
            x_left,
            y_upper,
            spec.height,
            door_center=jitter(
                0.5 * (y_upper + y_lower), 0.05 * spec.height
            ),
            door_height=(y_lower - y_upper) * 0.6,
        )

    # Right partition: below lower level, sometimes with a door, sometimes solid
    if x_right is not None:
        if rng.random() < 0.7:
            add_vertical_wall(
                x_right,
                y_lower,
                spec.height,
                door_center=jitter(
                    0.5 * (y_lower + spec.height), 0.05 * spec.height
                ),
                door_height=(spec.height - y_lower) * 0.4,
            )
        else:
            add_vertical_wall(x_right, y_lower, spec.height)

    # --- Horizontal walls (like y=250 and y=350 in SVG) ---

    # Upper horizontal: from left boundary to main spine, with door on the left side
    door_w = 0.12 * spec.width
    add_horizontal_wall(
        y_upper,
        0.0,
        x_main,
        door_center=jitter(0.35 * spec.width, 0.05 * spec.width),
        door_width=door_w,
    )

    # Lower horizontal: from main spine to right boundary, with door on the right side
    add_horizontal_wall(
        y_lower,
        x_main,
        spec.width,
        door_center=jitter(0.75 * spec.width, 0.05 * spec.width),
        door_width=door_w,
    )

    return segments, spec, interior, res


def generate_scenario(spec: ScenarioSpec):
    """
    Build an environment layout and return its wall segments and interior mask.

    The layout is controlled by `spec.layout`:
    - "union": random union of overlapping rectangles (original behaviour)
    - "apartment": grid of rooms with walls and doorway openings (apartment floorplan)
    """
    layout = getattr(spec, "layout", "union")
    if layout == "apartment":
        return _generate_apartment_scenario(spec)
    else:
        return _generate_union_scenario(spec)