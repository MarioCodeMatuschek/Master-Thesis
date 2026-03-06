
from dataclasses import dataclass
from typing import List, Tuple, Optional
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
    apt_door_prob: float = 0.8  # chance of doorway between adjacent rooms
    apt_min_door_width: float = 0.6  # min door gap (m); should exceed 2× occupancy wall thickness (~0.15 m) so doors stay traversable after inflation
    apt_verify_connectivity: bool = False  # if True, check that free space is one connected component after generation
    apt_iterations: int = 4  # BSP split iterations for apartment layout (number of recursive splits)

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


def _sample_partition(total: float, n: int, rng: random.Random, min_val: float) -> List[float]:
    """Sample n positive values that sum to total, each >= min_val."""
    if n <= 0:
        return []
    if n == 1:
        return [total]
    raw = [rng.uniform(0.1, 1.0) for _ in range(n)]
    s = sum(raw)
    vals = [total * x / s for x in raw]
    for i in range(n):
        if vals[i] < min_val:
            vals[i] = min_val
    s = sum(vals)
    if s > 0:
        vals = [v * total / s for v in vals]
    return vals


# --- BSP apartment layout (from apartment_scenario_generator.py) ---

class _AptRoom:
    """Axis-aligned room rectangle for BSP apartment generation."""
    __slots__ = ("x", "y", "width", "height")

    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


def _apt_find_shared_wall(r1: _AptRoom, r2: _AptRoom):
    """
    If two rooms share a wall, returns (door_x_or_mid_x, door_y_or_mid_y, "vertical"|"horizontal").
    Otherwise returns None.
    """
    tol = 0.1
    # Vertical shared wall
    if abs((r1.x + r1.width) - r2.x) < tol or abs((r2.x + r2.width) - r1.x) < tol:
        overlap_y_start = max(r1.y, r2.y)
        overlap_y_end = min(r1.y + r1.height, r2.y + r2.height)
        if overlap_y_start < overlap_y_end:
            mid_y = (overlap_y_start + overlap_y_end) / 2
            door_x = r2.x if abs((r1.x + r1.width) - r2.x) < tol else r1.x
            return (door_x, mid_y, "vertical")
    # Horizontal shared wall
    if abs((r1.y + r1.height) - r2.y) < tol or abs((r2.y + r2.height) - r1.y) < tol:
        overlap_x_start = max(r1.x, r2.x)
        overlap_x_end = min(r1.x + r1.width, r2.x + r2.width)
        if overlap_x_start < overlap_x_end:
            mid_x = (overlap_x_start + overlap_x_end) / 2
            door_y = r2.y if abs((r1.y + r1.height) - r2.y) < tol else r1.y
            return (mid_x, door_y, "horizontal")
    return None


def _apt_split_space(room: _AptRoom, rng: random.Random, min_size: float = 2.5) -> List[_AptRoom]:
    """Split one room along the longer axis; returns [room] or [r1, r2]."""
    split_vertically = room.width > room.height
    if split_vertically and room.width > min_size * 2:
        split_point = rng.uniform(min_size, room.width - min_size)
        return [
            _AptRoom(room.x, room.y, split_point, room.height),
            _AptRoom(room.x + split_point, room.y, room.width - split_point, room.height),
        ]
    if not split_vertically and room.height > min_size * 2:
        split_point = rng.uniform(min_size, room.height - min_size)
        return [
            _AptRoom(room.x, room.y, room.width, split_point),
            _AptRoom(room.x, room.y + split_point, room.width, room.height - split_point),
        ]
    return [room]


def _generate_apartment_bsp(
    width: float, height: float, iterations: int, rng: random.Random
) -> Tuple[List[_AptRoom], List[Tuple[float, float, str]]]:
    """
    BSP apartment generator. Returns (rooms, doors) where doors are
    (x_or_mid_x, y_or_mid_y, "vertical"|"horizontal").
    """
    rooms: List[_AptRoom] = [_AptRoom(0, 0, width, height)]
    doors: List[Tuple[float, float, str]] = []
    for _ in range(iterations):
        new_rooms: List[_AptRoom] = []
        for r in rooms:
            res = _apt_split_space(r, rng)
            if len(res) > 1:
                door = _apt_find_shared_wall(res[0], res[1])
                if door is not None:
                    doors.append(door)
                new_rooms.extend(res)
            else:
                new_rooms.append(r)
        rooms = new_rooms
    return rooms, doors


def get_apartment_layout_data(spec: ScenarioSpec) -> Optional[Tuple[List[_AptRoom], List[Tuple[float, float, str]]]]:
    """
    Return (rooms, doors) for apartment layout visualization only.
    Uses the same BSP as _generate_apartment_scenario so drawing matches the generated segments.
    Returns None if spec.layout != "apartment".
    """
    if getattr(spec, "layout", "union") != "apartment":
        return None
    random.seed(spec.seed)
    rng = random.Random(spec.seed)
    iterations = max(1, getattr(spec, "apt_iterations", 4))
    rooms, doors = _generate_apartment_bsp(spec.width, spec.height, iterations, rng)
    return (rooms, doors)


def _apt_wall_extent_for_door(
    rooms: List[_AptRoom], dx: float, dy: float, orientation: str, tol: float = 0.1
) -> Tuple[float, float]:
    """
    Returns (start, end) in world coordinates along the wall that contains the door.
    For vertical wall: start/end are y_min, y_max. For horizontal: x_min, x_max.
    Uses the two rooms that share the wall containing (dx, dy).
    """
    if orientation == "vertical":
        # Wall at x = dx; find the two rooms that share this wall (door at (dx, dy) lies on it)
        left_room = None  # r.x + r.width ~ dx
        right_room = None  # r.x ~ dx
        for r in rooms:
            if abs((r.x + r.width) - dx) < tol and r.y <= dy + tol and dy - tol <= r.y + r.height:
                left_room = r
            if abs(r.x - dx) < tol and r.y <= dy + tol and dy - tol <= r.y + r.height:
                right_room = r
        if left_room is not None and right_room is not None:
            y_min = max(left_room.y, right_room.y)
            y_max = min(left_room.y + left_room.height, right_room.y + right_room.height)
            if y_min < y_max:
                return (y_min, y_max)
        return (dy - 0.5, dy + 0.5)
    else:
        # Horizontal wall at y = dy
        below_room = None
        above_room = None
        for r in rooms:
            if abs((r.y + r.height) - dy) < tol and r.x <= dx + tol and dx - tol <= r.x + r.width:
                below_room = r
            if abs(r.y - dy) < tol and r.x <= dx + tol and dx - tol <= r.x + r.width:
                above_room = r
        if below_room is not None and above_room is not None:
            x_min = max(below_room.x, above_room.x)
            x_max = min(below_room.x + below_room.width, above_room.x + above_room.width)
            if x_min < x_max:
                return (x_min, x_max)
        return (dx - 0.5, dx + 0.5)


def _generate_apartment_scenario(spec: ScenarioSpec):
    """
    Build an apartment-like floorplan using BSP (binary space partitioning):
    recursively split the space along the longer axis and place a door between
    each pair of sibling rooms. Matches the layout produced by apartment_scenario_generator.py.

    Returns (segments, spec, interior, res) with the same structure as _generate_union_scenario.
    """
    random.seed(spec.seed)
    rng = random.Random(spec.seed)

    res = max(1e-3, float(spec.outline_res))
    W = int(math.ceil(spec.width / res))
    H = int(math.ceil(spec.height / res))
    interior = np.ones((H, W), dtype=np.uint8)

    iterations = max(1, getattr(spec, "apt_iterations", 4))
    min_door_width = float(getattr(spec, "apt_min_door_width", 0.6))
    eps = 1e-6

    rooms, doors = _generate_apartment_bsp(spec.width, spec.height, iterations, rng)

    segments: List[Segment] = []
    segments += _rect_edges(
        spec.width / 2.0, spec.height / 2.0, spec.width, spec.height, 0.0
    )

    half_gap = min_door_width / 2.0
    for door in doors:
        dx, dy, orientation = door
        if orientation == "vertical":
            y_min, y_max = _apt_wall_extent_for_door(rooms, dx, dy, "vertical")
            if y_min < dy - half_gap - eps:
                segments.append(Segment(dx, y_min, dx, dy - half_gap))
            if dy + half_gap + eps < y_max:
                segments.append(Segment(dx, dy + half_gap, dx, y_max))
        else:
            x_min, x_max = _apt_wall_extent_for_door(rooms, dx, dy, "horizontal")
            if x_min < dx - half_gap - eps:
                segments.append(Segment(x_min, dy, dx - half_gap, dy))
            if dx + half_gap + eps < x_max:
                segments.append(Segment(dx + half_gap, dy, x_max, dy))

    if getattr(spec, "apt_verify_connectivity", False):
        from .planner import free_space_connected_components
        n_comp, _ = free_space_connected_components(
            segments, spec.width, spec.height, res, interior=interior
        )
        if n_comp != 1:
            import warnings
            warnings.warn(
                f"Apartment scenario free space has {n_comp} connected components (expected 1). Layout may have enclosed rooms."
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
