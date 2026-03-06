
from typing import List, Tuple, Optional
import numpy as np
from .geometry import Segment, bresenham_line


def free_space_connected_components(
    segments: List[Segment],
    width: float,
    height: float,
    res: float,
    interior: Optional[np.ndarray] = None,
    thickness: float = 0.15,
) -> Tuple[int, int]:
    """
    Build occupancy from segments (and optional interior mask), then count
    connected components of free (traversable) cells using 4-connectivity.

    Returns
    -------
    n_components : int
        Number of connected components of free space.
    n_free : int
        Total number of free cells (occ==0 and inside interior if provided).
    """
    occ, _ = occupancy_from_segments(segments, width, height, res, thickness)
    if interior is not None:
        occ = occ.copy()
        occ[interior == 0] = 1
    H, W = occ.shape
    nbrs4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    visited = np.zeros_like(occ, dtype=np.uint8)
    n_components = 0
    n_free = 0
    for sy in range(H):
        for sx in range(W):
            if occ[sy, sx] or visited[sy, sx]:
                continue
            n_components += 1
            stack = [(sx, sy)]
            visited[sy, sx] = 1
            while stack:
                x, y = stack.pop()
                n_free += 1
                for dx, dy in nbrs4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H and not occ[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = 1
                        stack.append((nx, ny))
    return n_components, n_free


def occupancy_from_segments(segments: List[Segment], width: float, height: float, res: float, thickness: float=0.15):
    W = int(np.ceil(width/res)); H = int(np.ceil(height/res))
    occ = np.zeros((H, W), dtype=np.uint8)
    inflate = int(np.ceil(thickness / res))
    def to_grid(x,y):
        return int(np.clip(x/res, 0, W-1)), int(np.clip(y/res, 0, H-1))
    for s in segments:
        x0, y0 = to_grid(s.x1, s.y1)
        x1, y1 = to_grid(s.x2, s.y2)
        for (ix,iy) in bresenham_line(x0,y0,x1,y1):
            xmn = max(0, ix - inflate); xmx = min(W-1, ix + inflate)
            ymn = max(0, iy - inflate); ymx = min(H-1, iy + inflate)
            occ[ymn:ymx+1, xmn:xmx+1] = 1
    return occ, res

def astar_path(occ: np.ndarray, start_xy: Tuple[float,float], goal_xy: Tuple[float,float], res: float):
    H, W = occ.shape
    def to_grid(p):
        x = int(np.clip(p[0]/res, 0, W-1)); y = int(np.clip(p[1]/res, 0, H-1))
        return x, y
    def to_world(g):
        return (g[0]*res, g[1]*res)
    start = to_grid(start_xy); goal = to_grid(goal_xy)
    if occ[start[1], start[0]] or occ[goal[1], goal[0]]:
        return None
    import heapq, math
    nbrs8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    g = {start:0.0}
    came = {}
    pq = [(0.0, start)]
    def h(a,b): return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5
    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return [to_world(p) for p in path]
        for dx,dy in nbrs8:
            nx, ny = cur[0]+dx, cur[1]+dy
            if nx<0 or ny<0 or nx>=W or ny>=H: continue
            if occ[ny,nx]: continue
            ng = g[cur] + ((dx*dx+dy*dy)**0.5)
            if (nx,ny) not in g or ng < g[(nx,ny)]:
                g[(nx,ny)] = ng
                came[(nx,ny)] = cur
                f = ng + h((nx,ny), goal)
                heapq.heappush(pq, (f, (nx,ny)))
    return None
