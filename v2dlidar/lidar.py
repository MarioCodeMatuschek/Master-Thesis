
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math, random
import numpy as np
from .geometry import ray_segment_intersection, Segment

@dataclass
class LidarSpec:
    fov_deg: float = 360.0
    num_rays: int = 720
    max_range: float = 12.0
    range_noise_std: float = 0.02
    dropout_prob: float = 0.01

class LidarSimulator:
    def __init__(self, spec: LidarSpec, segments: List[Segment]):
        self.spec = spec
        self.segments = segments

    def scan(
        self,
        pose: Tuple[float, float, float],
        rng: random.Random = None,
        noise_free: bool = False,
        target_xy: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, np.ndarray]:
        if rng is None:
            rng = random
        x, y, yaw = pose
        fov = math.radians(self.spec.fov_deg)
        start = yaw - fov/2.0
        thetas = np.linspace(0.0, fov, self.spec.num_rays, endpoint=False) + start
        ranges = np.full(self.spec.num_rays, self.spec.max_range, dtype=np.float32)
        hits = np.full((self.spec.num_rays, 2), np.nan, dtype=np.float32)
        eps = 1e-9

        # When a target is enabled, allow a "virtual point hit" at the exact
        # target center even though the corresponding wall segment is carved out.
        #
        # `target_tol` is fixed per scan (based on ray angular discretization) to keep
        # the number of virtual hits low while still ensuring that a ray aligned
        # with the target can register a hit.
        target_x, target_y = (None, None) if target_xy is None else target_xy
        if target_xy is not None:
            ray_step = fov / float(self.spec.num_rays)
            # Perpendicular miss distance for the closest ray when angular error is
            # at most ray_step/2: miss ~= t * sin(ray_step/2).
            target_tol = self.spec.max_range * math.sin(ray_step / 2.0) + 1e-6
        else:
            target_tol = None

        for i, th in enumerate(thetas):
            dx, dy = math.cos(th), math.sin(th)
            best_t = None; best_ix = None; best_iy = None
            for seg in self.segments:
                res = ray_segment_intersection(x, y, dx, dy, seg)
                if res is None: continue
                t, ix, iy = res
                if t <= 0.0: continue
                if t > self.spec.max_range: continue
                if best_t is None or t < best_t:
                    best_t, best_ix, best_iy = t, ix, iy

            # Decide whether the virtual target point overrides the wall hit.
            target_wins = False
            if target_xy is not None:
                vx, vy = target_x - x, target_y - y
                # Ray direction (dx,dy) is unit length, so t_target is the projected distance.
                t_target = vx * dx + vy * dy
                if 0.0 - eps <= t_target <= self.spec.max_range + eps:
                    # Perpendicular distance from target to the ray line at parameter t_target.
                    vv = vx * vx + vy * vy
                    perp_sq = vv - t_target * t_target
                    if perp_sq < 0.0:
                        perp_sq = 0.0
                    dist_perp = math.sqrt(perp_sq)
                    if dist_perp <= target_tol:
                        # "No wall in the way": allow the target only if it is not behind
                        # the closest wall intersection along this ray.
                        if best_t is None or best_t >= t_target - eps:
                            target_wins = True

            if target_wins:
                ranges[i] = float(t_target)
                hits[i, 0] = float(target_x)
                hits[i, 1] = float(target_y)
            elif best_t is not None:
                ranges[i] = best_t
                hits[i, 0] = best_ix
                hits[i, 1] = best_iy
        noise_free_ranges = ranges.copy()
        # add noise and dropout
        mask_valid = np.isfinite(hits[:,0])
        if not noise_free:
            if self.spec.range_noise_std > 0:
                ranges[mask_valid] += rng.normalvariate(0, self.spec.range_noise_std)
                ranges = np.clip(ranges, 0.0, self.spec.max_range)
            if self.spec.dropout_prob > 0:
                drops = (np.random.rand(self.spec.num_rays) < self.spec.dropout_prob)
                # Physical-style dropout: no return -> reported range becomes 0.
                ranges[drops] = 0.0
                hits[drops] = np.nan
        theta_deg = (np.degrees(thetas) + 360.0) % 360.0
        return {
            "theta_deg": theta_deg.astype(np.float32),
            "ranges": ranges.astype(np.float32),
            "hits": hits.astype(np.float32),
            "noise_free_ranges": noise_free_ranges.astype(np.float32),
        }
