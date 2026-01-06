
from dataclasses import dataclass
from typing import List, Tuple, Dict
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

    def scan(self, pose: Tuple[float,float,float], rng: random.Random = None, noise_free: bool=False) -> Dict[str, np.ndarray]:
        if rng is None:
            rng = random
        x, y, yaw = pose
        fov = math.radians(self.spec.fov_deg)
        start = yaw - fov/2.0
        thetas = np.linspace(0.0, fov, self.spec.num_rays, endpoint=False) + start
        ranges = np.full(self.spec.num_rays, self.spec.max_range, dtype=np.float32)
        hits = np.full((self.spec.num_rays, 2), np.nan, dtype=np.float32)
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
            if best_t is not None:
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
                ranges[drops] = self.spec.max_range
                hits[drops] = np.nan
        theta_deg = (np.degrees(thetas) + 360.0) % 360.0
        return {
            "theta_deg": theta_deg.astype(np.float32),
            "ranges": ranges.astype(np.float32),
            "hits": hits.astype(np.float32),
            "noise_free_ranges": noise_free_ranges.astype(np.float32),
        }
