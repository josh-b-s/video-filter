"""Simple per-landmark exponential smoothing across frames, to reduce
jitter in extracted 3D points."""

import config


class LandmarkSmoother:
    def __init__(self, alpha: float = config.smoothing_alpha):
        self._alpha = alpha
        self._smoothed: dict[int, tuple[float, float, float]] = {}

    def smooth(self, idx: int, x: float, y: float, z: float) -> tuple[float, float, float]:
        if self._alpha <= 0 or idx not in self._smoothed:
            self._smoothed[idx] = (x, y, z)
        else:
            px, py, pz = self._smoothed[idx]
            a = self._alpha
            self._smoothed[idx] = (
                px + a * (x - px),
                py + a * (y - py),
                pz + a * (z - pz),
            )
        return self._smoothed[idx]
