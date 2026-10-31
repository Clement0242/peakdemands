"""From positions (or speed) to the kinematics the other modules use.

Acceleration is a vector: a change of speed *or* of direction is an
acceleration (Buchheit & Simpson 2017; Buchheit et al. 2026, "GPS 3.0"). With
x, y positions the complete acceleration is split into

* the **tangential** component a_t = dv/dt, the only one most GPS exports keep;
* the **centripetal** component a_c = v·dθ/dt, from turning, which GPS 2.0
  metrics miss even when speed does not change.

|a|² = a_t² + a_c². Positions are smoothed and differentiated with a
Savitzky-Golay filter (one fit gives position, velocity and acceleration),
short gaps are interpolated and longer ones split the recording.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import savgol_filter


@dataclass
class Kinematics:
    time: np.ndarray        # s
    speed: np.ndarray       # m/s
    a_tangential: np.ndarray  # m/s², signed (+ accelerating, − braking)
    a_centripetal: np.ndarray  # m/s², ≥ 0; NaN when only speed is known
    fs: float
    a_normal: np.ndarray | None = None  # signed centripetal: + turning left, − turning right
    heading: np.ndarray | None = None   # rad, direction of travel

    @property
    def a_complete(self) -> np.ndarray:
        """Magnitude of the whole acceleration vector (speed and direction changes)."""
        ac = np.nan_to_num(self.a_centripetal)
        return np.hypot(self.a_tangential, ac)

    @property
    def has_direction(self) -> bool:
        return bool(np.isfinite(self.a_centripetal).any())

    @property
    def dt(self) -> float:
        return 1.0 / self.fs

    def distance(self, mask: np.ndarray | None = None) -> float:
        v = np.nan_to_num(self.speed)
        return float((v if mask is None else v * mask).sum() * self.dt)

    @property
    def duration(self) -> float:
        return float(np.isfinite(self.speed).sum() * self.dt)


def _window(fs: float, seconds: float, order: int) -> int:
    n = max(order + 2, int(round(seconds * fs)))
    return n + 1 if n % 2 == 0 else n


def _fill_gaps(t, *ys, fs, max_gap):
    """Resample on a regular grid; gaps up to ``max_gap`` s are interpolated,
    longer gaps stay NaN (they split the recording)."""
    grid = np.arange(t[0], t[-1] + 0.5 / fs, 1.0 / fs)
    out = []
    for y in ys:
        ok = np.isfinite(y)
        yi = np.interp(grid, t[ok], y[ok])
        # mark grid points falling in long gaps
        tt = t[ok]
        idx = np.searchsorted(tt, grid)
        prev = tt[np.clip(idx - 1, 0, tt.size - 1)]
        nxt = tt[np.clip(idx, 0, tt.size - 1)]
        yi[(nxt - prev) > max_gap] = np.nan
        yi[(grid < tt[0]) | (grid > tt[-1])] = np.nan
        out.append(yi)
    return grid, out


def _segments(valid: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate(([0], valid.astype(int), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def from_positions(time: ArrayLike, x: ArrayLike, y: ArrayLike, *, fs: float | None = None,
                   smoothing_s: float = 0.6, order: int = 2, max_gap: float = 1.0,
                   max_speed: float = 10.5, max_acc: float = 10.0) -> Kinematics:
    """Kinematics from planar positions (m) sampled over ``time`` (s).

    ``smoothing_s`` is the Savitzky-Golay window: longer is smoother, shorter
    follows sharp cuts better (0.4–1.0 s is the usual range at 10–25 Hz).
    Tracking errors are removed twice: positions that jump faster than
    1.5 × ``max_speed`` from the previous one are dropped before smoothing
    (and the gap interpolated if short), then speeds above ``max_speed`` m/s
    and accelerations above ``max_acc`` m/s² are set to NaN.
    """
    t = np.asarray(time, float).ravel()
    x = np.asarray(x, float).ravel().copy()
    y = np.asarray(y, float).ravel().copy()
    if fs is None:
        fs = 1.0 / float(np.median(np.diff(t[np.isfinite(t)])))
    # drop position jumps (identity swaps, reflections) before any smoothing
    for _ in range(3):
        ok = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
        step = np.hypot(np.diff(x[ok]), np.diff(y[ok])) / np.maximum(np.diff(t[ok]), 1e-9)
        jump = ok[1:][step > 1.5 * max_speed]
        if not jump.size:
            break
        x[jump] = np.nan
        y[jump] = np.nan
    grid, (xi, yi) = _fill_gaps(t, x, y, fs=fs, max_gap=max_gap)
    w = _window(fs, smoothing_s, order)
    vx = np.full(grid.size, np.nan); vy = vx.copy(); ax = vx.copy(); ay = vx.copy()
    for s, e in _segments(np.isfinite(xi) & np.isfinite(yi)):
        if e - s < w:
            continue
        for src, v_out, a_out in ((xi, vx, ax), (yi, vy, ay)):
            v_out[s:e] = savgol_filter(src[s:e], w, order, deriv=1, delta=1 / fs)
            a_out[s:e] = savgol_filter(src[s:e], w, order, deriv=2, delta=1 / fs)
    v = np.hypot(vx, vy)
    with np.errstate(invalid="ignore", divide="ignore"):
        a_t = (vx * ax + vy * ay) / v             # projection on the direction of travel
        a_n = (vx * ay - vy * ax) / v             # perpendicular part = v²·curvature, signed
    slow = v < 0.3                                 # direction is meaningless when standing
    a_t[slow] = np.nan_to_num(a_t[slow]); a_n[slow] = 0.0
    a_c = np.abs(a_n)
    heading = np.arctan2(vy, vx)
    bad = (v > max_speed) | (np.hypot(a_t, a_c) > max_acc)
    for arr in (v, a_t, a_c, a_n, heading):
        arr[bad] = np.nan
    for a, b in _segments(np.isfinite(v)):          # islands left between errors
        if b - a < w:
            v[a:b] = a_t[a:b] = a_c[a:b] = a_n[a:b] = heading[a:b] = np.nan
    return Kinematics(grid - grid[0], v, a_t, a_c, fs, a_n, heading)


def from_speed(time: ArrayLike, speed: ArrayLike, *, fs: float | None = None,
               smoothing_s: float = 0.6, order: int = 2, max_gap: float = 1.0) -> Kinematics:
    """Kinematics from speed only (most GPS exports): no centripetal component."""
    t = np.asarray(time, float).ravel()
    v = np.asarray(speed, float).ravel()
    if fs is None:
        fs = 1.0 / float(np.median(np.diff(t[np.isfinite(t)])))
    grid, (vi,) = _fill_gaps(t, v, fs=fs, max_gap=max_gap)
    w = _window(fs, smoothing_s, order)
    vs = np.full(grid.size, np.nan); a = vs.copy()
    for s, e in _segments(np.isfinite(vi)):
        if e - s < w:
            vs[s:e] = vi[s:e]
            continue
        vs[s:e] = savgol_filter(vi[s:e], w, order)
        a[s:e] = savgol_filter(vi[s:e], w, order, deriv=1, delta=1 / fs)
    return Kinematics(grid - grid[0], np.clip(vs, 0, None), a, np.full(grid.size, np.nan), fs)
