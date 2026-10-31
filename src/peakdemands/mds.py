"""Most demanding scenarios (MDS, also "worst-case scenarios", "peak demands",
"most intense periods").

The peak of a running average over a window of fixed length, for several
lengths. Rolling windows, not fixed epochs: fixed epochs underestimate the
peak by up to 25 % for high-speed distance (Varley et al. 2012; Fereday et al.
2020), and moving averages are the method of choice in the review by
Whitehead et al. (2018). Intensity falls with window length, which a power
law describes well: I(t) = c·t^n, n < 0 (Delaney et al. 2018).

Novak et al. (2021) showed that the single worst-case value is unstable
(CV up to 76 % for sprinting) and lacks context: report it with its time,
the following window and how often near-peak passages recur, not alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kinematics import Kinematics

DURATIONS = (15, 30, 60, 90, 120, 180, 300, 600)


def _rate(kin: Kinematics, metric: str, hsr: float, acc: float) -> tuple[np.ndarray, str]:
    """A per-sample quantity whose window mean is the metric, per minute."""
    v = np.nan_to_num(kin.speed)
    if metric == "distance":
        return v * 60, "m/min"
    if metric == "hsr":
        return np.where(v >= hsr, v, 0.0) * 60, "m/min"
    if metric == "accdec":
        a = np.abs(np.nan_to_num(kin.a_tangential)) >= acc
        starts = np.diff(np.concatenate(([0], a.astype(int)))) == 1
        return starts.astype(float) * kin.fs * 60, "efforts/min"
    if metric == "power":
        from .gps3 import velocity_change_power
        return np.nan_to_num(velocity_change_power(kin)), "W/kg"
    raise ValueError("metric must be 'distance', 'hsr', 'accdec' or 'power'")


def rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Trailing mean over ``n`` samples (value i = mean of samples i-n+1..i)."""
    c = np.concatenate(([0.0], np.cumsum(x)))
    out = np.full(x.size, np.nan)
    if n <= x.size:
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


@dataclass
class PeakCurve:
    metric: str
    unit: str
    durations: np.ndarray
    peak: np.ndarray          # rolling peaks
    peak_start: np.ndarray    # s, start of the peak window
    fixed: np.ndarray         # peaks with fixed, non-overlapping epochs
    following: np.ndarray     # the next window of the same length, after the peak
    session_mean: float
    near_peak: dict = field(default_factory=dict)  # duration -> count of separate windows ≥ 90 % of the peak

    @property
    def fixed_underestimation_pct(self) -> np.ndarray:
        return 100 * (1 - self.fixed / self.peak)

    def power_law(self) -> tuple[float, float, float]:
        """Fit I = c·t^n by least squares in log-log space; returns (c, n, R²)."""
        ok = (self.peak > 0) & np.isfinite(self.peak)
        x, y = np.log(self.durations[ok]), np.log(self.peak[ok])
        n, logc = np.polyfit(x, y, 1)
        r2 = 1 - np.sum((y - (logc + n * x)) ** 2) / np.sum((y - y.mean()) ** 2)
        return float(np.exp(logc)), float(n), float(r2)

    def predict(self, duration) -> np.ndarray:
        c, n, _ = self.power_law()
        return c * np.asarray(duration, float) ** n


def peak_curve(kin: Kinematics, metric: str = "distance", durations=DURATIONS, *,
               hsr: float = 5.5, acc: float = 2.5, near: float = 0.90) -> PeakCurve:
    """Most demanding scenario for each window length, for one metric.

    ``metric``: "distance" (m/min), "hsr" (m/min above ``hsr`` m/s),
    "accdec" (accelerations + decelerations above ``acc`` m/s² started per
    minute) or "power" (velocity-change power, see :mod:`peakdemands.gps3`).
    """
    x, unit = _rate(kin, metric, hsr, acc)
    valid = np.isfinite(kin.speed)
    x = np.where(valid, x, 0.0)
    d = np.asarray(durations, float)
    peak, start, fixed, follow, near_count = [], [], [], [], {}
    for T in d:
        n = int(round(T * kin.fs))
        r = rolling_mean(x, n)
        cover = rolling_mean(valid.astype(float), n)
        r[cover < 0.9] = np.nan                     # windows mostly made of gaps don't count
        if np.all(np.isnan(r)):
            peak.append(np.nan); start.append(np.nan); fixed.append(np.nan); follow.append(np.nan)
            continue
        i = int(np.nanargmax(r))
        peak.append(float(r[i])); start.append(float(kin.time[i - n + 1]))
        j = i + n
        follow.append(float(r[j]) if j < r.size and np.isfinite(r[j]) else np.nan)
        m = x.size // n
        fixed.append(float(np.max(x[: m * n].reshape(m, n).mean(axis=1))) if m else np.nan)
        # separate passages ≥ near × peak: greedy, non-overlapping
        cnt, k, hits = 0, 0, np.flatnonzero(r >= near * r[i])
        last_end = -1
        for h in hits:
            if h - n + 1 > last_end:
                cnt += 1
                last_end = h
        near_count[float(T)] = cnt
    return PeakCurve(metric, unit, d, np.array(peak), np.array(start), np.array(fixed),
                     np.array(follow), float(np.mean(x[valid])) if valid.any() else np.nan, near_count)
