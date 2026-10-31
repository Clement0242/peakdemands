"""GPS 3.0 tools (Buchheit et al. 2026, Sport Perf Sci Rep #280).

The paper argues for direction-sensitive mechanics, intensity anchored to
the player's capacity, and *exposure* (time spent above a relative intensity
in rolling windows) rather than distance in zones. What it proposes that is
published as a formula is implemented as such:

* Bundle et al. (2003) speed-duration model: maximal speed for an all-out
  effort of duration t is MAS + (MSS − MAS)·e^(−k·t), k ≈ 0.013 s⁻¹. Averaged
  over 60 s this gives the **Estimated Maximal S1min = MAS + 0.695·(MSS − MAS)**
  (Equation 1 of the paper; MAS and MSS in km/h).
* **Estimated Maximal MP1min = 1.045 × [MAS/3.6 + 0.695·(MSS − MAS)/3.6]**
  W/kg (Equation 2), the capacity anchor for mechanical power.
* **Intensity Exposure Time (IET)**: minutes during which the 60-s rolling
  average of a metric is above 60, 70 or 80 % of a reference (the player's
  peak match value or a capacity estimate); **WIET** sums IET over a week.

The paper's mechanical work and power come from a proprietary analyser
(ADI, Hudl) that is not published. This package does **not** reproduce it:
:func:`velocity_change_power` is an open, documented proxy (speed × the
magnitude of the complete acceleration vector) that shows the same thing —
how much of the effort comes from turning rather than from speed changes —
but its values are not ADI watts and must not be compared with them, nor
with Estimated Maximal MP1min.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .kinematics import Kinematics
from .mds import rolling_mean

BUNDLE_K = 0.013  # s⁻¹


def bundle_speed(duration: ArrayLike, mas: float, mss: float, k: float = BUNDLE_K) -> np.ndarray:
    """Maximal speed sustainable in an all-out effort lasting ``duration`` s (same unit as mas, mss)."""
    t = np.asarray(duration, float)
    return mas + (mss - mas) * np.exp(-k * t)


def bundle_mean_speed(window: ArrayLike, mas: float, mss: float, k: float = BUNDLE_K) -> np.ndarray:
    """Maximal *mean* speed over an all-out effort of ``window`` s (the curve averaged over the effort)."""
    T = np.asarray(window, float)
    return mas + (mss - mas) * (1 - np.exp(-k * T)) / (k * T)


def estimated_max_s1min(mas: float, mss: float) -> float:
    """Equation 1 of the paper (km/h in, km/h out)."""
    return mas + 0.695 * (mss - mas)


def estimated_max_mp1min(mas: float, mss: float) -> float:
    """Equation 2 of the paper: W/kg, from MAS and MSS in km/h."""
    return 1.045 * (mas / 3.6 + 0.695 * (mss - mas) / 3.6)


def velocity_change_power(kin: Kinematics) -> np.ndarray:
    """Open proxy (W/kg = m²/s³): speed × |complete acceleration|.

    Zero in steady straight running, positive whenever the velocity vector
    changes, whether its length (accelerating, braking) or its direction
    (turning). Not a metabolic or mechanical power in the physiological sense.
    """
    return np.nan_to_num(kin.speed) * kin.a_complete


def directional_share(kin: Kinematics, *, min_acc: float = 1.0) -> float:
    """Share of the squared complete acceleration due to direction change,
    over samples where |a| ≥ ``min_acc`` m/s² (|a|² = a_t² + a_c²)."""
    if not kin.has_direction:
        return float("nan")
    a = kin.a_complete
    m = a >= min_acc
    ac = np.nan_to_num(kin.a_centripetal)
    return float((ac[m] ** 2).sum() / max((a[m] ** 2).sum(), 1e-12))


def intensity_exposure_time(kin: Kinematics, signal: np.ndarray, reference: float,
                            thresholds=(0.6, 0.7, 0.8), window: float = 60.0) -> dict[float, float]:
    """IET: minutes with the ``window``-s rolling mean of ``signal`` ≥ each threshold × ``reference``.

    The rolling mean is trailing; each sample counts once, so overlapping
    windows do not inflate the total.
    """
    n = int(round(window * kin.fs))
    r = rolling_mean(np.nan_to_num(signal), n)
    return {thr: float(np.nansum(r >= thr * reference) / kin.fs / 60) for thr in thresholds}


def weekly_iet(sessions: list[dict[float, float]]) -> dict[float, float]:
    """WIET: IET summed over the sessions of a week."""
    out: dict[float, float] = {}
    for s in sessions:
        for k, v in s.items():
            out[k] = out.get(k, 0.0) + v
    return out
