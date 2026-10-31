"""Direction-sensitive mechanical work and power, and movement signatures.

GPS 3.0 (Buchheit et al. 2026) asks for mechanical work and power that count
the effort of changing *direction* as well as speed. The analyser it uses
(ADI, Hudl; Gray) is proprietary, so this module builds an open model from
published physics, and says where it is a modelling choice.

Speed-change work — Gray et al. (2018)
    The horizontal work done on the centre of mass is the change in kinetic
    energy between samples, summed in absolute value (positive work when
    accelerating, negative when braking): W_speed = Σ |Δ(½v²)| (J/kg).
    By construction it is blind to turning: a curve at constant speed gives 0.

Direction-change work — per-axis kinetic energy (Cavagna et al. 1976)
    Cavagna counted the kinetic energy of the centre of mass separately along
    each axis. Along fixed axes, a turn moves kinetic energy from one axis to
    the other: it has to be absorbed in the old direction and produced in the
    new one, which the speed-only count misses. Here the axes are the
    athlete's own: for every manoeuvre (a run of samples with centripetal
    acceleration ≥ ``min_acc``), the frame is set by the heading at its
    start, the kinetic energy is split into the parts along and across that
    heading, and

        W_direction = Σ|Δ(½v∥²)| + Σ|Δ(½v⊥²)| − Σ|Δ(½v²)|   over the manoeuvre (≥ 0).

    Worked values (per kg, speed v): straight run 0; stop-and-go 180° cut 0
    (all of it is speed work, v²); a 90° turn at constant speed v² (½v²
    absorbed along the old direction, ½v² produced along the new one); a
    U-turn on a curve at constant speed 2v². These are the tests.

Noise. Summing |ΔKE| sample by sample also sums tracking noise: on a
simulated run with 5 cm position noise at 25 Hz it doubles the true work.
Speed first goes through a backlash (hysteresis) filter of width
``deadband`` (0.2 m/s by default): changes smaller than the deadband are
not counted. On the same simulations the error falls to about 5 %
(tests/test_mechanics.py).

Mechanical power is the time derivative of the work (W/kg); its rolling
peaks, bursts and exposure time follow the GPS 3.0 logic. The numbers are
those of this model: they are not ADI watts and are not comparable with them
or with the paper's Estimated Maximal MP1min. The model is horizontal and
centre-of-mass only (no vertical, internal or air work; Gray et al. 2018,
Fig. 5, list those as further components).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import Kinematics, _segments

FAMILIES = ("acceleration", "deceleration", "turn", "curved acceleration", "curved deceleration")


@dataclass
class Manoeuvre:
    start: float
    end: float
    family: str
    side: str            # "left" / "right" / "" for linear
    entry_speed: float
    heading_change: float  # degrees, signed (+ left)
    peak_acc: float      # peak |complete acceleration|, m/s²
    w_speed: float       # J/kg
    w_direction: float   # J/kg

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def work(self) -> float:
        return self.w_speed + self.w_direction


@dataclass
class MechanicalWork:
    time: np.ndarray
    p_speed: np.ndarray       # W/kg
    p_direction: np.ndarray   # W/kg
    manoeuvres: list[Manoeuvre]
    fs: float

    @property
    def power(self) -> np.ndarray:
        return self.p_speed + self.p_direction

    @property
    def w_speed(self) -> float:
        return float(np.nansum(self.p_speed) / self.fs)

    @property
    def w_direction(self) -> float:
        return float(np.nansum(self.p_direction) / self.fs)

    @property
    def w_total(self) -> float:
        return self.w_speed + self.w_direction

    @property
    def direction_share(self) -> float:
        return self.w_direction / self.w_total if self.w_total else float("nan")

    def signature(self, by: str = "work") -> dict[tuple[str, str], float]:
        """Movement signature: total work (J/kg), count or time (s) per family and side."""
        out: dict[tuple[str, str], float] = {}
        for m in self.manoeuvres:
            key = (m.family, m.side)
            val = {"work": m.work, "count": 1.0, "time": m.duration}[by]
            out[key] = out.get(key, 0.0) + val
        return out


def _family(at_mean: float, an_mean: float) -> tuple[str, str]:
    phi = np.degrees(np.arctan2(abs(an_mean), at_mean))   # 0 = pure acceleration, 180 = pure braking
    side = "left" if an_mean > 0 else "right"
    if phi < 30:
        return "acceleration", ""
    if phi > 150:
        return "deceleration", ""
    if phi < 60:
        return "curved acceleration", side
    if phi > 120:
        return "curved deceleration", side
    return "turn", side


def backlash(x: np.ndarray, width: float) -> np.ndarray:
    """Hysteresis filter: the output moves only when the input leaves a band of ``width`` around it."""
    y = np.array(x, float)
    h = width / 2
    for i in range(1, y.size):
        y[i] = min(max(y[i - 1], x[i] - h), x[i] + h)
    return y


def mechanical_work(kin: Kinematics, *, min_acc: float = 1.0, min_duration: float = 0.3,
                    min_speed: float = 1.0, deadband: float = 0.2) -> MechanicalWork:
    """Speed- and direction-change work and power from positional kinematics (per kg).

    With speed-only data (no heading) the direction term is 0 and the result
    is the Gray et al. (2018) horizontal work.
    """
    v = np.nan_to_num(kin.speed)
    ok = np.isfinite(kin.speed)
    vb = backlash(v, deadband) if deadband > 0 else v
    dke = np.diff(0.5 * vb ** 2)
    dke[~(ok[1:] & ok[:-1])] = 0.0           # no work across gaps
    p_speed = np.concatenate(([0.0], np.abs(dke))) * kin.fs
    p_dir = np.zeros(v.size)
    mans: list[Manoeuvre] = []
    at = np.nan_to_num(kin.a_tangential)
    if kin.heading is not None and kin.a_normal is not None:
        an = np.nan_to_num(kin.a_normal)
        a = np.hypot(at, an)
        active = (a >= min_acc) & (v >= min_speed) & np.isfinite(kin.heading)
        gap = int(round(0.2 * kin.fs))
        segs = _segments(active)
        merged: list[list[int]] = []
        for s, e in segs:
            if merged and s - merged[-1][1] <= gap:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        for s, e in merged:
            if (e - s) / kin.fs < min_duration:
                continue
            s0 = max(s - 1, 0)
            th = np.unwrap(np.nan_to_num(kin.heading[s0:e + 1]))
            rel = th - th[0]
            vv = v[s0:e + 1]
            v_par, v_perp = vv * np.cos(rel), vv * np.sin(rel)
            w_axes = np.abs(np.diff(0.5 * v_par ** 2)).sum() + np.abs(np.diff(0.5 * v_perp ** 2)).sum()
            w_spd = np.abs(np.diff(0.5 * vv ** 2)).sum()   # same accounting as p_speed
            w_dir = max(w_axes - w_spd, 0.0)
            # spread the direction work over the manoeuvre in proportion to |a_n|·v
            weight = np.abs(an[s:e]) * v[s:e]
            if weight.sum() > 0 and w_dir > 0:
                p_dir[s:e] += w_dir * kin.fs * weight / weight.sum()
            fam, side = _family(float(at[s:e].mean()), float(an[s:e].mean()))
            mans.append(Manoeuvre(float(kin.time[s]), float(kin.time[e - 1] + 1 / kin.fs), fam, side,
                                  float(v[s]), float(np.degrees(rel[-1])), float(a[s:e].max()),
                                  float(p_speed[s:e].sum() / kin.fs), float(w_dir)))
    return MechanicalWork(kin.time, p_speed, p_dir, mans, kin.fs)
