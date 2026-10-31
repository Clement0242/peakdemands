"""Bursts: discrete high-intensity efforts and how they cluster.

An effort is a run of samples above a threshold that lasts at least a
minimum duration. Definitions are parameters, because they differ between
studies and devices; the defaults are the published ones cited below.

* **Sprints / high-speed efforts** — speed above a threshold, absolute
  (e.g. > 7.0 m/s, > 5.5 m/s for HSR; Fereday et al. 2020, Novak et al. 2021)
  or relative to the player's maximal sprinting speed (MSS).
* **Near-to-MSS exposures** — efforts above 85, 90 or 95 % MSS
  (Buchheit et al. 2023).
* **Accelerations / decelerations** — tangential acceleration above a
  threshold (> 2.5 m/s² "high", > 3.5 m/s² "very high" in the review by
  Harper et al. 2019) held for a minimum time.
* **Repeated-sprint sequences** — at least 2 consecutive sprints separated by
  at most 60 s, a sprint being ≥ 1 s above 61 % of peak speed
  (Buchheit et al. 2010).
* **Repeated-sprint bouts** — at least 3 sprints with a mean recovery under
  21 s (Spencer et al. 2004).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import Kinematics


@dataclass
class Effort:
    kind: str
    start: float      # s
    end: float        # s
    peak: float       # peak of the signal during the effort
    entry_speed: float
    max_speed: float
    distance: float   # m

    @property
    def duration(self) -> float:
        return self.end - self.start


def detect(kin: Kinematics, signal: np.ndarray, threshold: float, *, min_duration: float = 1.0,
           merge_gap: float = 0.0, kind: str = "effort") -> list[Effort]:
    """Runs of ``signal >= threshold`` lasting ``min_duration`` s or more.

    Two runs separated by less than ``merge_gap`` s are merged (dropouts of a
    few samples should not split one effort into two).
    """
    on = np.nan_to_num(signal, nan=-np.inf) >= threshold
    d = np.diff(np.concatenate(([0], on.astype(int), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    gap = int(round(merge_gap * kin.fs))
    runs: list[list[int]] = []
    for s, e in zip(starts, ends):
        if runs and s - runs[-1][1] <= gap:
            runs[-1][1] = e
        else:
            runs.append([s, e])
    out = []
    v = np.nan_to_num(kin.speed)
    for s, e in runs:
        if (e - s) / kin.fs + 1e-9 < min_duration:
            continue
        seg = np.nan_to_num(signal[s:e], nan=-np.inf)
        out.append(Effort(kind, float(kin.time[s]), float(kin.time[e - 1] + kin.dt),
                          float(seg.max()), float(v[s]), float(v[s:e].max()),
                          float(v[s:e].sum() * kin.dt)))
    return out


def sprints(kin: Kinematics, threshold: float = 7.0, *, min_duration: float = 1.0,
            mss: float | None = None, relative: float | None = None) -> list[Effort]:
    """Efforts above ``threshold`` m/s, or above ``relative`` × ``mss`` if given."""
    thr = relative * mss if (relative is not None and mss) else threshold
    return detect(kin, kin.speed, thr, min_duration=min_duration, merge_gap=0.3, kind="sprint")


def near_mss_exposures(kin: Kinematics, mss: float, levels=(0.85, 0.90, 0.95), *,
                       min_duration: float = 1.0) -> dict[float, list[Effort]]:
    """Efforts above each fraction of maximal sprinting speed (Buchheit et al. 2023)."""
    return {lv: sprints(kin, mss=mss, relative=lv, min_duration=min_duration) for lv in levels}


def accelerations(kin: Kinematics, threshold: float = 2.5, *, min_duration: float = 0.5) -> list[Effort]:
    return detect(kin, kin.a_tangential, threshold, min_duration=min_duration, merge_gap=0.2,
                  kind="acceleration")


def decelerations(kin: Kinematics, threshold: float = 2.5, *, min_duration: float = 0.5) -> list[Effort]:
    out = detect(kin, -kin.a_tangential, threshold, min_duration=min_duration, merge_gap=0.2,
                 kind="deceleration")
    for e in out:
        e.peak = -e.peak
    return out


def turns(kin: Kinematics, threshold: float = 2.5, *, min_duration: float = 0.5) -> list[Effort]:
    """Efforts dominated by direction change (centripetal acceleration above threshold)."""
    if not kin.has_direction:
        return []
    return detect(kin, kin.a_centripetal, threshold, min_duration=min_duration, merge_gap=0.2, kind="turn")


@dataclass
class Sequence:
    efforts: list[Effort]

    @property
    def start(self) -> float:
        return self.efforts[0].start

    @property
    def end(self) -> float:
        return self.efforts[-1].end

    @property
    def count(self) -> int:
        return len(self.efforts)

    @property
    def recoveries(self) -> list[float]:
        return [b.start - a.end for a, b in zip(self.efforts, self.efforts[1:])]

    @property
    def mean_recovery(self) -> float:
        r = self.recoveries
        return float(np.mean(r)) if r else float("nan")


def repeated_sequences(efforts: list[Effort], *, max_recovery: float = 60.0, min_count: int = 2) -> list[Sequence]:
    """Chains of efforts separated by at most ``max_recovery`` s (Buchheit et al. 2010)."""
    ev = sorted(efforts, key=lambda e: e.start)
    seqs, cur = [], []
    for e in ev:
        if cur and e.start - cur[-1].end <= max_recovery:
            cur.append(e)
        else:
            if len(cur) >= min_count:
                seqs.append(Sequence(cur))
            cur = [e]
    if len(cur) >= min_count:
        seqs.append(Sequence(cur))
    return seqs


def repeated_sprint_bouts(efforts: list[Effort], *, max_mean_recovery: float = 21.0, min_count: int = 3) -> list[Sequence]:
    """Spencer et al. (2004): ≥ 3 sprints with a mean recovery under 21 s.

    Greedy: a bout grows while adding the next sprint keeps the mean recovery
    under the limit.
    """
    ev = sorted(efforts, key=lambda e: e.start)
    out, i = [], 0
    while i < len(ev):
        j = i + 1
        while j < len(ev):
            rec = [b.start - a.end for a, b in zip(ev[i:j + 1], ev[i + 1:j + 1])]
            if np.mean(rec) <= max_mean_recovery:
                j += 1
            else:
                break
        if j - i >= min_count:
            out.append(Sequence(ev[i:j]))
            i = j
        else:
            i += 1
    return out
