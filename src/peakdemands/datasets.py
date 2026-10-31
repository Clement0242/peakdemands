"""Example data.

* :func:`simulate_session` — a **synthetic** 20-min session with known
  sprints, accelerations and turns, for examples and tests.
* :func:`load_metrica` — **real** optical tracking of a full professional
  match, from the Metrica Sports sample data (anonymised; 25 Hz; pitch
  105 × 68 m). The files are not redistributed here: they are downloaded
  from github.com/metrica-sports/sample-data on first use and cached.
  Source to acknowledge: Metrica Sports.
"""

from __future__ import annotations

import csv
import os
import urllib.request
from pathlib import Path

import numpy as np

METRICA = ("https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
           "Sample_Game_{g}/Sample_Game_{g}_RawTrackingData_{team}_Team.csv")


def _cache() -> Path:
    p = Path(os.environ.get("PEAKDEMANDS_CACHE", Path.home() / ".cache" / "peakdemands"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_metrica(player: str = "Player9", game: int = 2, team: str = "Home") -> dict:
    """Positions (m) of one player over a whole match: ``time`` (s), ``x``, ``y``, ``period``.

    Coordinates are converted from the 0-1 scale to metres (105 × 68). The
    two halves are joined with a 10-s gap of NaN so that no window spans the
    break.
    """
    f = _cache() / f"metrica_g{game}_{team}.csv"
    if not f.exists():
        urllib.request.urlretrieve(METRICA.format(g=game, team=team), f)
    with f.open() as fh:
        rows = list(csv.reader(fh))
    header = rows[2]
    try:
        j = header.index(player)
    except ValueError as exc:
        raise ValueError(f"{player} not in {[h for h in header if h.startswith('Player')]}") from exc
    data = np.array([[r[0], r[2], r[j], r[j + 1]] for r in rows[3:]], dtype=float)
    period, t, x, y = data.T
    # second half continues after the first, with a gap
    t2 = t.copy()
    if (period == 2).any():
        t2[period == 2] = t[period == 2] - t[period == 2].min() + t[period == 1].max() + 10.0
    return {"time": t2, "x": x * 105.0, "y": y * 68.0, "period": period.astype(int),
            "source": "Metrica Sports sample data, game %d, %s team, %s" % (game, team, player)}


def simulate_session(seed: int = 3, fs: float = 10.0) -> dict:
    """20 min: jogging and walking, 6 straight sprints, 4 curved sprints, 8 cuts.

    Returns positions (``time``, ``x``, ``y``) and the ground truth
    (``sprint_starts``) used by the tests.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, 1200, 1 / fs)
    speed = np.where((t // 60) % 2 == 0, 2.2, 1.2)            # jog / walk minutes
    heading = np.zeros(t.size)
    sprint_starts = [100.0, 130.0, 160.0, 400.0, 700.0, 1000.0]
    for s in sprint_starts:                                   # 4-s sprints peaking at 8 m/s
        m = (t >= s) & (t < s + 4)
        speed[m] = 8.0 * np.minimum(1, (t[m] - s + 0.6) / 1.6)
    for s in (250.0, 550.0, 850.0, 1100.0):                   # curved sprints, constant speed
        m = (t >= s) & (t < s + 5)
        speed[m] = 7.0
        heading[m] = np.cumsum(np.full(m.sum(), 0.5 / fs))    # 0.5 rad/s turn: a_c = 3.5 m/s²
    for s in np.arange(300, 1200, 110.0):                     # cuts: 90° change in 0.5 s at 4 m/s
        m = (t >= s) & (t < s + 0.5)
        speed[(t >= s - 1) & (t < s + 1.5)] = 4.0
        heading[m] = np.pi / 2 * (t[m] - s) / 0.5
        heading[(t >= s + 0.5) & (t < s + 1.5)] = np.pi / 2
    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)
    x = np.cumsum(vx) / fs + rng.normal(0, 0.03, t.size)
    y = np.cumsum(vy) / fs + rng.normal(0, 0.03, t.size)
    return {"time": t, "x": x, "y": y, "sprint_starts": sprint_starts, "true_speed": speed}
