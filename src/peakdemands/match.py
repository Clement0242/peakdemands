"""A whole match: every player of a team, the game context, and the
summaries a physical-preparation report needs.

Context comes from the event data: who has the ball (team of the last pass,
recovery, set piece or shot) and whether the ball is in play (dead from a
ball out, a foul or a goal until the next set piece). Positions are
standardised so that the team always attacks to the right.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np

from . import efforts as E
from . import mds
from .datasets import METRICA, _cache
from .kinematics import Kinematics, from_positions
from .mechanics import MechanicalWork, mechanical_work

EVENTS = ("https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
          "Sample_Game_{g}/Sample_Game_{g}_RawEventsData.csv")
PITCH = (105.0, 68.0)


@dataclass
class PlayerMatch:
    name: str
    kin: Kinematics
    x: np.ndarray            # m, team attacking to the right
    y: np.ndarray
    mw: MechanicalWork
    on: np.ndarray           # bool, on the pitch

    @property
    def minutes(self) -> float:
        return float(self.on.sum() / self.kin.fs / 60)


@dataclass
class Match:
    team: str
    players: dict[str, PlayerMatch]
    time: np.ndarray
    fs: float
    period: np.ndarray
    in_play: np.ndarray       # bool
    possession: np.ndarray    # +1 team, -1 opponent, 0 unknown / dead
    goals: list = field(default_factory=list)   # (time s, "for"/"against")
    half_time: float = 0.0
    source: str = ""
    goalkeeper: str = ""

    @property
    def outfield(self) -> list[str]:
        return [n for n in self.players if n != self.goalkeeper]

    @property
    def starters(self) -> list[str]:
        return [n for n, p in self.players.items() if p.on[: int(60 * self.fs)].any()]


def _read(path):
    with open(path) as f:
        return list(csv.reader(f))


def load_metrica_match(game: int = 2, team: str = "Home", min_minutes: float = 5.0,
                       smoothing_s: float = 0.6) -> Match:
    """All players of ``team`` in a Metrica sample game, with possession and ball-in-play."""
    import urllib.request
    cache = _cache()
    ft = cache / f"metrica_g{game}_{team}.csv"
    fe = cache / f"metrica_g{game}_events.csv"
    if not ft.exists():
        urllib.request.urlretrieve(METRICA.format(g=game, team=team), ft)
    if not fe.exists():
        urllib.request.urlretrieve(EVENTS.format(g=game), fe)
    rows = _read(ft)
    header = rows[2]
    data = np.array([[np.nan if v == "" else float(v) for v in r] for r in rows[3:]], dtype=float)
    period, t = data[:, 0], data[:, 2]
    t1max = t[period == 1].max()
    t2min = t[period == 2].min()
    shift = t1max + 10.0 - t2min
    tc = np.where(period == 2, t + shift, t)
    fs = 25.0
    grid = np.arange(tc[0], tc[-1] + 0.5 / fs, 1 / fs)
    per = np.where(grid > t1max + 5.0, 2, 1)
    # attacking direction: the goalkeeper (first player column) stands on the side the team defends
    cols = [(j, h) for j, h in enumerate(header) if h.startswith("Player")]
    gk_j = cols[0][0]
    flip = {}
    for p_ in (1, 2):
        gx = np.nanmean(data[period == p_, gk_j])
        flip[p_] = gx > 0.5           # keeper on the right: team attacks to the left -> flip
    players = {}
    for j, name in cols:
        x = data[:, j] * PITCH[0]
        y = data[:, j + 1] * PITCH[1]
        for p_ in (1, 2):
            if flip[p_]:
                m = period == p_
                x[m] = PITCH[0] - x[m]
                y[m] = PITCH[1] - y[m]
        ok = np.isfinite(x)
        if ok.sum() / fs / 60 < min_minutes:
            continue
        kin = from_positions(tc, x, y, fs=fs, smoothing_s=smoothing_s)
        n = kin.time.size
        xi = np.interp(kin.time + grid[0], tc[ok], x[ok]); yi = np.interp(kin.time + grid[0], tc[ok], y[ok])
        on_raw = np.interp(kin.time + grid[0], tc, ok.astype(float)) > 0.5
        xi[~on_raw] = np.nan; yi[~on_raw] = np.nan
        players[name] = PlayerMatch(name, kin, xi, yi, mechanical_work(kin), on_raw & np.isfinite(kin.speed))
        grid = grid[:n]
    time = players[next(iter(players))].kin.time
    per = per[: time.size]
    # events -> possession and ball in play
    ev = _read(fe)
    hdr = ev[0]
    I = {k: hdr.index(k) for k in ("Team", "Type", "Subtype", "Period", "Start Time [s]", "End Time [s]")}
    poss = np.zeros(time.size)
    live = np.ones(time.size, bool)
    goals = []
    cur, dead_from = 0, None
    marks = []
    for r in ev[1:]:
        p_ = int(r[I["Period"]])
        ts = float(r[I["Start Time [s]"]]) + (shift if p_ == 2 else 0.0) - grid[0]
        te = float(r[I["End Time [s]"]]) + (shift if p_ == 2 else 0.0) - grid[0]
        typ, sub, tm = r[I["Type"]], r[I["Subtype"]], r[I["Team"]]
        marks.append((ts, te, typ, sub, tm))
        if typ == "SHOT" and "GOAL" in sub:
            goals.append((te, "for" if tm == team else "against"))
    marks.sort()
    idx = lambda s_: int(np.clip(round(s_ * fs), 0, time.size))
    last_t = 0.0
    restart = ("PASS", "RECOVERY", "SET PIECE", "SHOT")
    for ts, te, typ, sub, tm in marks:
        if dead_from is None:
            poss[idx(last_t):idx(ts)] = cur
        elif typ in restart:                     # play restarts (set piece, or play-on after a foul)
            live[idx(dead_from):idx(ts)] = False
            dead_from = None
        if typ in restart:
            cur = 1 if tm == team else -1
        if typ in ("BALL OUT", "FAULT RECEIVED") or (typ == "SHOT" and ("GOAL" in sub or "OUT" in sub)):
            dead_from = te if dead_from is None else min(dead_from, te)
        last_t = ts
    poss[idx(last_t):] = cur
    live[per != per] = False
    # the half-time gap
    gap = (time > t1max - grid[0]) & (time < t1max - grid[0] + 10)
    live[gap] = False
    poss[~live] = 0
    return Match(team, players, time, fs, per, live, poss, goals, float(t1max - grid[0]),
                 f"Metrica Sports sample data, game {game}, {team} team", cols[0][1])


# ---------------------------------------------------------------- summaries

def _per_min(x, mask, fs):
    n = mask.sum()
    return float(x[mask].sum() / n * 60) if n else np.nan


def player_table(m: Match, *, hsr: float = 5.5, sprint: float = 7.0, acc: float = 2.5) -> list[dict]:
    out = []
    for name, p in m.players.items():
        k = p.kin
        v = np.nan_to_num(k.speed)
        on = p.on
        dist = float((v * on).sum() / k.fs)
        mins = p.minutes
        sp = E.sprints(k, sprint)
        hs = E.sprints(k, hsr)
        c = mds.peak_curve(k, "distance", (60, 180, 300))
        cp = mds.peak_curve(k, "power", (60, 180, 300))
        ch = mds.peak_curve(k, "hsr", (60, 180, 300), hsr=hsr)
        mw = p.mw
        side = {"left": 0.0, "right": 0.0}
        for man in mw.manoeuvres:
            if man.side in side:
                side[man.side] += man.work
        lr = (side["left"] - side["right"]) / max(side["left"] + side["right"], 1e-9) * 100
        h1 = on & (m.period == 1); h2 = on & (m.period == 2)
        out.append(dict(
            name=name, minutes=mins, distance=dist, m_per_min=dist / mins if mins else np.nan,
            hsr=float((v * on * (v >= hsr)).sum() / k.fs), sprint_dist=float((v * on * (v >= sprint)).sum() / k.fs),
            sprints=len(sp), hsr_efforts=len(hs), vmax=float(np.nanmax(np.where(on, k.speed, np.nan))),
            acc=len(E.accelerations(k, acc)), dec=len(E.decelerations(k, acc)),
            rss=len(E.repeated_sequences(hs)),
            work=mw.w_total / 1000, work_per_min=mw.w_total / mins if mins else np.nan,
            dir_share=mw.direction_share * 100, lr=lr,
            peak1=c.peak[0], peak3=c.peak[1], peak5=c.peak[2],
            hsr1=ch.peak[0], hsr3=ch.peak[1],
            pow1=cp.peak[0], pow3=cp.peak[1], peak1_at=c.peak_start[0],
            h1=_per_min(v, h1, k.fs), h2=_per_min(v, h2, k.fs),
            pw_h1=_per_min(np.nan_to_num(mw.power) / 60, h1, k.fs), pw_h2=_per_min(np.nan_to_num(mw.power) / 60, h2, k.fs),
            full=mins > 85,
            sprint_paths=[(p.x[int(e.start * k.fs)], p.y[int(e.start * k.fs)],
                           p.x[min(int(e.end * k.fs), p.x.size - 1)], p.y[min(int(e.end * k.fs), p.y.size - 1)]) for e in sp],
            avg_xy=(float(np.nanmean(p.x)), float(np.nanmean(p.y))),
        ))
    return out


def phase_table(m: Match, *, hsr: float = 5.5) -> dict:
    """Team mean intensity per game phase: in possession, out of possession, ball dead."""
    phases = {"in_possession": m.in_play & (m.possession == 1),
              "out_of_possession": m.in_play & (m.possession == -1),
              "ball_dead": ~m.in_play}
    res = {k: {"share": float(v.mean() * 100)} for k, v in phases.items()}
    for k, mask in phases.items():
        d, h, pw = [], [], []
        for p in (m.players[n] for n in m.outfield):
            mm = mask & p.on
            if mm.sum() < 60 * m.fs:
                continue
            v = np.nan_to_num(p.kin.speed)
            d.append(_per_min(v, mm, m.fs)); h.append(_per_min(v * (v >= hsr), mm, m.fs))
            pw.append(float(np.nanmean(p.mw.power[mm])))
        res[k].update(m_per_min=float(np.mean(d)), hsr_per_min=float(np.mean(h)), power=float(np.mean(pw)))
    return res


def blocks(m: Match, minutes: float = 5.0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """m/min per block of ``minutes`` for each player (NaN when off the pitch)."""
    n = int(minutes * 60 * m.fs)
    nb = m.time.size // n
    out = {}
    for name, p in m.players.items():
        v = np.nan_to_num(p.kin.speed)[: nb * n].reshape(nb, n)
        on = p.on[: nb * n].reshape(nb, n)
        with np.errstate(invalid="ignore"):
            r = (v * on).sum(1) / on.sum(1) * 60
        r[on.mean(1) < 0.5] = np.nan
        out[name] = r
    starts = np.arange(nb) * minutes
    return starts, out


def team_peak_passage(m: Match, window: float = 180.0) -> dict:
    """The team's most demanding passage: mean m/min of the players on the pitch, rolling."""
    n = int(window * m.fs)
    vs = np.array([np.where(m.players[n].on, np.nan_to_num(m.players[n].kin.speed), np.nan) for n in m.outfield])
    cnt = np.isfinite(vs).sum(0)
    team = np.where(cnt > 0, np.nansum(vs, axis=0) / np.maximum(cnt, 1), 0.0) * 60
    r = mds.rolling_mean(team, n)
    i = int(np.nanargmax(r))
    s, e = i - n + 1, i + 1
    poss = m.possession[s:e]
    return {"start": float(m.time[s]), "m_per_min": float(r[i]),
            "in_possession": float((poss == 1).mean() * 100), "out_of_possession": float((poss == -1).mean() * 100),
            "dead": float((~m.in_play[s:e]).mean() * 100), "series": team}
