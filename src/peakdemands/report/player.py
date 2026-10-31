"""Single-player match report: one player, every number set against his
team-mates and the game context. Clear before complete.

    from peakdemands.match import load_metrica_match
    from peakdemands.report.player import player_report
    player_report(load_metrica_match(), "Player5", "j5.pdf")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import __version__
from .. import efforts as E
from .. import mds
from ..match import PITCH, Match, phase_table, player_table
from ..mechanics import FAMILIES
from .engine import PALETTE, fig_to_uri, fmt, mpl_theme, render_pdf
from .match import REFERENCES, _label, _pitch, _roles

TEMPLATES = Path(__file__).resolve().parent / "templates"

ROLE = {"fr": {"GK": "gardien", "DEF": "défenseur", "MID": "milieu", "ATT": "attaquant"},
        "en": {"GK": "goalkeeper", "DEF": "defender", "MID": "midfielder", "ATT": "forward"}}

# metric key, FR label, EN label, unit, digits
PROFILE = [
    ("m_per_min", "Distance par minute", "Distance per minute", "m/min", 0),
    ("hsr_pm", "Haute vitesse par minute", "High speed per minute", "m/min", 1),
    ("sprints_90", "Sprints par 90 min", "Sprints per 90 min", "", 1),
    ("vmax", "Vitesse maximale", "Maximal speed", "m/s", 2),
    ("accdec_90", "Acc. + déc. par 90 min", "Acc. + dec. per 90 min", "", 0),
    ("work_per_min", "Travail mécanique", "Mechanical work", "J/kg/min", 0),
    ("dir_share", "Part liée aux virages", "Share from turning", "%", 0),
    ("peak1", "Pic 1 min", "Peak 1 min", "m/min", 0),
    ("peak3", "Pic 3 min", "Peak 3 min", "m/min", 0),
    ("rss", "Séquences répétées", "Repeated sequences", "", 0),
]


def _enrich(r):
    mins = max(r["minutes"], 1e-9)
    r["hsr_pm"] = r["hsr"] / mins
    r["sprints_90"] = r["sprints"] / mins * 90
    r["accdec_90"] = (r["acc"] + r["dec"]) / mins * 90
    return r


def _player_context(m: Match, name: str, hsr: float = 5.5) -> dict:
    p = m.players[name]
    v = np.nan_to_num(p.kin.speed)
    out = {}
    for k, mask in (("in", m.in_play & (m.possession == 1)), ("out", m.in_play & (m.possession == -1)),
                    ("dead", ~m.in_play)):
        mm = mask & p.on
        n = mm.sum()
        out[k] = dict(minutes=n / m.fs / 60, m_per_min=v[mm].sum() / max(n, 1) * 60,
                      hsr_per_min=(v * (v >= hsr))[mm].sum() / max(n, 1) * 60,
                      power=float(np.nanmean(p.mw.power[mm])) if n else np.nan)
    return out


def _phase_at(m: Match, t: float) -> str:
    i = int(np.clip(round(t * m.fs), 0, m.time.size - 1))
    if not m.in_play[i]:
        return "dead"
    return {1: "in", -1: "out"}.get(int(m.possession[i]), "dead")


def player_report(m: Match, player: str, out: str | Path, *, lang: str = "fr", organisation: str | None = None,
                  logo: str | Path | None = None, match_date: str | None = None, mss: float | None = None,
                  hsr: float = 5.5, sprint: float = 7.0) -> Path:
    """Write the match report of one player, compared with his team-mates."""
    import matplotlib.pyplot as plt

    if player not in m.players:
        raise ValueError(f"{player!r} not in {list(m.players)}")
    fr = lang == "fr"
    f = lambda x, d=0: fmt(x, d, lang) if x is not None and np.isfinite(x) else "—"
    rows = [_enrich(r) for r in player_table(m, hsr=hsr, sprint=sprint)]
    roles = _roles(m, rows)
    me = next(r for r in rows if r["name"] == player)
    peers = [r for r in rows if r["name"] != m.goalkeeper and r["minutes"] >= 45]
    if me not in peers:
        peers.append(me)
    p = m.players[player]
    k = p.kin
    who = _label(player)
    role = ROLE[lang][roles[player]]

    def rank(key, higher=True):
        vals = sorted([r[key] for r in peers], reverse=higher)
        return vals.index(me[key]) + 1, len(vals)

    mpl_theme()
    P = PALETTE
    figs = {}

    # 1. profile against the team: dot = player, line = team range, tick = team median
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    for i, (key, lfr, len_, unit, d) in enumerate(PROFILE):
        vals = np.array([r[key] for r in peers], float)
        lo, hi, med = np.nanmin(vals), np.nanmax(vals), np.nanmedian(vals)
        span = hi - lo if hi > lo else 1.0
        y = len(PROFILE) - 1 - i
        ax.plot([0, 1], [y, y], color="#E4E4EC", lw=6, solid_capstyle="round")
        ax.plot((med - lo) / span, y, "|", color=P["mut"], ms=12, mew=2)
        xv = (me[key] - lo) / span
        ax.plot(xv, y, "o", color=P["red"], ms=9, zorder=3)
        ax.text(-0.02, y, lfr if fr else len_, ha="right", va="center", fontsize=8, color=P["ink"])
        ax.text(1.06, y, f"{f(me[key], d)} {unit}".strip(), ha="left", va="center", fontsize=8, color=P["red"], fontweight="bold")
        rk, n = rank(key)
        ax.text(1.30, y, f"{rk}/{n}", ha="left", va="center", fontsize=7.5, color=P["mut"])
    ax.set_xlim(-0.6, 1.4); ax.set_ylim(-0.7, len(PROFILE) - 0.3)
    ax.axis("off")
    ax.text(0, len(PROFILE) - 0.25, "min équipe" if fr else "team min", fontsize=6.5, color=P["mut"], ha="center")
    ax.text(1, len(PROFILE) - 0.25, "max équipe" if fr else "team max", fontsize=6.5, color=P["mut"], ha="center")
    ax.text(1.30, len(PROFILE) - 0.25, "rang" if fr else "rank", fontsize=6.5, color=P["mut"])
    figs["fig_profile"] = fig_to_uri(fig)

    # 2. timeline: 1-min rolling distance, HSR shading, sprints, goals, possession strip
    v = np.where(p.on, np.nan_to_num(k.speed), np.nan)
    r1 = mds.rolling_mean(np.nan_to_num(v) * 60, int(180 * m.fs))
    r1[~p.on] = np.nan
    team = []
    for n_ in m.outfield:
        q = m.players[n_]
        team.append(np.where(q.on, np.nan_to_num(q.kin.speed), np.nan))
    tm = np.array(team)
    cnt = np.isfinite(tm).sum(0)
    tmean = mds.rolling_mean(np.where(cnt > 0, np.nansum(tm, 0) / np.maximum(cnt, 1), 0) * 60, int(180 * m.fs))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.4, 2.7), sharex=True, gridspec_kw={"height_ratios": [5, 0.6], "hspace": 0.08})
    tt = m.time / 60
    ax.plot(tt, tmean, color=P["mut"], lw=0.9, label="équipe" if fr else "team")
    ax.plot(tt, r1, color=P["navy"], lw=1.3, label=who)
    sp = E.sprints(k, sprint)
    for e in sp:
        ax.axvline(e.start / 60, ymin=0, ymax=0.07, color=P["red"], lw=1.6)
    for t_, side in m.goals:
        ax.axvline(t_ / 60, color=P["gold"] if side == "for" else P["red"], lw=1, ls="--")
    ax.axvline(m.half_time / 60, color=P["mut"], lw=0.8)
    ax.set_ylabel("m/min"); ax.legend(fontsize=7, loc="upper right", ncol=2)
    strip = np.where(~m.in_play, 0, m.possession)
    ax2.imshow(strip[None, :], aspect="auto", extent=(0, tt[-1], 0, 1),
               cmap=plt.matplotlib.colors.ListedColormap([P["red"], "#E6E6EC", P["navy"]]), vmin=-1, vmax=1)
    ax2.set_yticks([]); ax2.set_xlabel("min"); ax2.grid(False)
    figs["fig_time"] = fig_to_uri(fig)

    # 3. peak demands against the team band
    durs = np.array([15, 30, 60, 120, 180, 300, 600], float)
    curves = {r["name"]: mds.peak_curve(m.players[r["name"]].kin, "distance", durs).peak for r in peers}
    allp = np.array(list(curves.values()))
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    ax.fill_between(durs, allp.min(0), allp.max(0), color=P["gold_light"], alpha=.45, lw=0, label="équipe min–max" if fr else "team min–max")
    ax.plot(durs, np.median(allp, 0), color=P["mut"], lw=1, ls="--", label="médiane" if fr else "median")
    ax.plot(durs, curves[player], "-o", color=P["red"], ms=4, lw=1.6, label=who)
    ax.set_xscale("log"); ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xticks(durs, ["15s", "30s", "1", "2", "3", "5", "10min"], fontsize=7)
    ax.set_ylabel("m/min"); ax.legend(fontsize=6.5)
    figs["fig_mds"] = fig_to_uri(fig)

    # 4. occupation heatmap + sprints on the pitch
    fig, ax = plt.subplots(figsize=(4.0, 2.75))
    _pitch(ax, P)
    ok = np.isfinite(p.x) & np.isfinite(p.y)
    H, xe, ye = np.histogram2d(p.x[ok], p.y[ok], bins=(21, 14), range=((0, PITCH[0]), (0, PITCH[1])))
    ax.imshow(H.T, origin="lower", extent=(0, PITCH[0], 0, PITCH[1]), cmap="Blues", alpha=.75, aspect="auto", zorder=1)
    for e in sp:
        i0 = int(e.start * m.fs); i1 = min(int(e.end * m.fs), p.x.size - 1)
        if np.all(np.isfinite([p.x[i0], p.y[i0], p.x[i1], p.y[i1]])):
            ax.annotate("", (p.x[i1], p.y[i1]), (p.x[i0], p.y[i0]), arrowprops=dict(arrowstyle="-|>", color=P["red"], lw=1.3), zorder=3)
    ax.set_xlim(-2, PITCH[0] + 2); ax.set_ylim(-2, PITCH[1] + 2)
    figs["fig_pitch"] = fig_to_uri(fig)

    # 5. movement signature
    psi = np.arctan2(np.nan_to_num(k.a_normal), np.nan_to_num(k.a_tangential))
    w = np.nan_to_num(p.mw.power) / m.fs
    bins = np.linspace(-np.pi, np.pi, 17)
    h, _ = np.histogram(psi, bins=bins, weights=w)
    fig = plt.figure(figsize=(2.3, 2.0))
    ax = fig.add_subplot(projection="polar"); ax.set_theta_zero_location("N"); ax.set_theta_direction(1)
    cen = (bins[:-1] + bins[1:]) / 2
    bars = ax.bar(cen, h / h.sum() * 100, width=2 * np.pi / 16 * .92,
                  color=[P["navy"] if abs(c) < np.pi / 4 or abs(c) > 3 * np.pi / 4 else P["red"] for c in cen], alpha=.85)
    for b in bars:
        b.set_clip_on(False)
    ax.set_xticks([0, np.pi / 2, np.pi, -np.pi / 2], ["accél." if fr else "accel.", "gauche" if fr else "left",
                                                       "frein" if fr else "brake", "droite" if fr else "right"], fontsize=7)
    ax.set_yticklabels([]); ax.grid(False); ax.spines["polar"].set_visible(False)
    figs["fig_rose"] = fig_to_uri(fig)

    # tables
    ctx = _player_context(m, player, hsr)
    team_ph = phase_table(m, hsr=hsr)
    names = {"in": "Avec ballon" if fr else "In possession", "out": "Sans ballon" if fr else "Out of possession",
             "dead": "Ballon mort" if fr else "Ball dead"}
    tkey = {"in": "in_possession", "out": "out_of_possession", "dead": "ball_dead"}
    context = [[names[c], f(ctx[c]["minutes"]), f(ctx[c]["m_per_min"]), f(team_ph[tkey[c]]["m_per_min"]),
                f(ctx[c]["hsr_per_min"], 1), f(team_ph[tkey[c]]["hsr_per_min"], 1), f(ctx[c]["power"], 2)]
               for c in ("in", "out", "dead")]
    h1 = p.on & (m.period == 1); h2 = p.on & (m.period == 2)
    vv = np.nan_to_num(k.speed)
    def half(mask):
        n = mask.sum()
        if n < 60 * m.fs:
            return [np.nan] * 5
        mins = n / m.fs / 60
        s_ = [e for e in sp if mask[int(e.start * m.fs)]]
        return [vv[mask].sum() / n * 60, (vv * (vv >= hsr))[mask].sum() / n * 60, len(s_) / mins * 45,
                float(np.nanmean(p.mw.power[mask])), float(np.nanmean(p.mw.p_direction[mask]) / max(np.nanmean(p.mw.power[mask]), 1e-9) * 100)]
    a, b = half(h1), half(h2)
    labs = (["Distance / min", "Haute vitesse / min", "Sprints / 45 min", "Puissance (W/kg)", "Part virages (%)"] if fr else
            ["Distance / min", "High speed / min", "Sprints / 45 min", "Power (W/kg)", "Turning share (%)"])
    digs = [0, 1, 1, 2, 0]
    halves = [[l_, f(x, d_), f(y, d_), (f(100 * (y - x) / x, 0) + " %") if np.isfinite(x) and np.isfinite(y) and x else "—"]
              for l_, x, y, d_ in zip(labs, a, b, digs)]
    sprints = [[f"{int(e.start // 60) + 1}'", f(e.duration, 1) + " s", f(e.max_speed, 2), f(e.entry_speed, 2),
                f(e.distance), names[_phase_at(m, e.start)]] for e in sp]
    seqs = E.repeated_sequences(E.sprints(k, hsr))
    peaks_tab = []
    for d_, lab in ((60, "1 min"), (180, "3 min"), (300, "5 min")):
        j = list(durs).index(d_)
        vals = sorted(allp[:, j], reverse=True)
        peaks_tab.append([lab, f(curves[player][j]), f(np.median(allp[:, j])), f"{vals.index(curves[player][j]) + 1}/{len(vals)}"])
    mw = p.mw
    fam = {}
    for mn in mw.manoeuvres:
        d__ = fam.setdefault(mn.family, [0, 0.0, 0.0, 0.0])
        d__[0] += 1; d__[1] += mn.work
        if mn.side == "left":
            d__[2] += mn.work
        elif mn.side == "right":
            d__[3] += mn.work
    tot = sum(v_[1] for v_ in fam.values()) or 1
    famname = ({"acceleration": "Accélérations", "deceleration": "Décélérations", "turn": "Virages",
                "curved acceleration": "Accélérations en courbe", "curved deceleration": "Freinages en courbe"} if fr else
               {"acceleration": "Accelerations", "deceleration": "Decelerations", "turn": "Turns",
                "curved acceleration": "Curved accelerations", "curved deceleration": "Curved braking"})
    families = [[famname[fm], str(fam[fm][0]), f(fam[fm][1] / tot * 100) + " %",
                 (f(fam[fm][2] / (fam[fm][2] + fam[fm][3]) * 100) + " %") if (fam[fm][2] + fam[fm][3]) > 0 else "—"]
                for fm in FAMILIES if fm in fam]

    # reading and leads
    rk_mpm, n = rank("m_per_min"); rk_w, _ = rank("work_per_min"); rk_h, _ = rank("hsr_pm")
    drop = 100 * (b[0] - a[0]) / a[0] if np.isfinite(a[0]) and np.isfinite(b[0]) else np.nan
    team_drop = np.nanmean([100 * (r["h2"] - r["h1"]) / r["h1"] for r in peers if np.isfinite(r["h1"]) and np.isfinite(r["h2"])])
    out_vs_in = ctx["out"]["m_per_min"] / max(ctx["in"]["m_per_min"], 1e-9) * 100 - 100
    n_out = sum(1 for e in sp if _phase_at(m, e.start) == "out")
    if fr:
        reading = [
            f"<b>{who}</b>, {role}, {f(me['minutes'])} min : <b>{f(me['distance'] / 1000, 2)} km</b> à {f(me['m_per_min'])} m/min, "
            f"{rk_mpm}<sup>e</sup> sur {n} joueurs de champ ; haute vitesse {rk_h}<sup>e</sup>, charge mécanique {rk_w}<sup>e</sup>.",
            f"{me['sprints']} sprints au-dessus de {f(sprint, 0)} m/s (vitesse maximale {f(me['vmax'], 2)} m/s), dont {n_out} sans ballon ; "
            f"{len(seqs)} séquences d'efforts à haute vitesse répétés.",
            f"Sans ballon, il court {f(abs(out_vs_in))} % {'de plus' if out_vs_in >= 0 else 'de moins'} qu'avec "
            f"({f(ctx['out']['m_per_min'])} contre {f(ctx['in']['m_per_min'])} m/min).",
            (f"De la 1re à la 2e mi-temps, sa distance par minute évolue de <b>{f(drop)} %</b> (équipe : {f(team_drop)} %)."
             if np.isfinite(drop) else "Il n'a joué qu'une mi-temps : pas de comparaison entre périodes."),
            f"<b>{f(me['dir_share'])} %</b> de son travail mécanique vient des virages ; sa minute la plus intense atteint "
            f"<b>{f(me['peak1'])} m/min</b> à la {int(me['peak1_at'] // 60) + 1}<sup>e</sup> minute.",
        ]
    else:
        reading = [
            f"<b>{who}</b>, {role}, {f(me['minutes'])} min: <b>{f(me['distance'] / 1000, 2)} km</b> at {f(me['m_per_min'])} m/min, "
            f"rank {rk_mpm} of {n} outfield players; high speed rank {rk_h}, mechanical load rank {rk_w}.",
            f"{me['sprints']} sprints above {f(sprint, 0)} m/s (max speed {f(me['vmax'], 2)} m/s), {n_out} of them out of possession; "
            f"{len(seqs)} sequences of repeated high-speed efforts.",
            f"Out of possession he runs {f(abs(out_vs_in))} % {'more' if out_vs_in >= 0 else 'less'} than in possession "
            f"({f(ctx['out']['m_per_min'])} vs {f(ctx['in']['m_per_min'])} m/min).",
            (f"From the first to the second half his distance per minute changes by <b>{f(drop)} %</b> (team: {f(team_drop)} %)."
             if np.isfinite(drop) else "He played one half only: no comparison between periods."),
            f"<b>{f(me['dir_share'])} %</b> of his mechanical work comes from turning; his most intense minute reaches "
            f"<b>{f(me['peak1'])} m/min</b> in minute {int(me['peak1_at'] // 60) + 1}.",
        ]
    leads = []
    wv = np.array([r["work_per_min"] for r in peers])
    if me["work_per_min"] > wv.mean() + wv.std():
        leads.append(("Récupération" if fr else "Recovery",
                      "Charge mécanique parmi les plus élevées de l'équipe : récupération prioritaire à J+1 et J+2, volume de course réduit."
                      if fr else "Mechanical load among the team's highest: recovery first at D+1 and D+2, reduced running volume."))
    if np.isfinite(drop) and drop < team_drop - 5:
        leads.append(("Fin de match" if fr else "End of match",
                      f"Baisse en 2e mi-temps plus marquée que l'équipe ({f(drop)} % contre {f(team_drop)} %) : à croiser avec le score et les consignes."
                      if fr else f"Second-half drop steeper than the team's ({f(drop)} % vs {f(team_drop)} %): check score and instructions."))
    if me["sprints"] <= 1:
        leads.append(("Vitesse" if fr else "Speed",
                      "Peu d'exposition à la vitesse maximale : un stimulus proche de la VMS dans la semaine peut être envisagé (Buchheit et al. 2023)."
                      if fr else "Little exposure to maximal speed: a near-maximal speed stimulus during the week may be considered (Buchheit et al. 2023)."))
    if mss and me["vmax"] < 0.9 * mss:
        leads.append(("Vitesse" if fr else "Speed", (f"Vitesse maximale du match à {f(100 * me['vmax'] / mss)} % de sa VMS." if fr else
                                                     f"Match peak speed at {f(100 * me['vmax'] / mss)} % of his MSS.")))
    if abs(me["lr"]) >= 15:
        leads.append(("Asymétrie" if fr else "Asymmetry",
                      f"Virages déséquilibrés ({f(me['lr'])} % {'vers la gauche' if me['lr'] > 0 else 'vers la droite'}) : souvent lié au poste, à suivre sur plusieurs matchs."
                      if fr else f"Unbalanced turning ({f(me['lr'])} % to the {'left' if me['lr'] > 0 else 'right'}): often positional, follow over several matches."))
    if me["minutes"] < 60:
        leads.append(("Compensation" if fr else "Top-up",
                      f"Moins de 60 min jouées : compléter avec un bloc de 3 min à ~{f(np.median(allp[:, 4]))} m/min et quelques sprints."
                      if fr else f"Under 60 min played: top up with a 3-min block at ~{f(np.median(allp[:, 4]))} m/min and a few sprints."))
    peak_target = [[lab, f(curves[player][list(durs).index(d_)]), f(0.8 * curves[player][list(durs).index(d_)]),
                    f(0.9 * curves[player][list(durs).index(d_)])] for d_, lab in ((60, "1 min"), (180, "3 min"), (300, "5 min"))]

    tiles = [
        {"val": f(me["distance"] / 1000, 2), "unit": "km", "lab": "Distance", "sub": f"{f(me['minutes'])} min", "accent": True},
        {"val": f(me["m_per_min"]), "unit": "m/min", "lab": "Intensité" if fr else "Intensity", "sub": f"{'rang' if fr else 'rank'} {rk_mpm}/{n}"},
        {"val": f(me["hsr"]), "unit": "m", "lab": f"> {f(hsr, 1)} m/s", "sub": f"{f(me['sprint_dist'])} m > {f(sprint, 0)} m/s"},
        {"val": str(me["sprints"]), "unit": "", "lab": "Sprints", "sub": f"max {f(me['vmax'], 2)} m/s"},
        {"val": f(me["work_per_min"]), "unit": "J/kg/min", "lab": "Travail mécanique" if fr else "Mechanical work",
         "sub": f"{f(me['dir_share'])} % {'virages' if fr else 'turning'}"},
    ]
    method = ([
        "Comparaisons avec les joueurs de champ de l'équipe ayant joué ≥ 45 min. Haute vitesse > 5,5 m/s, sprint > 7 m/s "
        "tenu 1 s, séquences répétées à ≤ 60 s (Buchheit et al. 2010), pics en fenêtres glissantes (Varley et al. 2012). "
        "Contexte tiré des événements du match (possession = équipe du dernier événement de jeu). Charge mécanique : "
        "vitesse (Gray et al. 2018) et virages (Cavagna et al. 1976), modèle ouvert et horizontal."] if fr else [
        "Compared with the team's outfield players with ≥ 45 min. High speed > 5.5 m/s, sprint > 7 m/s held 1 s, repeated "
        "sequences ≤ 60 s apart (Buchheit et al. 2010), peaks over rolling windows (Varley et al. 2012). Context from the "
        "match events (possession = team of the last on-ball event). Mechanical load: speed (Gray et al. 2018) and turning "
        "(Cavagna et al. 1976), open horizontal model."])
    ctx_ = {
        "lang": lang, "fr": fr, "title": ("Rapport joueur" if fr else "Player report"),
        "title_html": (f"Rapport <em>joueur</em> · {who}" if fr else f"Player <em>report</em> · {who}"),
        "kicker": ("Préparation physique · analyse individuelle de match" if fr else "Physical preparation · individual match analysis"),
        "athlete": f"{who} — {role}", "test": "Match", "test_date": match_date or ("non communiquée" if fr else "not given"),
        "extra_meta": {("Temps de jeu" if fr else "Minutes"): f"{f(me['minutes'])} min"},
        "organisation": organisation, "logo": logo, "tool": "peakdemands", "version": __version__,
        "tiles": tiles, "reading": reading, "context": context, "halves": halves, "sprints": sprints,
        "peaks": peaks_tab, "families": families, "leads": leads, "peak_target": peak_target, "who": who,
        "lr": f(me["lr"]), "method": method, "references": REFERENCES, "source": m.source,
        "disclaimer": ("Pistes indicatives, à croiser avec le ressenti du joueur, le staff médical et le calendrier."
                       if fr else "Indicative leads, to be crossed with the player's feedback, the medical staff and the schedule."),
        **figs,
    }
    return render_pdf(TEMPLATES, "player.html", ctx_, out)


__all__ = ["player_report"]
