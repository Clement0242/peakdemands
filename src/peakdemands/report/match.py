"""Designed PDF match report for physical preparation: the whole team, the
game context, peak demands, efforts, direction-sensitive mechanical load,
and practical leads for the following days.

    from peakdemands.match import load_metrica_match
    from peakdemands.report.match import match_report
    match_report(load_metrica_match(), "match.pdf")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import __version__
from .. import mds
from ..match import PITCH, Match, blocks, phase_table, player_table, team_peak_passage
from . import REFERENCES as SESSION_REFS
from .engine import PALETTE, fig_to_uri, fmt, mpl_theme, render_pdf

TEMPLATES = Path(__file__).resolve().parent / "templates"

T = {
    "fr": dict(
        title="Rapport de match", title_html="Rapport <em>de match</em>",
        kicker="Préparation physique · analyse collective et individuelle",
        s_summary="Synthèse", s_team="L'équipe joueur par joueur", s_context="Charge selon le contexte de jeu",
        s_time="Le match dans le temps", s_peak="Demandes de pointe", s_efforts="Sprints et efforts",
        s_mech="Charge mécanique et direction", s_leads="Pistes pour la semaine", s_method="Méthode",
        reading="Lecture", player="Joueur", role="Rôle*", min_="Min", dist="Distance", mpm="m/min", hsr="HSR (m)",
        spr="Sprints", vmax="V max", accdec="Acc/Déc", work="Travail", dirsh="Dir.", lr="G/D",
        p1="Pic 1 min", p3="Pic 3 min", p5="Pic 5 min", pw1="P 1 min", rss="Séq.",
        in_poss="Avec ballon", out_poss="Sans ballon", dead="Ballon mort", share="Temps",
        h1="1re MT", h2="2e MT", delta="Écart",
        drill="Repères pour les exercices (joueurs de champ ayant joué tout le match)",
        duration="Durée", median="Médiane", p90="90e centile",
        cap_pos="Positions moyennes (équipe attaquant vers la droite) ; taille du point = distance parcourue.",
        cap_phase="Intensité moyenne des joueurs de champ selon la phase de jeu.",
        cap_heat="Distance par minute, par tranche de 5 minutes ; gris : hors du terrain. Traits : mi-temps (blanc) et buts (or pour, rouge contre).",
        cap_halves="Évolution de la distance par minute entre les deux mi-temps (joueurs ayant joué les deux).",
        cap_team="Intensité de l'équipe (joueurs de champ, moyenne glissante sur 3 min) ; bande : passage le plus exigeant ; "
                 "tirets : buts (or pour, rouge contre) ; trait gris : mi-temps.",
        cap_mds="Pic de distance par minute selon la durée de fenêtre, par joueur.",
        cap_spr="Sprints (> 7 m/s) du match : départ (point) et arrivée (flèche), équipe attaquant vers la droite.",
        cap_rose="Signature de mouvement de l'équipe : travail mécanique selon l'orientation de l'accélération.",
        cap_mech="Travail mécanique par minute : part liée à la vitesse et part liée à la direction.",
        roles={"GK": "G", "DEF": "Déf", "MID": "Mil", "ATT": "Att"},
        disclaimer="Pistes indicatives, à croiser avec le ressenti des joueurs, le staff médical et le calendrier.",
        none="Rien à signaler.",
    ),
    "en": dict(
        title="Match report", title_html="Match <em>report</em>",
        kicker="Physical preparation · team and individual analysis",
        s_summary="Summary", s_team="The team player by player", s_context="Load by game context",
        s_time="The match over time", s_peak="Peak demands", s_efforts="Sprints and efforts",
        s_mech="Mechanical load and direction", s_leads="Leads for the week", s_method="Method",
        reading="Reading", player="Player", role="Role*", min_="Min", dist="Distance", mpm="m/min", hsr="HSR (m)",
        spr="Sprints", vmax="Max v", accdec="Acc/Dec", work="Work", dirsh="Dir.", lr="L/R",
        p1="Peak 1 min", p3="Peak 3 min", p5="Peak 5 min", pw1="P 1 min", rss="Seq.",
        in_poss="In possession", out_poss="Out of possession", dead="Ball dead", share="Time",
        h1="1st half", h2="2nd half", delta="Change",
        drill="References for drills (outfield players who played the whole match)",
        duration="Duration", median="Median", p90="90th percentile",
        cap_pos="Average positions (team attacking to the right); dot size = distance covered.",
        cap_phase="Mean intensity of outfield players by game phase.",
        cap_heat="Distance per minute by 5-minute block; grey: off the pitch. Lines: half-time (white) and goals (gold for, red against).",
        cap_halves="Change in distance per minute between halves (players who played both).",
        cap_team="Team intensity (outfield players, 3-min rolling average); band: most demanding passage; dashes: goals "
                 "(gold for, red against); grey line: half-time.",
        cap_mds="Peak distance per minute against window length, per player.",
        cap_spr="Sprints (> 7 m/s) of the match: start (dot) and end (arrow), team attacking to the right.",
        cap_rose="Team movement signature: mechanical work by orientation of the acceleration.",
        cap_mech="Mechanical work per minute: speed-change and direction-change parts.",
        roles={"GK": "GK", "DEF": "Def", "MID": "Mid", "ATT": "Att"},
        disclaimer="Indicative leads, to be crossed with the players' feedback, the medical staff and the schedule.",
        none="Nothing to flag.",
    ),
}

REFERENCES = SESSION_REFS + [
    "Metrica Sports sample data (anonymised professional match, optical tracking 25 Hz and event data), "
    "github.com/metrica-sports/sample-data.",
]


def _label(name: str) -> str:
    return name.replace("Player", "J") if name.startswith("Player") else name


def _roles(m: Match, rows) -> dict[str, str]:
    out = {m.goalkeeper: "GK"}
    starters = [r for r in rows if r["name"] != m.goalkeeper and r["minutes"] > 45]
    xs = sorted(r["avg_xy"][0] for r in starters)
    if len(xs) >= 3:
        t1, t2 = np.quantile(xs, [1 / 3, 2 / 3])
    else:
        t1 = t2 = np.median(xs) if xs else 50
    for r in rows:
        if r["name"] == m.goalkeeper:
            continue
        x = r["avg_xy"][0]
        out[r["name"]] = "DEF" if x <= t1 else ("MID" if x <= t2 else "ATT")
    return out


def _pitch(ax, P):
    L, W = PITCH
    ax.set_facecolor("#F4F6F2")
    kw = dict(color="#9BA39A", lw=0.8)
    ax.plot([0, L, L, 0, 0], [0, 0, W, W, 0], **kw)
    ax.plot([L / 2, L / 2], [0, W], **kw)
    th = np.linspace(0, 2 * np.pi, 80)
    ax.plot(L / 2 + 9.15 * np.cos(th), W / 2 + 9.15 * np.sin(th), **kw)
    for x0, s in ((0, 1), (L, -1)):
        ax.plot([x0, x0 + s * 16.5, x0 + s * 16.5, x0], [W / 2 - 20.16, W / 2 - 20.16, W / 2 + 20.16, W / 2 + 20.16], **kw)
        ax.plot([x0, x0 + s * 5.5, x0 + s * 5.5, x0], [W / 2 - 9.16, W / 2 - 9.16, W / 2 + 9.16, W / 2 + 9.16], **kw)
    ax.set_xlim(-2, L + 2); ax.set_ylim(-2, W + 2); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _figures(m: Match, rows, roles, peak, lang):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    mpl_theme()
    P = PALETTE
    L = T[lang]
    out = {}
    by = {r["name"]: r for r in rows}
    role_col = {"GK": P["mut"], "DEF": P["navy"], "MID": P["gold_dark"], "ATT": P["red"]}

    # positions
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    _pitch(ax, P)
    dmax = max(r["distance"] for r in rows)
    for r in rows:
        x, y = r["avg_xy"]
        ax.scatter(x, y, s=40 + 260 * r["distance"] / dmax, color=role_col[roles[r["name"]]], alpha=.85,
                   edgecolor="white", lw=.8, zorder=3)
        ax.text(x, y, _label(r["name"]), color="white", fontsize=6.5, ha="center", va="center", zorder=4, fontweight="bold")
    out["fig_pos"] = fig_to_uri(fig)

    # phases
    ph = phase_table(m)
    keys = ("in_possession", "out_of_possession", "ball_dead")
    names = (L["in_poss"], L["out_poss"], L["dead"])
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.0))
    for ax, (k, lab, unit) in zip(axes, (("m_per_min", "m/min", ""), ("hsr_per_min", "HSR m/min", ""), ("power", "W/kg", ""))):
        vals = [ph[c][k] for c in keys]
        ax.bar(names, vals, color=[P["navy"], P["red"], P["mut"]], width=.6)
        for i_, v_ in enumerate(vals):
            ax.text(i_, v_, fmt(v_, 0 if k != "power" else 2, lang), ha="center", va="bottom", fontsize=7.5, color=P["ink"])
        ax.set_title(lab, fontsize=8.5)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_ylim(0, max(vals) * 1.22)
    out["fig_phase"] = fig_to_uri(fig)
    out["phase"] = ph

    # heatmap of 5-min blocks
    starts, bl = blocks(m, 5.0)
    order = sorted(rows, key=lambda r: ({"GK": 0, "DEF": 1, "MID": 2, "ATT": 3}[roles[r["name"]]], -r["minutes"]))
    Z = np.array([bl[r["name"]] for r in order])
    cmap = LinearSegmentedColormap.from_list("pd", ["#F9E9CA", P["gold"], P["red"], P["navy"]])
    cmap.set_bad("#E6E6EC")
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    im = ax.imshow(np.ma.masked_invalid(Z), aspect="auto", cmap=cmap, vmin=60, vmax=150,
                   extent=(0, starts[-1] + 5, len(order), 0))
    ax.set_yticks(np.arange(len(order)) + .5, [f"{_label(r['name'])} · {L['roles'][roles[r['name']]]}" for r in order], fontsize=7)
    ax.axvline(m.half_time / 60, color="white", lw=2)
    for t_, side in m.goals:
        ax.axvline(t_ / 60, color=P["gold"] if side == "for" else P["red"], lw=1.4, ls="--")
    ax.set_xlabel("min"); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=.025, pad=.01); cb.set_label("m/min", fontsize=7); cb.ax.tick_params(labelsize=6)
    out["fig_heat"] = fig_to_uri(fig)

    # halves
    both = [r for r in rows if np.isfinite(r["h1"]) and np.isfinite(r["h2"]) and r["name"] != m.goalkeeper]
    both.sort(key=lambda r: (r["h2"] - r["h1"]) / r["h1"])
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    vals = [100 * (r["h2"] - r["h1"]) / r["h1"] for r in both]
    ax.barh([_label(r["name"]) for r in both], vals, color=[P["red"] if v < np.mean(vals) - 5 else P["navy"] for v in vals])
    ax.axvline(np.mean(vals), color=P["gold_dark"], ls="--", lw=1)
    ax.set_xlabel("%"); ax.tick_params(axis="y", labelsize=7)
    out["fig_halves"] = fig_to_uri(fig)

    # team timeline
    fig, ax = plt.subplots(figsize=(7.4, 1.8))
    r3 = mds.rolling_mean(peak["series"], int(180 * m.fs))
    r3[~m.in_play & False] = np.nan
    ax.plot(m.time / 60, r3, color=P["navy"], lw=1.3)
    ax.axvspan(peak["start"] / 60, peak["start"] / 60 + 3, color=P["gold_light"], alpha=.6, lw=0)
    for t_, side in m.goals:
        ax.axvline(t_ / 60, color=P["gold"] if side == "for" else P["red"], lw=1.2, ls="--")
    ax.axvline(m.half_time / 60, color=P["mut"], lw=1)
    ax.set(xlabel="min", ylabel="m/min", xlim=(0, m.time[-1] / 60))
    out["fig_team"] = fig_to_uri(fig)

    # MDS curves
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    durs = (30, 60, 120, 180, 300, 600)
    for r in order:
        if r["name"] == m.goalkeeper or r["minutes"] < 30:
            continue
        c = mds.peak_curve(m.players[r["name"]].kin, "distance", durs)
        ax.plot(c.durations, c.peak, "-o", ms=2.5, lw=1, color=role_col[roles[r["name"]]], alpha=.8)
    ax.set_xscale("log"); ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xticks(durs, ["30 s", "1", "2", "3", "5", "10 min"], fontsize=7)
    ax.set_ylabel("m/min")
    out["fig_mds"] = fig_to_uri(fig)

    # sprint map
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    _pitch(ax, P)
    for r in rows:
        col = role_col[roles[r["name"]]]
        for x0, y0, x1, y1 in r["sprint_paths"]:
            if np.all(np.isfinite([x0, y0, x1, y1])):
                ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1, alpha=.85))
                ax.plot(x0, y0, "o", ms=2.5, color=col)
    out["fig_spr"] = fig_to_uri(fig)

    # team rose
    psi_all, w_all = [], []
    for n in m.outfield:
        p = m.players[n]
        psi_all.append(np.arctan2(np.nan_to_num(p.kin.a_normal), np.nan_to_num(p.kin.a_tangential)))
        w_all.append(np.nan_to_num(p.mw.power) / m.fs)
    psi_all = np.concatenate(psi_all); w_all = np.concatenate(w_all)
    bins = np.linspace(-np.pi, np.pi, 17)
    h, _ = np.histogram(psi_all, bins=bins, weights=w_all)
    fig = plt.figure(figsize=(3.0, 2.8))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(1)
    cen = (bins[:-1] + bins[1:]) / 2
    bars = ax.bar(cen, h / h.sum() * 100, width=2 * np.pi / 16 * .92,
                  color=[P["navy"] if abs(c) < np.pi / 4 or abs(c) > 3 * np.pi / 4 else P["red"] for c in cen], alpha=.85)
    for b in bars:
        b.set_clip_on(False)
    ax.set_xticks([0, np.pi / 2, np.pi, -np.pi / 2], ["accél." if lang == "fr" else "accel.",
                                                       "gauche" if lang == "fr" else "left",
                                                       "frein" if lang == "fr" else "brake",
                                                       "droite" if lang == "fr" else "right"], fontsize=7)
    ax.set_yticklabels([]); ax.grid(False); ax.spines["polar"].set_visible(False)
    out["fig_rose"] = fig_to_uri(fig)

    # mechanical stacked bars
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    sel = [r for r in order if r["minutes"] >= 30]
    sp_ = [m.players[r["name"]].mw.w_speed / r["minutes"] for r in sel]
    di_ = [m.players[r["name"]].mw.w_direction / r["minutes"] for r in sel]
    lab = [_label(r["name"]) for r in sel]
    ax.bar(lab, sp_, color=P["navy"], label="vitesse" if lang == "fr" else "speed")
    ax.bar(lab, di_, bottom=sp_, color=P["red"], label="direction")
    ax.set_ylabel("J/kg/min"); ax.tick_params(axis="x", labelsize=6.5, rotation=0); ax.legend(fontsize=7)
    out["fig_mech"] = fig_to_uri(fig)
    return out


def _leads(rows, roles, m, lang):
    f = lambda x, d=0: fmt(x, d, lang)
    full = [r for r in rows if r["full"] and r["name"] != m.goalkeeper]
    leads = []
    if full:
        w = np.array([r["work_per_min"] for r in full])
        mu, sd = w.mean(), w.std()
        hi = [r for r in full if r["work_per_min"] > mu + sd]
        drops = [(r, 100 * (r["h2"] - r["h1"]) / r["h1"]) for r in full]
        mdrop = np.mean([d for _, d in drops])
        steep = [(r, d) for r, d in drops if d < mdrop - 5]
        p3 = np.median([r["peak3"] for r in full])
        hsr3 = np.median([r["hsr3"] for r in full])
        spr = np.median([r["sprints"] for r in full])
        short = [r for r in rows if r["minutes"] < 60 and r["name"] != m.goalkeeper]
        few = [r for r in rows if r["sprints"] <= 1 and r["minutes"] >= 60 and r["name"] != m.goalkeeper]
        asym = [r for r in rows if abs(r["lr"]) >= 15 and r["minutes"] >= 30 and r["name"] != m.goalkeeper]
        names = lambda lst: ", ".join(_label(r["name"]) for r in lst)
        if lang == "fr":
            if hi:
                leads.append(("Récupération", f"Charge mécanique la plus élevée de l'équipe (> moyenne + 1 écart-type) : "
                              f"<b>{names(hi)}</b>. Priorité à la récupération à J+1 et à J+2, volume de course réduit."))
            if steep:
                leads.append(("Fin de match", "Baisse en 2e mi-temps nettement plus forte que l'équipe "
                              f"({f(mdrop, 0)} % en moyenne) : <b>" + ", ".join(f"{_label(r['name'])} ({f(d, 0)} %)" for r, d in steep) +
                              "</b>. À croiser avec le contexte (score, consignes) avant d'y lire de la fatigue."))
            if short:
                leads.append(("Compensation", f"Temps de jeu inférieur à 60 min : <b>{names(short)}</b>. Repères pour une séance "
                              f"de compensation, tirés du match des titulaires : un bloc de 3 min à ~{f(p3, 0)} m/min dont "
                              f"~{f(hsr3, 0)} m/min à haute vitesse, et ~{f(spr, 0)} sprints répartis dans la séance."))
            if few:
                leads.append(("Vitesse", f"Un sprint ou moins au-dessus de 7 m/s : <b>{names(few)}</b>. Une exposition "
                              "proche de la vitesse maximale dans la semaine peut être envisagée (Buchheit et al. 2023)."))
            if asym:
                leads.append(("Asymétrie", "Travail des virages déséquilibré (≥ 15 % entre gauche et droite) : <b>" +
                              ", ".join(f"{_label(r['name'])} ({f(r['lr'], 0)} %)" for r in asym) +
                              "</b>. Souvent lié au poste ; à suivre sur plusieurs matchs avant d'agir."))
        else:
            if hi:
                leads.append(("Recovery", f"Highest mechanical load of the team (> mean + 1 SD): <b>{names(hi)}</b>. "
                              "Recovery first at D+1 and D+2, reduced running volume."))
            if steep:
                leads.append(("End of match", f"Second-half drop clearly larger than the team's ({f(mdrop, 0)} % on average): <b>" +
                              ", ".join(f"{_label(r['name'])} ({f(d, 0)} %)" for r, d in steep) +
                              "</b>. Check the context (score, instructions) before reading fatigue into it."))
            if short:
                leads.append(("Top-up", f"Less than 60 min played: <b>{names(short)}</b>. References for a top-up session, from the "
                              f"starters' match: a 3-min block at ~{f(p3, 0)} m/min with ~{f(hsr3, 0)} m/min at high speed, "
                              f"and ~{f(spr, 0)} sprints spread over the session."))
            if few:
                leads.append(("Speed", f"One sprint or fewer above 7 m/s: <b>{names(few)}</b>. A near-maximal speed exposure "
                              "during the week may be considered (Buchheit et al. 2023)."))
            if asym:
                leads.append(("Asymmetry", "Unbalanced turning work (≥ 15 % between left and right): <b>" +
                              ", ".join(f"{_label(r['name'])} ({f(r['lr'], 0)} %)" for r in asym) +
                              "</b>. Often linked to the position; follow over several matches before acting."))
    return leads


def match_report(m: Match, out: str | Path, *, title: str | None = None, match_date: str | None = None,
                 lang: str = "fr", organisation: str | None = None, logo: str | Path | None = None) -> Path:
    """Write the full physical-preparation match report of one team."""
    if lang not in T:
        raise ValueError("lang must be 'fr' or 'en'")
    L = T[lang]
    f = lambda x, d=0: fmt(x, d, lang) if np.isfinite(x) else "—"
    rows = player_table(m)
    roles = _roles(m, rows)
    peak = team_peak_passage(m)
    figs = _figures(m, rows, roles, peak, lang)
    ph = figs.pop("phase")
    full = [r for r in rows if r["full"] and r["name"] != m.goalkeeper]
    out_rows = sorted(rows, key=lambda r: ({"GK": 0, "DEF": 1, "MID": 2, "ATT": 3}[roles[r["name"]]], -r["minutes"]))
    table = [[_label(r["name"]), L["roles"][roles[r["name"]]], f(r["minutes"]), f(r["distance"]), f(r["m_per_min"]),
              f(r["hsr"]), f(r["sprint_dist"]), str(r["sprints"]), f(r["vmax"], 2), f"{r['acc']}/{r['dec']}", str(r["rss"])]
             for r in out_rows]
    peaks = [[_label(r["name"]), f(r["peak1"]), f(r["peak3"]), f(r["peak5"]), f(r["hsr1"]), f(r["hsr3"]),
              f(r["pow1"], 2), f(r["pow3"], 2), f"{int(r['peak1_at'] // 60)}'" if np.isfinite(r["peak1_at"]) else "—"]
             for r in out_rows if r["name"] != m.goalkeeper]
    mech = [[_label(r["name"]), f(r["work"], 2), f(r["work_per_min"]), f(r["dir_share"]), f(r["lr"]),
             str(len(m.players[r["name"]].mw.manoeuvres))] for r in out_rows if r["minutes"] >= 30]
    drill = []
    if full:
        for lab, k1, k2, k3 in (("1 min", "peak1", "hsr1", "pow1"), ("3 min", "peak3", "hsr3", "pow3"), ("5 min", "peak5", None, None)):
            a = np.array([r[k1] for r in full])
            row = [lab, f(np.median(a)), f(np.percentile(a, 90))]
            if k2:
                b = np.array([r[k2] for r in full]); c = np.array([r[k3] for r in full])
                row += [f(np.median(b)), f(np.percentile(b, 90)), f(np.median(c), 2), f(np.percentile(c, 90), 2)]
            else:
                row += ["—"] * 4
            drill.append(row)
    team_dist = np.mean([r["distance"] for r in full]) if full else np.nan
    team_mpm = np.mean([r["m_per_min"] for r in full]) if full else np.nan
    goals_for = sum(1 for _, s in m.goals if s == "for"); goals_ag = len(m.goals) - goals_for
    dir_team = 100 * sum(m.players[n].mw.w_direction for n in m.outfield) / sum(m.players[n].mw.w_total for n in m.outfield)
    tiles = [
        {"val": f"{goals_for} – {goals_ag}", "unit": "", "lab": "Score", "sub": f"{len(m.players)} " + ("joueurs suivis" if lang == "fr" else "players tracked"), "accent": True},
        {"val": f(m.in_play.mean() * 100), "unit": "%", "lab": "Temps de jeu effectif" if lang == "fr" else "Ball in play",
         "sub": f"{f(ph['in_possession']['share'])} % " + ("avec ballon" if lang == "fr" else "in possession")},
        {"val": f(team_dist / 1000, 1), "unit": "km", "lab": "Distance moyenne" if lang == "fr" else "Mean distance",
         "sub": f"{f(team_mpm)} m/min · " + ("titulaires de champ" if lang == "fr" else "full-match outfield")},
        {"val": f(np.mean([r['sprints'] for r in full]) if full else np.nan, 1), "unit": "", "lab": "Sprints / joueur" if lang == "fr" else "Sprints / player",
         "sub": "> 7 m/s"},
        {"val": f(dir_team), "unit": "%", "lab": "Travail directionnel" if lang == "fr" else "Directional work",
         "sub": "virages / travail total" if lang == "fr" else "turning / total work"},
    ]
    top_mpm = max(full, key=lambda r: r["m_per_min"]) if full else None
    top_work = max(full, key=lambda r: r["work_per_min"]) if full else None
    drop = np.mean([100 * (r["h2"] - r["h1"]) / r["h1"] for r in full]) if full else np.nan
    o, i_ = ph["out_of_possession"], ph["in_possession"]
    if lang == "fr":
        reading = [
            f"Les joueurs de champ ayant joué tout le match ont couvert <b>{f(team_dist / 1000, 1)} km</b> en moyenne "
            f"({f(team_mpm)} m/min). Le plus intense : <b>{_label(top_mpm['name'])}</b> ({f(top_mpm['m_per_min'])} m/min) ; "
            f"la plus forte charge mécanique : <b>{_label(top_work['name'])}</b> ({f(top_work['work_per_min'])} J/kg/min).",
            f"Sans ballon, l'équipe court <b>{f(o['m_per_min'] / i_['m_per_min'] * 100 - 100)} % de plus</b> qu'avec "
            f"({f(o['m_per_min'])} contre {f(i_['m_per_min'])} m/min) et fournit {f(o['hsr_per_min'] / max(i_['hsr_per_min'], 1e-9), 1)} fois "
            "plus de haute vitesse : le pressing et le repli pèsent plus que la conservation.",
            f"De la 1re à la 2e mi-temps, la distance par minute baisse de <b>{f(-drop)} %</b> en moyenne.",
            f"Passage collectif le plus exigeant (3 min) : <b>{f(peak['m_per_min'])} m/min</b> à la {int(peak['start'] // 60) + 1}e minute, "
            f"{f(peak['out_of_possession'])} % du temps sans ballon.",
            f"<b>{f(dir_team)} %</b> du travail mécanique de l'équipe vient des changements de direction, une charge invisible "
            "pour les métriques de distance et de vitesse.",
        ]
    else:
        reading = [
            f"Outfield players who played the whole match covered <b>{f(team_dist / 1000, 1)} km</b> on average "
            f"({f(team_mpm)} m/min). Most intense: <b>{_label(top_mpm['name'])}</b> ({f(top_mpm['m_per_min'])} m/min); "
            f"highest mechanical load: <b>{_label(top_work['name'])}</b> ({f(top_work['work_per_min'])} J/kg/min).",
            f"Out of possession the team runs <b>{f(o['m_per_min'] / i_['m_per_min'] * 100 - 100)} % more</b> than in possession "
            f"({f(o['m_per_min'])} vs {f(i_['m_per_min'])} m/min) with {f(o['hsr_per_min'] / max(i_['hsr_per_min'], 1e-9), 1)} times "
            "more high-speed running: pressing and recovery runs weigh more than keeping the ball.",
            f"From the first to the second half, distance per minute falls by <b>{f(-drop)} %</b> on average.",
            f"Most demanding team passage (3 min): <b>{f(peak['m_per_min'])} m/min</b> in minute {int(peak['start'] // 60) + 1}, "
            f"{f(peak['out_of_possession'])} % of it out of possession.",
            f"<b>{f(dir_team)} %</b> of the team's mechanical work comes from changes of direction, a load invisible to "
            "distance and speed metrics.",
        ]
    halves = [[_label(r["name"]), f(r["h1"]), f(r["h2"]), f(100 * (r["h2"] - r["h1"]) / r["h1"]) + " %",
               f(r["pw_h1"], 2), f(r["pw_h2"], 2)]
              for r in out_rows if np.isfinite(r["h1"]) and np.isfinite(r["h2"]) and r["name"] != m.goalkeeper]
    method = {
        "fr": [
            "Positions optiques à 25 Hz lissées et dérivées (Savitzky-Golay 0,6 s), sauts de tracking retirés ; l'équipe est "
            "représentée attaquant vers la droite. Rôles estimés d'après la position moyenne (tiers de la profondeur).",
            "Contexte de jeu tiré des événements : possession = équipe du dernier événement de jeu (passe, récupération, coup "
            "de pied arrêté, tir) ; ballon mort d'une sortie, d'une faute ou d'un but jusqu'à la reprise.",
            "Haute vitesse > 5,5 m/s, sprint > 7 m/s tenu 1 s ; accélérations > 2,5 m/s² tenues 0,5 s ; séquences répétées : ≥ 2 "
            "efforts à haute vitesse à ≤ 60 s (Buchheit et al. 2010). Pics en fenêtres glissantes (Varley et al. 2012).",
            "Charge mécanique : travail lié à la vitesse (variation d'énergie cinétique, Gray et al. 2018, filtre à hystérésis "
            "0,2 m/s) et lié à la direction (énergie cinétique par axe dans le repère de chaque manœuvre, Cavagna et al. 1976). "
            "Modèle ouvert, horizontal : non comparable aux watts d'un analyseur commercial.",
            "Pistes : règles simples et transparentes (au-delà de la moyenne + 1 écart-type, baisse supérieure de 5 points à "
            "celle de l'équipe, < 60 min jouées, ≤ 1 sprint, asymétrie ≥ 15 %), à interpréter par le staff."],
        "en": [
            "Optical positions at 25 Hz smoothed and differentiated (Savitzky-Golay 0.6 s), tracking jumps removed; the team "
            "is shown attacking to the right. Roles estimated from the average position (thirds of the depth).",
            "Game context from the events: possession = team of the last on-ball event (pass, recovery, set piece, shot); "
            "ball dead from a ball out, a foul or a goal until the restart.",
            "High speed > 5.5 m/s, sprint > 7 m/s held 1 s; accelerations > 2.5 m/s² held 0.5 s; repeated sequences: ≥ 2 "
            "high-speed efforts ≤ 60 s apart (Buchheit et al. 2010). Peaks over rolling windows (Varley et al. 2012).",
            "Mechanical load: speed-change work (kinetic energy changes, Gray et al. 2018, 0.2 m/s hysteresis) and "
            "direction-change work (per-axis kinetic energy in each manoeuvre's frame, Cavagna et al. 1976). Open, "
            "horizontal model: not comparable with a commercial analyser's watts.",
            "Leads: simple, transparent rules (above mean + 1 SD, drop 5 points steeper than the team's, < 60 min played, "
            "≤ 1 sprint, asymmetry ≥ 15 %), for the staff to interpret."],
    }[lang]
    ctx = {
        "lang": lang, "L": L, "title": L["title"], "title_html": L["title_html"], "kicker": L["kicker"],
        "athlete": title or ("Équipe à domicile — match 2" if lang == "fr" else "Home team — game 2"),
        "test": "Match", "test_date": match_date or ("non communiquée" if lang == "fr" else "not given"), "extra_meta": {},
        "organisation": organisation, "logo": logo, "tool": "peakdemands", "version": __version__,
        "tiles": tiles, "reading": reading, "table": table, "peaks": peaks, "mech": mech, "drill": drill,
        "halves": halves, "phase": [(L["in_poss"], ph["in_possession"]), (L["out_poss"], ph["out_of_possession"]),
                                    (L["dead"], ph["ball_dead"])],
        "leads": _leads(rows, roles, m, lang), "method": method, "references": REFERENCES, "source": m.source,
        "disclaimer": L["disclaimer"], **figs,
    }
    return render_pdf(TEMPLATES, "match.html", ctx, out)


__all__ = ["match_report"]
