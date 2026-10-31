"""Designed PDF report of one session or match: bursts, most demanding
scenarios and GPS 3.0 exposure.

    from peakdemands.report import session_report
    session_report(kin, "match.pdf", athlete="Player 9", session="Match", mss=9.2, mas=16.5)

``pip install "peakdemands[report]"``. Your organisation and logo are
options; nothing is added by default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import __version__
from .. import efforts as E
from .. import gps3, mds
from ..mechanics import FAMILIES, mechanical_work
from ..kinematics import Kinematics
from .engine import PALETTE, fig_to_uri, fmt, mpl_theme, render_pdf

TEMPLATES = Path(__file__).resolve().parent / "templates"

L = {
    "fr": dict(title="Demandes de pointe", title_html="Demandes <em>de pointe</em>",
               kicker="Tracking · efforts et scénarios les plus exigeants", summary="Synthèse", reading="Lecture",
               mds="Scénarios les plus exigeants", timeline="Le match minute par minute", bursts="Efforts",
               gps3="Direction et exposition (GPS 3.0)", method="Méthode", duration="Fenêtre",
               rolling="Glissante", fixed="Fixe", under="Écart", at="Début", following="Fenêtre suivante",
               near="Passages ≥ 90 %", start="Début", dur="Durée", vmax="Vitesse max", entry="Vitesse d'entrée",
               dist="Distance", kind="Type", count="Nombre", seq="Séquences de sprints répétés",
               iet="Temps d'exposition (IET)", thr="Seuil", minutes="Minutes", ref="Référence",
               cap_mds="Pic de la moyenne glissante (points) et par périodes fixes (losanges) pour chaque durée de "
                       "fenêtre ; trait : loi de puissance ajustée.",
               cap_tl="Distance par minute (moyenne glissante sur 60 s). Bandes : fenêtre la plus exigeante de 1 et "
                      "5 min ; traits rouges : sprints.",
               cap_dir="Chaque point est un instant d'accélération ≥ 1 m/s² : composante liée à la vitesse (horizontale) "
                       "et au changement de direction (verticale).",
               sprints="Sprints", accs="Accélérations", decs="Décélérations", turns="Changements de direction",
               rss="Séquences répétées", peak1="Pic 1 min", hsr="Haute vitesse", total="Distance totale",
               dir_share="Part directionnelle", none="Aucun", rec="Récup.", data="Données :",
               mds_legend="m/min : distance ; HSR : m/min au-dessus du seuil ; acc+dec : efforts par minute ; Fixe : "
                          "sous-estimation par périodes fixes ; Début : moment du pic (min:s) ; ≥ 90 % : passages distincts "
                          "à 90 % du pic.",
               mp_note="repère du papier pour une puissance mécanique non reproduite ici.",
               mech="Charge mécanique directionnelle", w_tot="Travail horizontal", w_dir="Part directionnelle",
               mp1="Puissance pic 1 min", mbursts="Bursts mécaniques", family="Manœuvre", work="Travail",
               fam_names={"acceleration": "Accélération", "deceleration": "Décélération", "turn": "Virage",
                          "curved acceleration": "Accélération en courbe", "curved deceleration": "Freinage en courbe"},
               left="gauche", right="droite",
               cap_rose="Signature de mouvement : travail mécanique selon l'orientation de l'accélération (haut : "
                        "accélérer, bas : freiner, côtés : tourner à gauche / à droite)."),
    "en": dict(title="Peak demands", title_html="Peak <em>demands</em>",
               kicker="Tracking · efforts and most demanding scenarios", summary="Summary", reading="Reading",
               mds="Most demanding scenarios", timeline="The match minute by minute", bursts="Efforts",
               gps3="Direction and exposure (GPS 3.0)", method="Method", duration="Window",
               rolling="Rolling", fixed="Fixed", under="Gap", at="Start", following="Next window",
               near="Passages ≥ 90 %", start="Start", dur="Duration", vmax="Max speed", entry="Entry speed",
               dist="Distance", kind="Type", count="Count", seq="Repeated-sprint sequences",
               iet="Intensity exposure time (IET)", thr="Threshold", minutes="Minutes", ref="Reference",
               cap_mds="Peak of the rolling average (dots) and of fixed epochs (diamonds) for each window length; "
                       "line: fitted power law.",
               cap_tl="Distance per minute (60-s rolling average). Bands: most demanding 1- and 5-min windows; red "
                      "ticks: sprints.",
               cap_dir="Each dot is an instant with acceleration ≥ 1 m/s²: speed-change component (horizontal) and "
                       "direction-change component (vertical).",
               sprints="Sprints", accs="Accelerations", decs="Decelerations", turns="Direction changes",
               rss="Repeated sequences", peak1="Peak 1 min", hsr="High speed", total="Total distance",
               dir_share="Directional share", none="None", rec="Recovery", data="Data:",
               mds_legend="m/min: distance; HSR: m/min above the threshold; acc+dec: efforts per minute; Fixed: "
                          "underestimation by fixed epochs; Start: time of the peak (min:s); ≥ 90 %: distinct passages "
                          "at 90 % of the peak.",
               mp_note="the paper's reference for a mechanical power not reproduced here.",
               mech="Direction-sensitive mechanical load", w_tot="Horizontal work", w_dir="Directional share",
               mp1="Peak 1-min power", mbursts="Mechanical bursts", family="Manoeuvre", work="Work",
               fam_names={"acceleration": "Acceleration", "deceleration": "Deceleration", "turn": "Turn",
                          "curved acceleration": "Curved acceleration", "curved deceleration": "Curved braking"},
               left="left", right="right",
               cap_rose="Movement signature: mechanical work by orientation of the acceleration (top: speeding up, "
                        "bottom: braking, sides: turning left / right)."),
}

REFERENCES = [
    "Gray A.J., Shorter K., Cummins C., Murphy A. & Waldron M. (2018). Modelling movement energetics using Global "
    "Positioning System devices in contact team sports: limitations and solutions. Sports Med 48, 1357–1368.",
    "Cavagna G.A., Thys H. & Zamboni A. (1976). The sources of external work in level walking and running. "
    "J Physiol 262, 639–657.",
    "Buchheit M., Lopez Sagarra A., Boskovic A., Komino P., Norman D. & Hader K. (2026). GPS 3.0: from distance into "
    "zones toward better proxies of internal neuromuscular load in elite football. Sport Perf Sci Rep 280.",
    "Buchheit M. & Simpson B.M. (2017). Player-tracking technology: half-full or half-empty glass? Int J Sports "
    "Physiol Perform 12 (Suppl 2), S235–S241.",
    "Bundle M.W., Hoyt R.W. & Weyand P.G. (2003). High-speed running performance: a new approach to assessment and "
    "prediction. J Appl Physiol 95, 1955–1962.",
    "Buchheit M., Mendez-Villanueva A., Simpson B.M. & Bourdon P.C. (2010). Repeated-sprint sequences during youth "
    "soccer matches. Int J Sports Med 31, 709–716.",
    "Buchheit M., Settembre M., Hader K. & McHugh D. (2023). Exposures to near-to-maximal speed running bouts during "
    "different turnarounds in elite football. Biol Sport 40, 1057–1067.",
    "Spencer M. et al. (2004). Time-motion analysis of elite field hockey, with special reference to repeated-sprint "
    "activity. J Sports Sci 22, 843–850.",
    "Harper D.J., Carling C. & Kiely J. (2019). High-intensity acceleration and deceleration demands in elite team "
    "sports competitive match play. Sports Med 49, 1923–1947.",
    "Varley M.C., Elias G.P. & Aughey R.J. (2012). Current match-analysis techniques' underestimation of intense "
    "periods of high-velocity running. Int J Sports Physiol Perform 7, 183–185.",
    "Fereday K. et al. (2020). A comparison of rolling averages versus discrete time epochs for assessing the "
    "worst-case scenario locomotor demands of professional soccer match-play. J Sci Med Sport 23, 764–769.",
    "Delaney J.A., Thornton H.R., Rowell A.E. et al. (2018). Modelling the decrement in running intensity within "
    "professional soccer players. Sci Med Football 2, 86–92.",
    "Novak A.R., Impellizzeri F.M., Trivedi A., Coutts A.J. & McCall A. (2021). Analysis of the worst-case scenarios "
    "in an elite football team. J Sports Sci 39, 1850–1859.",
    "Whitehead S. et al. (2018). The use of microtechnology to quantify the peak match demands of the football "
    "codes: a systematic review. Sports Med 48, 2549–2575.",
]


def _dlabel(d: float) -> str:
    d = int(d)
    if d < 60:
        return f"{d} s"
    return f"{d // 60} min" + (f" {d % 60:02d}" if d % 60 else "")


def _mmss(s: float) -> str:
    if not np.isfinite(s):
        return "—"
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


def _figures(kin, curves, spr, lang, T, mw=None):
    import matplotlib.pyplot as plt

    mpl_theme()
    P = PALETTE
    out = {}
    c = curves["distance"]
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    ax.plot(c.durations, c.peak, "o", color=P["navy"], ms=5, label=T["rolling"])
    ax.plot(c.durations, c.fixed, "D", color=P["gold"], ms=4, label=T["fixed"])
    tt = np.logspace(np.log10(c.durations.min()), np.log10(c.durations.max()), 100)
    ax.plot(tt, c.predict(tt), color=P["red"], lw=1.4)
    ax.set_xscale("log"); ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ticks = [d for d in c.durations if d < 60 or d % 60 == 0]
    ax.set_xticks(ticks, [_dlabel(d) for d in ticks], fontsize=7)
    ax.set(ylabel="m/min"); ax.legend(fontsize=7)
    out["fig_mds"] = fig_to_uri(fig)

    fig, ax = plt.subplots(figsize=(7.4, 2.3))
    v = np.nan_to_num(kin.speed) * 60
    r = mds.rolling_mean(v, int(60 * kin.fs))
    r[~np.isfinite(kin.speed)] = np.nan
    tm = kin.time / 60
    for T_, col in ((300, P["gold_light"]), (60, P["gold"])):
        k = list(c.durations).index(T_) if T_ in c.durations else None
        if k is not None and np.isfinite(c.peak_start[k]):
            ax.axvspan(c.peak_start[k] / 60, (c.peak_start[k] + T_) / 60, color=col, alpha=.45, lw=0)
    ax.plot(tm, r, color=P["navy"], lw=1)
    for e in spr:
        ax.axvline(e.start / 60, ymin=0, ymax=.08, color=P["red"], lw=1.2)
    ax.set(xlabel="min", ylabel="m/min", xlim=(0, tm[-1]))
    out["fig_tl"] = fig_to_uri(fig)

    if mw is not None and kin.a_normal is not None:
        import matplotlib.pyplot as plt  # noqa: F811
        psi = np.arctan2(np.nan_to_num(kin.a_normal), np.nan_to_num(kin.a_tangential))
        w = np.nan_to_num(mw.power) / kin.fs
        bins = np.linspace(-np.pi, np.pi, 17)
        h, _ = np.histogram(psi, bins=bins, weights=w)
        fig = plt.figure(figsize=(3.4, 3.0))
        ax = fig.add_subplot(projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(1)            # counter-clockwise: left turns on the left
        cen = (bins[:-1] + bins[1:]) / 2
        cols = [P["navy"] if abs(c) < np.pi / 4 or abs(c) > 3 * np.pi / 4 else P["red"] for c in cen]
        bars = ax.bar(cen, h / h.sum() * 100, width=2 * np.pi / 16 * 0.92, color=cols, alpha=.85)
        for b in bars:                       # WeasyPrint mis-renders the polar clip path
            b.set_clip_on(False)
        ax.set_xticks([0, np.pi / 2, np.pi, -np.pi / 2],
                      ["accél." if lang == "fr" else "accel.", T["left"], "frein" if lang == "fr" else "brake", T["right"]])
        ax.set_yticklabels([])
        ax.grid(False)
        ax.spines["polar"].set_visible(False)
        out["fig_rose"] = fig_to_uri(fig)

    if kin.has_direction and mw is None:
        a = kin.a_complete
        m = (a >= 1) & np.isfinite(a)
        idx = np.flatnonzero(m)
        if idx.size > 6000:
            idx = idx[np.linspace(0, idx.size - 1, 6000).astype(int)]
        fig, ax = plt.subplots(figsize=(3.6, 2.6))
        ax.scatter(kin.a_tangential[idx], kin.a_centripetal[idx], s=2, color=P["navy"], alpha=.25, lw=0)
        lim = 8
        th = np.linspace(0, np.pi, 50)
        for rr in (2.5, 5):
            ax.plot(rr * np.cos(th), rr * np.sin(th), color=P["gold"], lw=.8, ls="--")
        ax.set(xlim=(-lim, lim), ylim=(0, lim), xlabel="a tangentielle (m/s²)" if lang == "fr" else "tangential a (m/s²)",
               ylabel="a centripète (m/s²)" if lang == "fr" else "centripetal a (m/s²)")
        ax.set_aspect("equal")
        out["fig_dir"] = fig_to_uri(fig)
    return out


def session_report(kin: Kinematics, out: str | Path, *, athlete: str | None = None, session: str | None = None,
                   session_date: str | None = None, mss: float | None = None, mas: float | None = None,
                   sprint_threshold: float = 7.0, hsr_threshold: float = 5.5, acc_threshold: float = 2.5,
                   durations=mds.DURATIONS, lang: str = "fr", organisation: str | None = None,
                   logo: str | Path | None = None, source: str | None = None) -> Path:
    """Write the PDF report of one session; ``mss`` (m/s) and ``mas`` (km/h... see below) are optional.

    ``mss`` is the maximal sprinting speed in m/s (near-to-MSS exposures);
    ``mas`` the maximal aerobic speed in km/h: with ``mss`` it gives the
    Bundle 1-min speed reference for the intensity exposure time.
    """
    if lang not in L:
        raise ValueError("lang must be 'fr' or 'en'")
    T = L[lang]
    f = lambda x, d=0: fmt(x, d, lang)
    spr = E.sprints(kin, sprint_threshold)
    hsr = E.sprints(kin, hsr_threshold)
    acc = E.accelerations(kin, acc_threshold)
    dec = E.decelerations(kin, acc_threshold)
    trn = E.turns(kin, acc_threshold)
    rss = E.repeated_sequences(hsr)
    rsb = E.repeated_sprint_bouts(hsr)
    metrics = ("distance", "hsr", "accdec") + (("power",) if kin.has_direction else ())
    curves = {m: mds.peak_curve(kin, m, durations, hsr=hsr_threshold, acc=acc_threshold) for m in metrics}
    mw = mechanical_work(kin) if kin.has_direction else None
    c = curves["distance"]
    cpl, npl, r2 = c.power_law()
    minutes = kin.duration / 60
    total = kin.distance()
    hsr_d = kin.distance(np.nan_to_num(kin.speed) >= hsr_threshold)
    i1 = list(c.durations).index(60) if 60 in c.durations else 0
    tiles = [
        {"val": f(total / 1000, 2), "unit": "km", "lab": T["total"], "sub": f"{f(total / minutes)} m/min · {f(minutes)} min", "accent": True},
        {"val": f(hsr_d), "unit": "m", "lab": T["hsr"], "sub": f"> {f(hsr_threshold, 1)} m/s · {len(hsr)} efforts"},
        {"val": str(len(spr)), "unit": "", "lab": T["sprints"], "sub": f"> {f(sprint_threshold, 1)} m/s · max {f(np.nanmax(kin.speed), 2)} m/s"},
        {"val": f"{len(acc)} / {len(dec)}", "unit": "", "lab": f'{T["accs"]} / {T["decs"]}', "sub": f"> {f(acc_threshold, 1)} m/s²"},
        {"val": f(c.peak[i1]), "unit": "m/min", "lab": T["peak1"], "sub": f"{T['at'].lower()} {_mmss(c.peak_start[i1])}"},
    ]
    mds_rows = []
    for k, d in enumerate(c.durations):
        mds_rows.append([_dlabel(d), f(c.peak[k]), f(curves['hsr'].peak[k]),
                         f(curves['accdec'].peak[k], 1),
                         f(curves['power'].peak[k], 2) if mw is not None else "—",
                         f(c.fixed_underestimation_pct[k], 1) + " %",
                         _mmss(c.peak_start[k]), f(c.following[k]), str(c.near_peak.get(float(d), 0))])
    bursts = [[_mmss(e.start), f(e.duration, 1) + " s", f(e.max_speed, 2), f(e.entry_speed, 2), f(e.distance)]
              for e in sorted(spr, key=lambda e: -e.max_speed)[:10]]
    seqs = [[_mmss(s.start), str(s.count), f(s.mean_recovery) + " s", f(sum(e.distance for e in s.efforts))]
            for s in sorted(rss, key=lambda s: -s.count)[:6]]
    near = {}
    if mss:
        near = {f"> {int(lv * 100)} % MSS": len(v) for lv, v in E.near_mss_exposures(kin, mss).items()}
    iet, ref_txt, mp_est = None, None, None
    if mas and mss:
        s1 = gps3.estimated_max_s1min(mas, mss * 3.6)              # km/h
        iet = gps3.intensity_exposure_time(kin, np.nan_to_num(kin.speed) * 3.6, s1)
        ref_txt = f"Estimated Maximal S1min = {f(s1, 1)} km/h ({f(s1 / 3.6 * 60)} m/min)"
        mp_est = gps3.estimated_max_mp1min(mas, mss * 3.6)
    iet_match = gps3.intensity_exposure_time(kin, np.nan_to_num(kin.speed) * 60, c.peak[i1])
    mech = None
    if mw is not None:
        cp = curves["power"]
        mp1 = cp.peak[i1]
        bursts_m = E.mechanical_bursts(kin, mw.power, 0.5 * np.nanmax(mds.rolling_mean(np.nan_to_num(mw.power), max(1, int(kin.fs)))))
        fam = {}
        for m_ in mw.manoeuvres:
            key = (m_.family, m_.side)
            d_ = fam.setdefault(key, [0, 0.0, 0.0])
            d_[0] += 1; d_[1] += m_.work; d_[2] += m_.w_direction
        tot_w = sum(v[1] for v in fam.values()) or 1
        rows = []
        for fam_name in FAMILIES:
            for side in ("", "left", "right"):
                if (fam_name, side) in fam:
                    n_, w_, wd_ = fam[(fam_name, side)]
                    rows.append([T["fam_names"][fam_name] + (f" ({T[side]})" if side else ""), str(n_),
                                 f(w_ / tot_w * 100) + " %", f(wd_ / w_ * 100 if w_ else 0) + " %"])
        mech = {
            "tiles": [
                {"val": f(mw.w_total / 1000, 2), "unit": "kJ/kg", "lab": T["w_tot"],
                 "sub": f"{f(mw.w_total / minutes, 1)} J/kg/min", "accent": True},
                {"val": f(mw.direction_share * 100), "unit": "%", "lab": T["w_dir"],
                 "sub": f"{f(mw.w_direction / max(sum(m_.work for m_ in mw.manoeuvres), 1e-9) * 100)} % "
                        + ("dans les manœuvres" if lang == "fr" else "within manoeuvres")},
                {"val": f(mp1, 2), "unit": "W/kg", "lab": T["mp1"], "sub": f"{T['at'].lower()} {_mmss(cp.peak_start[i1])}"},
                {"val": str(len(bursts_m)), "unit": "", "lab": T["mbursts"],
                 "sub": "≥ 50 % " + ("du pic 1 s" if lang == "fr" else "of the 1-s peak")},
            ],
            "rows": rows,
            "iet": gps3.intensity_exposure_time(kin, np.nan_to_num(mw.power), mp1),
            "ref": f(mp1, 2),
        }
    share = gps3.directional_share(kin)
    reading = []
    if lang == "fr":
        reading.append(f"{f(total)} m en {f(minutes)} min ({f(total / minutes)} m/min), dont {f(hsr_d)} m au-dessus de "
                       f"{f(hsr_threshold, 1)} m/s et <b>{len(spr)} sprints</b>.")
        reading.append(f"La minute la plus exigeante atteint <b>{f(c.peak[i1])} m/min</b> (à {_mmss(c.peak_start[i1])}), "
                       f"{f(c.peak[i1] / (total / minutes), 1)} fois la moyenne du match ; la fenêtre suivante retombe à "
                       f"{f(c.following[i1])} m/min. Des périodes fixes l'auraient sous-estimée de "
                       f"{f(c.fixed_underestimation_pct[i1], 0)} %.")
        reading.append(f"L'intensité de pointe décroît avec la durée selon une loi de puissance d'exposant "
                       f"{f(npl, 2)} (R² {f(r2, 2)}) : un repère pour programmer des tâches de durée différente.")
        if rss:
            reading.append(f"{len(rss)} séquences d'efforts à haute vitesse répétés (≤ 60 s de récupération), la plus "
                           f"longue de {max(s.count for s in rss)} efforts ; {len(rsb)} répondent au critère plus strict de "
                           f"Spencer (≥ 3, récupération moyenne < 21 s).")
        if mw is not None:
            reading.append(f"Changements de direction : <b>{f(mw.direction_share * 100)} %</b> du travail mécanique horizontal "
                           f"vient des virages et non des variations de vitesse — une part que les métriques GPS classiques "
                           f"ignorent (modèle ouvert, section 6).")
    else:
        reading.append(f"{f(total)} m in {f(minutes)} min ({f(total / minutes)} m/min), of which {f(hsr_d)} m above "
                       f"{f(hsr_threshold, 1)} m/s and <b>{len(spr)} sprints</b>.")
        reading.append(f"The most demanding minute reaches <b>{f(c.peak[i1])} m/min</b> (at {_mmss(c.peak_start[i1])}), "
                       f"{f(c.peak[i1] / (total / minutes), 1)} times the match average; the next window drops to "
                       f"{f(c.following[i1])} m/min. Fixed epochs would have underestimated it by "
                       f"{f(c.fixed_underestimation_pct[i1], 0)} %.")
        reading.append(f"Peak intensity falls with duration along a power law of exponent {f(npl, 2)} (R² {f(r2, 2)}): "
                       f"a reference for programming drills of different lengths.")
        if rss:
            reading.append(f"{len(rss)} sequences of repeated high-speed efforts (≤ 60 s recovery), the longest with "
                           f"{max(s.count for s in rss)} efforts; {len(rsb)} meet Spencer's stricter criterion (≥ 3, mean "
                           f"recovery < 21 s).")
        if mw is not None:
            reading.append(f"Direction changes: <b>{f(mw.direction_share * 100)} %</b> of the horizontal mechanical work comes "
                           f"from turning rather than from speed changes — a part classic GPS metrics ignore (open model, "
                           f"section 6).")
    fx = c.fixed_underestimation_pct
    ratio = c.peak / c.session_mean
    if lang == "fr":
        mds_notes = [f"Sur 15 s, le pic vaut {f(ratio[0], 1)} fois la moyenne du match ; sur 10 min, {f(ratio[-1], 1)} fois.",
                     f"Des périodes fixes sous-estiment le pic de {f(np.nanmin(fx), 1)} à {f(np.nanmax(fx), 1)} % selon la durée.",
                     "Un pic isolé est instable d'un match à l'autre : regarder aussi la fenêtre suivante et le nombre de "
                     "passages proches du pic."]
    else:
        mds_notes = [f"Over 15 s the peak is {f(ratio[0], 1)} times the match average; over 10 min, {f(ratio[-1], 1)} times.",
                     f"Fixed epochs underestimate the peak by {f(np.nanmin(fx), 1)} to {f(np.nanmax(fx), 1)} % depending on duration.",
                     "A single peak is unstable from match to match: look at the next window and at how often near-peak "
                     "passages recur too."]
    method = {
        "fr": [
            "Positions lissées et dérivées par un filtre de Savitzky-Golay (fenêtre 0,6 s) ; sauts de tracking retirés. "
            "L'accélération complète est décomposée en composante tangentielle (variation de vitesse) et centripète "
            "(virage) ; |a|² = a<sub>t</sub>² + a<sub>c</sub>².",
            "Efforts : passages au-dessus d'un seuil tenus au moins 1 s (vitesse) ou 0,5 s (accélération). Séquences "
            "répétées : au moins 2 efforts séparés de 60 s au plus (Buchheit et al. 2010) ; critère de Spencer et al. (2004) : "
            "au moins 3 efforts, récupération moyenne < 21 s.",
            "Scénarios les plus exigeants : pic de la moyenne glissante pour chaque durée, comparé aux périodes fixes "
            "(Varley et al. 2012 ; Fereday et al. 2020), loi de puissance I = c·t<sup>n</sup> (Delaney et al. 2018). La valeur "
            "de pic seule est instable (Novak et al. 2021) : elle est donnée avec son moment, la fenêtre suivante et le "
            "nombre de passages à 90 % du pic.",
            "GPS 3.0 (Buchheit et al. 2026) : temps d'exposition (IET) = minutes où la moyenne glissante sur 60 s dépasse "
            "60, 70 ou 80 % d'une référence — le pic de 1 min du match, et si VMA et VMS sont connues, la vitesse maximale "
            "sur 1 min du modèle de Bundle (VMA + 0,695·(VMS − VMA)). Le travail mécanique du papier vient d'un "
            "analyseur propriétaire non publié : il n'est pas reproduit ici.",
            "Travail directionnel (modèle ouvert) : le travail lié à la vitesse est la somme absolue des variations "
            "d'énergie cinétique du centre de masse (Gray et al. 2018), après un filtre à hystérésis de 0,2 m/s qui écarte "
            "le bruit du tracking. Le travail lié à la direction compte l'énergie cinétique le long et en travers du cap "
            "d'entrée de chaque manœuvre (comptabilité par axe, Cavagna et al. 1976) : un virage de 90° à la vitesse v "
            "coûte v² par kg, un arrêt-relance seulement du travail de vitesse. Horizontal, centre de masse seul ; non "
            "comparable aux watts d'ADI."],
        "en": [
            "Positions smoothed and differentiated with a Savitzky-Golay filter (0.6-s window); tracking jumps removed. "
            "The complete acceleration is split into a tangential (speed change) and a centripetal (turning) "
            "component; |a|² = a<sub>t</sub>² + a<sub>c</sub>².",
            "Efforts: passages above a threshold held for at least 1 s (speed) or 0.5 s (acceleration). Repeated "
            "sequences: at least 2 efforts no more than 60 s apart (Buchheit et al. 2010); Spencer et al. (2004) "
            "criterion: at least 3 efforts, mean recovery < 21 s.",
            "Most demanding scenarios: peak of the rolling average for each duration, compared with fixed epochs "
            "(Varley et al. 2012; Fereday et al. 2020), power law I = c·t<sup>n</sup> (Delaney et al. 2018). The peak alone is "
            "unstable (Novak et al. 2021): it is given with its time, the next window and the number of passages at "
            "90 % of the peak.",
            "GPS 3.0 (Buchheit et al. 2026): intensity exposure time (IET) = minutes during which the 60-s rolling "
            "average exceeds 60, 70 or 80 % of a reference — the match peak minute and, when MAS and MSS are known, "
            "Bundle's maximal 1-min speed (MAS + 0.695·(MSS − MAS)). The paper's mechanical work comes from a "
            "proprietary analyser that is not published: it is not reproduced here.",
            "Direction-sensitive work (open model): speed-change work is the absolute sum of the changes in kinetic "
            "energy of the centre of mass (Gray et al. 2018), after a 0.2 m/s hysteresis filter that keeps tracking noise "
            "out. Direction-change work counts kinetic energy along and across the heading at the start of each "
            "manoeuvre (per-axis accounting, Cavagna et al. 1976): a 90° turn at speed v costs v² per kg, a stop-and-go "
            "cut only speed work. Horizontal, centre of mass only; not comparable with ADI watts."],
    }[lang]
    ctx = {
        "lang": lang, "L": T, "title": T["title"], "title_html": T["title_html"], "kicker": T["kicker"],
        "athlete": athlete, "test": session, "test_date": session_date,
        "extra_meta": ({("VMS" if lang == "fr" else "MSS"): f"{f(mss, 2)} m/s"} if mss else {}),
        "organisation": organisation, "logo": logo, "tool": "peakdemands", "version": __version__,
        "tiles": tiles, "reading": reading, "mds_notes": mds_notes, "mech": mech, "mds_rows": mds_rows, "bursts": bursts, "seqs": seqs, "near": near,
        "counts": [(T["sprints"], len(spr)), ("> " + f(hsr_threshold, 1) + " m/s", len(hsr)), (T["accs"], len(acc)),
                   (T["decs"], len(dec)), (T["turns"], len(trn) if kin.has_direction else "—"),
                   (T["rss"], len(rss)), ("Spencer", len(rsb))],
        "iet": iet, "iet_match": iet_match, "ref_txt": ref_txt, "mp_est": f(mp_est, 1) if mp_est else None,
        "share": f(share * 100) + " %" if np.isfinite(share) else None, "has_dir": kin.has_direction,
        "pl": (f(cpl), f(npl, 3), f(r2, 3)), "peak1": f(c.peak[i1]), "method": method, "references": REFERENCES,
        "source": source,
        "disclaimer": ("Seuils et fenêtres sont des paramètres : les comparaisons ne valent qu'à méthode égale."
                       if lang == "fr" else "Thresholds and windows are parameters: compare only at equal method."),
        **_figures(kin, curves, spr, lang, T, mw),
    }
    return render_pdf(TEMPLATES, "session.html", ctx, out)


__all__ = ["session_report"]
