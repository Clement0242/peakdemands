"""Example reports and figures of the project page (docs/), from a real match.

Data: Metrica Sports sample data (game 2, home team), downloaded on first use.
"""

from pathlib import Path

import numpy as np

import peakdemands as pk
from peakdemands.report import session_report

DOCS = Path(__file__).resolve().parents[1] / "docs"
d = pk.load_metrica("Player9")
k = pk.from_positions(d["time"], d["x"], d["y"])
for lang, who, what in (("fr", "Joueur 9 (anonymisé)", "Match complet"), ("en", "Player 9 (anonymised)", "Full match")):
    session_report(k, DOCS / f"example_match_{lang}.pdf", athlete=who, session=what, mss=9.2, mas=17.0,
                   lang=lang, source=d["source"])

# Figures for the page
import matplotlib.pyplot as plt
from peakdemands.report.engine import PALETTE as P, mpl_theme

mpl_theme()
plt.rcParams["font.size"] = 11
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for name, col in (("Player9", P["navy"]), ("Player7", P["red"]), ("Player1", P["gold"])):
    dd = pk.load_metrica(name)
    kk = pk.from_positions(dd["time"], dd["x"], dd["y"])
    c = pk.peak_curve(kk, "distance")
    axes[0].plot(c.durations, c.peak, "o", color=col, label=name)
    tt = np.logspace(np.log10(15), np.log10(600), 100)
    axes[0].plot(tt, c.predict(tt), color=col, lw=1.2)
    axes[0].plot(c.durations, c.fixed, "D", color=col, ms=4, alpha=.45)
axes[0].set_xscale("log"); axes[0].xaxis.set_minor_formatter(plt.NullFormatter())
axes[0].set_xticks([15, 30, 60, 120, 300, 600], ["15 s", "30 s", "1 min", "2 min", "5 min", "10 min"])
axes[0].set(ylabel="peak distance (m/min)", title="Most demanding scenarios: rolling (dots) vs fixed (diamonds)")
axes[0].legend()
a = k.a_complete
m = np.flatnonzero((a >= 1) & np.isfinite(a))
axes[1].scatter(k.a_tangential[m], k.a_centripetal[m], s=2, alpha=.2, color=P["navy"], lw=0)
th = np.linspace(0, np.pi, 50)
for r in (2.5, 5):
    axes[1].plot(r * np.cos(th), r * np.sin(th), "--", color=P["gold"], lw=1)
axes[1].set(xlim=(-8, 8), ylim=(0, 8), xlabel="tangential acceleration (m/s²)", ylabel="centripetal acceleration (m/s²)",
            title=f"Complete acceleration: {pk.directional_share(k):.0%} from turning")
axes[1].set_aspect("equal")
fig.tight_layout()
fig.savefig(DOCS / "mds_direction.png", dpi=130)
print("ok")
