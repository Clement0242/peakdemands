"""Single-player match report (Metrica Sports sample data, game 2, home team, player 5)."""

from pathlib import Path

from peakdemands.match import load_metrica_match
from peakdemands.report.player import player_report

DOCS = Path(__file__).resolve().parents[1] / "docs"
m = load_metrica_match(game=2, team="Home")
for lang in ("fr", "en"):
    player_report(m, "Player5", DOCS / f"example_player_report_{lang}.pdf", lang=lang)
print("ok")
