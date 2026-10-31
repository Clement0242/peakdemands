"""Full physical-preparation match report of a real match (Metrica Sports sample data, game 2, home team)."""

from pathlib import Path

from peakdemands.match import load_metrica_match
from peakdemands.report.match import match_report

DOCS = Path(__file__).resolve().parents[1] / "docs"
m = load_metrica_match(game=2, team="Home")
match_report(m, DOCS / "example_match_report_fr.pdf", lang="fr")
match_report(m, DOCS / "example_match_report_en.pdf", lang="en")
print("ok")
