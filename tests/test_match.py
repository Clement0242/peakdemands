"""Smoke test of the match pipeline on the Metrica sample (skipped when the data is not available offline)."""

import os
from pathlib import Path

import numpy as np
import pytest

from peakdemands.datasets import _cache


def _have_data():
    c = _cache()
    return (c / "metrica_g2_Home.csv").exists() and (c / "metrica_g2_events.csv").exists() or os.environ.get("PEAKDEMANDS_NET")


@pytest.mark.skipif(not _have_data(), reason="Metrica sample data not cached")
def test_match_pipeline(tmp_path):
    from peakdemands.match import load_metrica_match, phase_table, player_table
    m = load_metrica_match()
    assert 11 <= len(m.players) <= 16
    ph = phase_table(m)
    assert sum(v["share"] for v in ph.values()) == pytest.approx(100, abs=3)
    assert ph["ball_dead"]["m_per_min"] < ph["in_possession"]["m_per_min"]
    rows = player_table(m)
    full = [r for r in rows if r["full"] and r["name"] != m.goalkeeper]
    assert 8000 < np.mean([r["distance"] for r in full]) < 13000
    pytest.importorskip("weasyprint")
    from peakdemands.report.match import match_report
    assert match_report(m, tmp_path / "m.pdf").read_bytes()[:4] == b"%PDF"
